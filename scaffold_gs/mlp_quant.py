"""Per-channel MLP weight quantization + static arithmetic entropy coding.

The PHG size accounting currently charges decoder MLP weights at 32 bit/param
(``bit_mlp = params * 32``). This module quantizes those weights to N bits with
per-output-channel scales and compresses the integer indices with a static
32-bit arithmetic coder (range coder), so the transmitted MLP payload becomes
``entropy_coded_indices + scales`` instead of raw float32.

Quantization is symmetric (zero-point 0), which is the standard choice for
weights:

    q = round(w / scale)            scale = max(|w|) per output channel
    w_hat = q * scale

The caller must re-run the attribute codec with the dequantized weights
(they participate in ``mlp_grid`` / ``mlp_deform`` / ``mlp_complexity`` and the
decoder MLPs), then decode and evaluate -- you cannot just swap the size line.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

import torch

# Static arithmetic-coder (range coder) parameters.
AC_BITS = 32
AC_TOP = 1 << AC_BITS
AC_HALF = AC_TOP >> 1
AC_QUARTER = AC_HALF >> 1
AC_THREE_Q = AC_HALF + AC_QUARTER
AC_FREQ_BITS = 16
AC_FREQ_TOTAL = 1 << AC_FREQ_BITS

MLP_GROUPS = (
    "mlp_opacity",
    "mlp_cov",
    "mlp_color2",
    "mlp_color",
    "mlp_asg",
    "mlp_grid",
    "mlp_deform",
    "mlp_complexity",
)


def mlp_param_items(core) -> Iterable[Tuple[str, torch.Tensor]]:
    """Yield (name, parameter) for every MLP parameter of the HAC++ core."""
    for name, param in core.named_parameters():
        if "mlp" in name:
            yield name, param


def _group_of(name: str) -> str:
    module_name = name.split(".", 1)[0]
    for g in MLP_GROUPS:
        if module_name == g or name.startswith(g + "."):
            return g
    return "other"


def _quantize_tensor(w: torch.Tensor, bits: int, per_channel: bool):
    """Symmetric per-output-channel quantization.

    Returns (q int16 tensor, scale tensor). For biases (1-D) per_channel is
    effectively per-tensor (one channel).
    """
    w = w.detach()
    shape = w.shape
    if per_channel and w.dim() >= 2:
        flat = w.reshape(w.shape[0], -1)
        amax = flat.abs().amax(dim=1).clamp_min(1e-12)
    else:
        amax = w.abs().amax().reshape(1).clamp_min(1e-12)
    qmax = float(2 ** (bits - 1) - 1)
    scale = amax / qmax
    if per_channel and w.dim() >= 2:
        q = torch.round(w / scale.view(-1, *([1] * (w.dim() - 1))))
    else:
        q = torch.round(w / scale[0])
    q = q.clamp(-qmax, qmax).to(torch.int16)
    return q, scale


def quantize_core_mlps(
    core,
    bits: int | None = None,
    groups: Tuple[str, ...] = MLP_GROUPS,
    per_channel: bool = True,
    bits_map: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, object]]:
    """Quantize selected MLP groups in-place and return metadata.

    ``bits`` applies to every group in ``groups``; alternatively ``bits_map``
    maps a group name to its bit width (per-MLP mixed precision), e.g.
    ``{"mlp_complexity": 8, "mlp_deform": 8, "mlp_opacity": 16}``. MLPs not in
    ``groups`` / not in ``bits_map`` stay at float32.

    The parameter values are replaced by the dequantized approximation, so a
    subsequent ``codec.encode`` uses exactly the quantized network.
    """
    meta: Dict[str, Dict[str, object]] = {}
    for name, param in mlp_param_items(core):
        group = _group_of(name)
        if bits_map is not None:
            b = bits_map.get(group)
            if b is None:
                continue
        else:
            if group not in groups:
                continue
            b = bits
        if b is None:
            continue
        q, scale = _quantize_tensor(param.data, b, per_channel)
        if scale.numel() == 1:
            dequant = q.float() * scale[0]
        else:
            dequant = q.float() * scale.view(-1, *([1] * (q.dim() - 1)))
        param.data.copy_(dequant)
        meta[name] = {
            "q": q.detach().cpu(),
            "scale": scale.detach().cpu(),
            "bits": b,
            "group": group,
        }
    return meta


def dequantize_meta(meta: Dict[str, Dict[str, object]]) -> Dict[str, torch.Tensor]:
    """Rebuild the dequantized weights from quantization metadata."""
    out = {}
    for name, m in meta.items():
        q = m["q"]
        scale = m["scale"]
        if scale.numel() == 1:
            out[name] = q.float() * scale[0]
        else:
            out[name] = q.float() * scale.view(-1, *([1] * (q.dim() - 1)))
    return out


# ---------------------------------------------------------------------------
# Static arithmetic coding (32-bit range coder, bit-packed output)
# ---------------------------------------------------------------------------


def _build_table(values: torch.Tensor) -> Tuple[List[int], List[int], Dict[int, int], int]:
    vals, counts = torch.unique(values, return_counts=True)
    vals_l = [int(v) for v in vals.tolist()]
    counts_l = [int(c) for c in counts.tolist()]
    total = sum(counts_l)
    if len(vals_l) == 1:
        return vals_l, counts_l, {vals_l[0]: 0}, total
    # Normalize frequencies so the cumulative total is exactly AC_FREQ_TOTAL.
    scaled = [max(1, int(c * AC_FREQ_TOTAL / total)) for c in counts_l]
    s = sum(scaled)
    i = 0
    n = len(scaled)
    while s < AC_FREQ_TOTAL:
        scaled[i % n] += 1
        s += 1
        i += 1
    while s > AC_FREQ_TOTAL:
        j = max(range(n), key=lambda k: (scaled[k], -k))
        if scaled[j] > 1:
            scaled[j] -= 1
            s -= 1
        else:
            break
    cdf = {}
    acc = 0
    for v, c in zip(vals_l, scaled):
        cdf[v] = acc
        acc += c
    return vals_l, scaled, cdf, acc


class _BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, bit: int) -> None:
        self.acc = (self.acc << 1) | bit
        self.n += 1
        if self.n == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.n = 0

    def flush(self) -> bytes:
        if self.n:
            self.buf.append(self.acc << (8 - self.n))
            self.n = 0
        return bytes(self.buf)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.acc = 0
        self.n = 0

    def read(self) -> int:
        if self.n == 0:
            if self.pos >= len(self.data):
                return 0  # pad with zeros at end of stream
            self.acc = self.data[self.pos]
            self.pos += 1
            self.n = 8
        bit = (self.acc >> 7) & 1
        self.acc = (self.acc << 1) & 0xFF
        self.n -= 1
        return bit


class _ArithEncoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = AC_TOP - 1
        self.pending = 0
        self.writer = _BitWriter()

    def _emit(self, bit: int) -> None:
        self.writer.write(bit)
        for _ in range(self.pending):
            self.writer.write(1 - bit)
        self.pending = 0

    def encode(self, cum: int, freq: int) -> None:
        r = self.high - self.low + 1
        self.high = self.low + (r * (cum + freq) // AC_FREQ_TOTAL) - 1
        self.low = self.low + (r * cum // AC_FREQ_TOTAL)
        while True:
            if self.high < AC_HALF:
                self._emit(0)
            elif self.low >= AC_HALF:
                self._emit(1)
                self.low -= AC_HALF
                self.high -= AC_HALF
            elif self.low >= AC_QUARTER and self.high < AC_THREE_Q:
                self.pending += 1
                self.low -= AC_QUARTER
                self.high -= AC_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        if self.low < AC_QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.writer.flush()


class _ArithDecoder:
    def __init__(self, data: bytes) -> None:
        self.low = 0
        self.high = AC_TOP - 1
        self.reader = _BitReader(data)
        self.code = 0
        for _ in range(AC_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode(self, cum: int, freq: int) -> None:
        r = self.high - self.low + 1
        target = ((self.code - self.low + 1) * AC_FREQ_TOTAL - 1) // r
        self.high = self.low + (r * (cum + freq) // AC_FREQ_TOTAL) - 1
        self.low = self.low + (r * cum // AC_FREQ_TOTAL)
        while True:
            if self.high < AC_HALF:
                pass
            elif self.low >= AC_HALF:
                self.low -= AC_HALF
                self.high -= AC_HALF
                self.code -= AC_HALF
            elif self.low >= AC_QUARTER and self.high < AC_THREE_Q:
                self.low -= AC_QUARTER
                self.high -= AC_QUARTER
                self.code -= AC_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.reader.read()


def arith_encode(values: torch.Tensor) -> Tuple[bytes, List[int], List[int]]:
    """Encode integer symbols with a static arithmetic coder.

    Returns (byte stream, value list, count list). The table must be sent as
    side information; the coder itself is lossless.
    """
    vals_l, counts_l, cdf, total = _build_table(values)
    if len(vals_l) == 1:
        return b"", vals_l, counts_l
    enc = _ArithEncoder()
    for v in values.tolist():
        enc.encode(cdf[v], counts_l[vals_l.index(v)])
    return enc.finish(), vals_l, counts_l


def arith_decode(
    data: bytes,
    n: int,
    vals_l: List[int],
    counts_l: List[int],
) -> List[int]:
    """Decode ``n`` symbols encoded by :func:`arith_encode`."""
    if len(vals_l) == 1:
        return [vals_l[0]] * n
    cdf = {}
    acc = 0
    for v, c in zip(vals_l, counts_l):
        cdf[v] = acc
        acc += c
    dec = _ArithDecoder(data)
    out = []
    for _ in range(n):
        sym = None
        for v in vals_l:
            r = dec.high - dec.low + 1
            target = ((dec.code - dec.low + 1) * AC_FREQ_TOTAL - 1) // r
            cum = cdf[v]
            freq = counts_l[vals_l.index(v)]
            if cum <= target < cum + freq:
                sym = v
                break
        if sym is None:
            raise ValueError("arithmetic decode: no symbol for target")
        dec.decode(cdf[sym], counts_l[vals_l.index(sym)])
        out.append(sym)
    return out


def save_mlp_payload(meta: Dict[str, Dict[str, object]], out_dir) -> Dict[str, int]:
    """Write quantized indices + scales/table metadata to ``out_dir``.

    - bits >= 16: raw int16 indices (frequency tables would cost more than the
      symbols themselves);
    - bits < 16: static arithmetic coder + per-tensor value/count tables.

    Returns byte counts for size accounting (``bits`` per file).
    """
    import os
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / "mlp_quant.bin"
    meta_path = out_dir / "mlp_quant_meta.json"
    serializable = {}
    with open(bin_path, "wb") as f:
        for name, m in meta.items():
            f.write(len(name).to_bytes(2, "little"))
            f.write(name.encode())
            q = m["q"].reshape(-1)
            f.write(int(q.numel()).to_bytes(4, "little"))
            if m["bits"] >= 16:
                f.write(q.numpy().astype("<i2").tobytes())
                vals, counts = [], []
            else:
                data, vals, counts = arith_encode(q)
                f.write(len(data).to_bytes(4, "little"))
                f.write(data)
                f.write(len(vals).to_bytes(4, "little"))
                f.write(
                    torch.tensor(vals, dtype=torch.int16)
                    .numpy()
                    .astype("<i2")
                    .tobytes()
                )
                f.write(
                    torch.tensor(counts, dtype=torch.int32)
                    .numpy()
                    .astype("<i4")
                    .tobytes()
                )
            serializable[name] = {
                "scale": m["scale"].tolist(),
                "shape": list(m["q"].shape),
                "bits": m["bits"],
                "group": m["group"],
            }
    with open(meta_path, "w") as f:
        json.dump(serializable, f, sort_keys=True)
    return {
        "mlp_quant_bin_bytes": bin_path.stat().st_size,
        "mlp_quant_meta_bytes": meta_path.stat().st_size,
    }


def self_test() -> None:
    torch.manual_seed(0)
    for bits in (16, 8, 6, 4):
        w = torch.randn(7, 13) * 3.0
        q, scale = _quantize_tensor(w, bits, True)
        data, vals, counts = arith_encode(q.reshape(-1))
        syms = arith_decode(data, q.numel(), vals, counts)
        assert syms == q.reshape(-1).tolist(), (
            f"arithmetic roundtrip failed for {bits}-bit"
        )
        deq = q.float() * scale.view(-1, 1)
        err = (deq - w).abs().max().item()
        print(f"self_test bits={bits} bytes={len(data)} max_err={err:.4f}")
    print("mlp_quant self_test OK")


if __name__ == "__main__":
    self_test()
