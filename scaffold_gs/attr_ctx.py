"""Optional R4 conditional entropy adjustment (P0-1 style) for scaling/offsets.

R4 is a post-training codec-side predictor: given the already-decoded
``feat_q`` (and, for offsets, ``scaling_q`` + masks), it adjusts the
``mlp_grid`` Gaussian mean/log-scale so the raw quantized symbols are coded
with a tighter distribution. It does NOT change reconstruction values, so
decode quality is identical; only the bitstream size changes.

The predictor weights are transmitted as a per-channel 16-bit quantization +
static arithmetic-coded payload (``attr_ctx.bin`` + ``attr_ctx_meta.json``),
and the payload byte count is charged to ``total_MB``.

Stage-A evidence (4-28 90k h32 l0p002, validation 20%): scaling+offsets gain
~11.9%, net saving after 16-bit predictor ~0.009 MB (0.11% of total_MB).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .mlp_quant import _quantize_tensor, arith_decode, arith_encode

ATTR_CTX_BITS = 8
ATTR_CTX_BIN = "attr_ctx.bin"
ATTR_CTX_META = "attr_ctx_meta.json"


def _zero_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class AttrCtxPredictor(nn.Module):
    """Two small MLPs adjusting scaling and offsets entropy parameters."""

    def __init__(
        self,
        feat_dim: int,
        grid_ctx_dim: int,
        n_offsets: int,
        hidden: int = 64,
        use_offsets: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.grid_ctx_dim = int(grid_ctx_dim)
        self.n_offsets = int(n_offsets)
        self.hidden = int(hidden)
        self.use_offsets = bool(use_offsets)
        k3 = 3 * self.n_offsets
        # scaling: mean_s(6) + scale_s(6) + feat_q + ctx -> mean_adj(6) + log_scale_adj(6)
        self.net_s = _zero_mlp(12 + self.feat_dim + self.grid_ctx_dim, self.hidden, 12)
        # offsets: mean_o(3K) + scale_o(3K) + feat_q + scaling_q(6) + masks(K) + ctx
        self.net_o = (
            _zero_mlp(
                2 * k3 + self.feat_dim + 6 + self.n_offsets + self.grid_ctx_dim,
                self.hidden,
                2 * k3,
            )
            if self.use_offsets
            else None
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _scaling_input(mean_s, scale_s, feat_q, ctx):
    return torch.cat([mean_s, scale_s, feat_q, ctx], dim=-1)


def _offsets_input(mean_o, scale_o, feat_q, scaling_q, masks, ctx):
    return torch.cat(
        [mean_o, scale_o, feat_q, scaling_q, masks.to(dtype=mean_o.dtype), ctx],
        dim=-1,
    )


def adjust_scaling(pred: AttrCtxPredictor, mean_s, scale_s, feat_q, ctx):
    """Return adjusted (mean, scale) for scaling, same shapes as inputs."""
    adj = pred.net_s(_scaling_input(mean_s, scale_s, feat_q, ctx))
    mean = mean_s + adj[:, :6]
    scale = scale_s * torch.exp(adj[:, 6:12].clamp(-1.4, 1.4))
    return mean, scale.clamp(min=1e-9)


def adjust_offsets(pred: AttrCtxPredictor, mean_o, scale_o, feat_q, scaling_q, masks, ctx):
    """Return adjusted (mean, scale) for offsets, same shapes as inputs."""
    if pred.net_o is None:
        return mean_o, scale_o
    n = ctx.shape[0]
    k3 = 3 * pred.n_offsets
    mean_o = mean_o.reshape(n, k3)
    scale_o = scale_o.reshape(n, k3)
    scaling_q = scaling_q.reshape(n, 6)
    masks = masks.reshape(n, pred.n_offsets)
    adj = pred.net_o(
        _offsets_input(mean_o, scale_o, feat_q, scaling_q, masks, ctx)
    )
    mean = mean_o + adj[:, :k3]
    scale = scale_o * torch.exp(adj[:, k3:].clamp(-1.4, 1.4))
    return mean.reshape(-1), scale.reshape(-1).clamp(min=1e-9)


# ---------------------------------------------------------------------------
# Payload (per-channel 16-bit + static arithmetic coding)
# ---------------------------------------------------------------------------


def save_attr_ctx_payload(pred: AttrCtxPredictor, out_dir) -> int:
    """Write quantized predictor weights; returns total payload bytes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / ATTR_CTX_BIN
    meta_path = out_dir / ATTR_CTX_META
    meta = {
        "version": 1,
        "bits": ATTR_CTX_BITS,
        "feat_dim": pred.feat_dim,
        "grid_ctx_dim": pred.grid_ctx_dim,
        "n_offsets": pred.n_offsets,
        "hidden": pred.hidden,
        "use_offsets": pred.use_offsets,
        "bits": ATTR_CTX_BITS,
    }
    with open(bin_path, "wb") as f:
        for name, w in pred.state_dict().items():
            q, scale = _quantize_tensor(w, ATTR_CTX_BITS, True)
            data, vals, counts = arith_encode(q.reshape(-1))
            nb = name.encode()
            f.write(len(nb).to_bytes(2, "little"))
            f.write(nb)
            f.write(int(w.numel()).to_bytes(4, "little"))
            f.write(len(w.shape).to_bytes(1, "little"))
            for d in w.shape:
                f.write(int(d).to_bytes(4, "little"))
            f.write(int(scale.numel()).to_bytes(4, "little"))
            f.write(scale.detach().cpu().numpy().astype("<f4").tobytes())
            f.write(len(data).to_bytes(4, "little"))
            f.write(data)
            f.write(len(vals).to_bytes(4, "little"))
            f.write(
                torch.tensor(vals, dtype=torch.int16)
                .numpy()
                .astype("<i2")
                .tobytes()
            )
            f.write(len(counts).to_bytes(4, "little"))
            f.write(
                torch.tensor(counts, dtype=torch.int32)
                .numpy()
                .astype("<i4")
                .tobytes()
            )
    with open(meta_path, "w") as f:
        json.dump(meta, f, sort_keys=True)
    return int(bin_path.stat().st_size + meta_path.stat().st_size)


def load_attr_ctx_payload(artifact_dir, device) -> AttrCtxPredictor:
    """Rebuild the predictor from ``attr_ctx.bin``/``attr_ctx_meta.json``."""
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / ATTR_CTX_META).read_text())
    pred = AttrCtxPredictor(
        meta["feat_dim"],
        meta["grid_ctx_dim"],
        meta["n_offsets"],
        meta["hidden"],
        meta.get("use_offsets", True),
    ).to(device)
    sd: Dict[str, torch.Tensor] = {}
    with open(artifact_dir / ATTR_CTX_BIN, "rb") as f:
        while True:
            hdr = f.read(2)
            if not hdr:
                break
            name_len = int.from_bytes(hdr, "little")
            name = f.read(name_len).decode()
            numel = int.from_bytes(f.read(4), "little")
            ndim = f.read(1)[0]
            shape = tuple(int.from_bytes(f.read(4), "little") for _ in range(ndim))
            nscale = int.from_bytes(f.read(4), "little")
            scale = torch.frombuffer(
                f.read(4 * nscale), dtype=torch.float32
            ).clone()
            ndata = int.from_bytes(f.read(4), "little")
            data = f.read(ndata)
            nvals = int.from_bytes(f.read(4), "little")
            vals = torch.frombuffer(
                f.read(2 * nvals), dtype=torch.int16
            ).tolist()
            ncounts = int.from_bytes(f.read(4), "little")
            counts = torch.frombuffer(
                f.read(4 * ncounts), dtype=torch.int32
            ).tolist()
            q = torch.tensor(
                arith_decode(data, numel, vals, counts), dtype=torch.float32
            ).view(shape)
            if scale.numel() == 1:
                deq = q * scale[0]
            else:
                deq = q * scale.view(-1, *([1] * (q.dim() - 1)))
            sd[name] = deq
    pred.load_state_dict(sd)
    return pred


def load_attr_ctx(path, device) -> AttrCtxPredictor:
    """Load a predictor saved by ``scripts/fit_attr_ctx.py`` (float state)."""
    obj = torch.load(path, map_location=device, weights_only=False)
    meta = obj["meta"]
    pred = AttrCtxPredictor(
        meta["feat_dim"],
        meta["grid_ctx_dim"],
        meta["n_offsets"],
        meta["hidden"],
        meta.get("use_offsets", True),
    ).to(device)
    pred.load_state_dict(obj["state_dict"])
    return pred


def quantize_attr_ctx_inplace(pred: AttrCtxPredictor, bits: int = ATTR_CTX_BITS):
    """Replace predictor weights with their per-channel ``bits``-bit dequant.

    Encode must use exactly the weights that decode rebuilds from the payload,
    otherwise the adjusted entropy parameters diverge and the bitstream does
    not round-trip. Call this after loading the fitted predictor and before
    any encode-side use; ``save_attr_ctx_payload`` then writes the same q/scales.
    """
    sd = {}
    for name, w in pred.state_dict().items():
        q, scale = _quantize_tensor(w, bits, True)
        if scale.numel() == 1:
            sd[name] = q.float() * scale[0]
        else:
            sd[name] = q.float() * scale.view(-1, *([1] * (q.dim() - 1)))
    pred.load_state_dict(sd)
    return pred


# ---------------------------------------------------------------------------
# Fitting (offline, on the codec-order train split)
# ---------------------------------------------------------------------------


def _gaussian_bin(x, mean, scale, q):
    dist = torch.distributions.Normal(mean, scale.clamp_min(1e-6))
    return (dist.cdf(x + 0.5 * q) - dist.cdf(x - 0.5 * q)).clamp_min(1e-12)


def _fit_adjust_net(
    net: nn.Sequential,
    x_tr,
    base_mean_tr,
    base_scale_tr,
    y_tr,
    q_tr,
    mask_tr,
    x_va,
    base_mean_va,
    base_scale_va,
    y_va,
    q_va,
    mask_va,
    out_dim: int,
    steps: int,
    weight_decay: float,
    device: torch.device,
):
    """Fit one adjustment MLP with zero-init start, early-stop on val bits."""
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    n_tr = x_tr.shape[0]

    def _bits(x, base_mean, base_scale, y, q, mask):
        adj = net(x)
        mean = base_mean + adj[:, :out_dim]
        scale = base_scale * torch.exp(
            adj[:, out_dim : 2 * out_dim].clamp(-1.4, 1.4)
        )
        if mask is not None:
            mean = mean[mask]
            scale = scale[mask]
            y = y[mask]
            q = q[mask]
        p = _gaussian_bin(y, mean, scale, q)
        return (-torch.log2(p)).sum(), adj

    with torch.no_grad():
        bits0, _ = _bits(x_va, base_mean_va, base_scale_va, y_va, q_va, mask_va)
        best_val = float(bits0.item())
        best_state = {k: v.clone() for k, v in net.state_dict().items()}

    for it in range(steps):
        idx = torch.randperm(n_tr, device=device)[: min(n_tr, 8192)]
        bits, adj = _bits(
            x_tr[idx],
            base_mean_tr[idx],
            base_scale_tr[idx],
            y_tr[idx],
            q_tr[idx],
            mask_tr[idx] if mask_tr is not None else None,
        )
        loss = bits / max(float(y_tr[idx].numel()), 1.0) + 0.1 * adj.abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 50 == 49 or it == steps - 1:
            with torch.no_grad():
                val = float(
                    _bits(
                        x_va, base_mean_va, base_scale_va, y_va, q_va, mask_va
                    )[0].item()
                )
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    with torch.no_grad():
        bits, _ = _bits(x_va, base_mean_va, base_scale_va, y_va, q_va, mask_va)
    return float(bits0.item()), float(bits.item())


def fit_attr_ctx_predictor(
    g: Dict[str, torch.Tensor],
    n_train: int,
    *,
    hidden: int = 64,
    steps: int = 1500,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: torch.device,
    fields: str = "both",
) -> Dict[str, object]:
    """Fit the R4 predictor on the codec-order train split.

    ``g`` must contain (detached, full-anchor) tensors: feat_q, xs, xo, q_s,
    q_o, mean_s, scale_s, mean_o, scale_o, mask, ctx, plus feat_dim/n_offsets
    metadata keys. Returns stats + fitted predictor.
    """
    torch.manual_seed(seed)
    feat_dim = int(g["feat_dim"])
    n_offsets = int(g["n_offsets"])
    k3 = 3 * n_offsets
    pred = AttrCtxPredictor(
        feat_dim,
        g["ctx"].shape[-1],
        n_offsets,
        hidden,
        use_offsets=(fields == "both"),
    ).to(device)

    tr = slice(0, n_train)
    va = slice(n_train, g["ctx"].shape[0])
    mask_tr = g["mask"][tr].reshape(-1, n_offsets).bool()
    mask_va = g["mask"][va].reshape(-1, n_offsets).bool()
    mask_tr_flat = mask_tr.repeat(1, 3).reshape(-1, k3)
    mask_va_flat = mask_va.repeat(1, 3).reshape(-1, k3)

    x_s_tr = _scaling_input(
        g["mean_s"][tr], g["scale_s"][tr], g["feat_q"][tr], g["ctx"][tr]
    )
    x_s_va = _scaling_input(
        g["mean_s"][va], g["scale_s"][va], g["feat_q"][va], g["ctx"][va]
    )
    b0_s, b1_s = _fit_adjust_net(
        pred.net_s,
        x_s_tr, g["mean_s"][tr], g["scale_s"][tr],
        g["xs"][tr], g["q_s"][tr], None,
        x_s_va, g["mean_s"][va], g["scale_s"][va],
        g["xs"][va], g["q_s"][va], None,
        6, steps, weight_decay, device,
    )

    if pred.net_o is not None:
        x_o_tr = _offsets_input(
            g["mean_o"][tr], g["scale_o"][tr], g["feat_q"][tr],
            g["xs"][tr], g["mask"][tr].reshape(-1, n_offsets), g["ctx"][tr],
        )
        x_o_va = _offsets_input(
            g["mean_o"][va], g["scale_o"][va], g["feat_q"][va],
            g["xs"][va], g["mask"][va].reshape(-1, n_offsets), g["ctx"][va],
        )
        b0_o, b1_o = _fit_adjust_net(
            pred.net_o,
            x_o_tr, g["mean_o"][tr], g["scale_o"][tr],
            g["xo"][tr], g["q_o"][tr], mask_tr_flat,
            x_o_va, g["mean_o"][va], g["scale_o"][va],
            g["xo"][va], g["q_o"][va], mask_va_flat,
            k3, steps, weight_decay, device,
        )
        offsets_stats = {
            "base_offsets_bits": b0_o,
            "adj_offsets_bits": b1_o,
            "offsets_gain_pct": (b0_o - b1_o) / b0_o * 100.0 if b0_o > 0 else 0.0,
        }
    else:
        offsets_stats = {
            "base_offsets_bits": None,
            "adj_offsets_bits": None,
            "offsets_gain_pct": 0.0,
        }
    stats = {
        "base_scaling_bits": b0_s,
        "adj_scaling_bits": b1_s,
        "scaling_gain_pct": (b0_s - b1_s) / b0_s * 100.0 if b0_s > 0 else 0.0,
        "n_params": pred.n_params,
        **offsets_stats,
    }
    return {"predictor": pred, "stats": stats}


def attr_ctx_meta(pred: AttrCtxPredictor) -> Dict[str, object]:
    return {
        "feat_dim": pred.feat_dim,
        "grid_ctx_dim": pred.grid_ctx_dim,
        "n_offsets": pred.n_offsets,
        "hidden": pred.hidden,
        "use_offsets": pred.use_offsets,
    }
