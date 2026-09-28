#!/usr/bin/env python3
"""Container format for the c25 layered bitstream (numpy only, no torch).

Byte layout (written by c25_real_bitstream.py, read here and by
c25_prefix_decode.py):

  [4B little-endian header len][header JSON utf-8]
  [4B blob len][blob]                      only if header has "s5_levels"
  per chunk, in file (level-major) order:
    [4B params len][params][4B data len][data]

params = locs int32 + p0s float32 + bts float32, all one shape: either
(n_groups,) for base-layer chunks (contribution group only) or
(n_groups, BUCKETS) for enhancement chunks (group x |coarse| bucket).
The shape is not stored per chunk; `split_params` recovers it from
header["groups"] and the byte count.

Keeping writer and reader in one module (instead of an inline writer) is
what lets the standalone prefix decoder and the truncation tests consume
exactly the bytes the encoder produced.
"""
import json
from pathlib import Path

import numpy as np

BUCKETS = 3  # residual context: |coarse symbol| in {0}, {1}, {>=2}


def _len4(n):
    return int(n).to_bytes(4, "little")


def params_bytes(params):
    """(locs, p0s, bts) -> the exact bytes the encoder ships per chunk."""
    locs, p0s, bts = params
    return (np.ascontiguousarray(locs, dtype=np.int32).tobytes()
            + np.ascontiguousarray(p0s, dtype=np.float32).tobytes()
            + np.ascontiguousarray(bts, dtype=np.float32).tobytes())


def split_params(params_raw, n_groups):
    """Inverse of params_bytes: recover (locs, p0s, bts) with shapes."""
    numel = len(params_raw) // 12
    assert len(params_raw) % 12 == 0 and numel % n_groups == 0, \
        f"params size {len(params_raw)} not divisible by groups {n_groups}"
    inner = numel // n_groups
    assert inner in (1, BUCKETS), f"unexpected params inner dim {inner}"
    shape = (n_groups,) if inner == 1 else (n_groups, BUCKETS)
    n = numel * 4
    locs = np.frombuffer(params_raw[:n], dtype=np.int32).reshape(shape)
    p0s = np.frombuffer(params_raw[n:2 * n], dtype=np.float32).reshape(shape)
    bts = np.frombuffer(params_raw[2 * n:], dtype=np.float32).reshape(shape)
    return locs, p0s, bts


def write_container(path, header_bytes, blob, chunk_params, chunk_data):
    """Write the file; returns total size. blob=None for v1/v2 files.

    Byte-identical to the inline writer this module replaces (covered by
    tests/test_c25_container.py against the historical layout).
    """
    blob = b"" if blob is None else blob
    with open(path, "wb") as fh:
        fh.write(_len4(len(header_bytes)))
        fh.write(header_bytes)
        if blob:
            fh.write(_len4(len(blob)))
            fh.write(blob)
        for pb, cdata in zip(chunk_params, chunk_data):
            fh.write(_len4(len(pb)))
            fh.write(pb)
            fh.write(_len4(len(cdata)))
            fh.write(cdata)
    return Path(path).stat().st_size


def read_container(path):
    """Parse a (possibly truncated) container.

    Returns (header_dict, blob_or_None, chunks) where chunks[i] =
    {"params_raw": bytes, "data": bytes}. A file truncated mid-chunk keeps
    only the complete chunks: trailing partial frames are dropped, which is
    exactly the prefix-decodability contract (a decoder reading until EOF
    never needs bytes past the quality prefix it stops at).
    """
    data = Path(path).read_bytes()
    off = 0
    hlen = int.from_bytes(data[off:off + 4], "little")
    off += 4
    header = json.loads(data[off:off + hlen].decode("utf-8"))
    off += hlen
    blob = None
    if "s5_levels" in header:
        blen = int.from_bytes(data[off:off + 4], "little")
        off += 4
        blob = data[off:off + blen]
        off += blen
    chunks = []
    while off + 4 <= len(data):
        plen = int.from_bytes(data[off:off + 4], "little")
        if off + 4 + plen + 4 > len(data):
            break
        off += 4
        params_raw = data[off:off + plen]
        off += plen
        dlen = int.from_bytes(data[off:off + 4], "little")
        if off + 4 + dlen > len(data):
            break
        off += 4
        chunks.append({"params_raw": params_raw,
                       "data": data[off:off + dlen]})
        off += dlen
    return header, blob, chunks


def chunk_end_offsets(path):
    """File offset just past each chunk's data, in file order.

    offsets[k] = prefix length that contains exactly chunks[0..k]; used by
    the truncation test to cut the file at quality-prefix boundaries.
    """
    data = Path(path).read_bytes()
    off = 0
    hlen = int.from_bytes(data[off:off + 4], "little")
    off += 4 + hlen
    header = json.loads(data[4:4 + hlen].decode("utf-8"))
    if "s5_levels" in header:
        blen = int.from_bytes(data[off:off + 4], "little")
        off += 4 + blen
    offsets = []
    while off + 4 <= len(data):
        plen = int.from_bytes(data[off:off + 4], "little")
        if off + 4 + plen + 4 > len(data):
            break
        dlen_at = off + 4 + plen
        dlen = int.from_bytes(data[dlen_at:dlen_at + 4], "little")
        end = dlen_at + 4 + dlen
        if end > len(data):
            break
        offsets.append(end)
        off = end
    return header, offsets


def prefix_byte_length(path, prefix_level):
    """Length of the file prefix holding every chunk with level <= prefix_level.

    The writer emits chunks level-major (stable sort), so those chunks are a
    contiguous head of the file body; header (and s5 blob, which sits before
    all chunk data) are always included.
    """
    header, offsets = chunk_end_offsets(path)
    levels = [c["level"] for c in header["chunks"]]
    assert len(levels) == len(offsets), "header/body chunk count mismatch"
    last = -1
    for i, lv in enumerate(levels):
        if lv <= prefix_level:
            last = i
    if last < 0:
        # no chunk at or below this level: header (and blob) only
        head = 4 + len(json.dumps(header).encode("utf-8"))
        if "s5_levels" in header:
            blen = int.from_bytes(Path(path).read_bytes()[head:head + 4],
                                  "little")
            head += 4 + blen
        return head
    return offsets[last]
