"""Container (c25_container.py) tests — byte parity with the historical
inline writer in c25_real_bitstream.py, plus read-back, shape recovery and
prefix truncation.

Run: python -m pytest tests/test_c25_container.py -q  (CPU only, <1s)

The byte-parity test is the load-bearing one: the standalone prefix decoder
(c25_prefix_decode.py) and the truncation contract assume the encoder's
file layout is exactly the one documented in c25_container's docstring.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from c25_container import (BUCKETS, chunk_end_offsets, params_bytes,
                           prefix_byte_length, read_container, split_params,
                           write_container)

N_GROUPS = 5


def _params(inner):
    shape = (N_GROUPS,) if inner == 1 else (N_GROUPS, BUCKETS)
    rng = np.random.default_rng(0)
    locs = rng.integers(-100, 100, shape).astype(np.int32)
    p0s = rng.random(shape).astype(np.float32)
    bts = rng.random(shape).astype(np.float32) + 0.05
    return locs, p0s, bts


def _fixture(tmp_path, with_blob=True):
    rng = np.random.default_rng(1)
    header = {
        "format": ("c25_layered_v3_s5" if with_blob
                   else "c25_layered_v2_fieldaware"),
        "groups": N_GROUPS,
        "step_tiers": 3,
        "steps": {"feat": [8, 4, 2, 1], "scaling": [2, 1],
                  "offset": [8, 4, 2, 1]},
        "fields": ["feat", "scaling", "offset"],
        "chunks": ([{"level": 0, "field": f}
                    for f in ("feat", "scaling", "offset")]
                   + [{"level": 1, "field": f}
                      for f in ("feat", "scaling", "offset")]),
        "n_alive": 123,
    }
    if with_blob:
        header["s5_levels"] = [1]
    header_bytes = json.dumps(header).encode("utf-8")
    blob = (bytes(rng.integers(0, 255, 97, dtype=np.uint8))
            if with_blob else None)
    chunks = []
    for c in header["chunks"]:
        inner = 1 if c["level"] == 0 else BUCKETS
        chunks.append((inner, _params(inner),
                       bytes(rng.integers(0, 255, 331, dtype=np.uint8))))
    path = tmp_path / "test.bin"
    size = write_container(path, header_bytes, blob,
                           [params_bytes(p) for _, p, _ in chunks],
                           [d for _, _, d in chunks])
    return path, header, blob, chunks, size


def test_params_bytes_matches_inline_writer():
    """params_bytes must equal the historical inline expression."""
    for inner in (1, BUCKETS):
        locs, p0s, bts = _params(inner)
        expected = (locs.astype(np.int32).tobytes()
                    + p0s.astype(np.float32).tobytes()
                    + bts.astype(np.float32).tobytes())
        assert params_bytes((locs, p0s, bts)) == expected


def test_byte_identical_with_historical_inline_writer(tmp_path):
    """Full-file byte parity with the pre-refactor inline writer block."""
    path, header, blob, chunks, size = _fixture(tmp_path)
    header_bytes = json.dumps(header).encode("utf-8")
    ref = tmp_path / "ref.bin"
    with open(ref, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(4, "little"))
        fh.write(header_bytes)
        if blob:
            fh.write(len(blob).to_bytes(4, "little"))
            fh.write(blob)
        for _, params, data in chunks:
            locs, p0s, bts = params
            pb = (locs.astype(np.int32).tobytes()
                  + p0s.astype(np.float32).tobytes()
                  + bts.astype(np.float32).tobytes())
            fh.write(len(pb).to_bytes(4, "little"))
            fh.write(pb)
            fh.write(len(data).to_bytes(4, "little"))
            fh.write(data)
    assert size == ref.stat().st_size
    assert path.read_bytes() == ref.read_bytes()


def test_read_back_roundtrip(tmp_path):
    path, header, blob, chunks, _ = _fixture(tmp_path)
    hdr2, blob2, chs2 = read_container(path)
    assert hdr2 == header
    assert blob2 == blob
    assert len(chs2) == len(chunks)
    for (inner, params, data), c2 in zip(chunks, chs2):
        locs, p0s, bts = split_params(c2["params_raw"], N_GROUPS)
        e_locs, e_p0s, e_bts = params
        assert np.array_equal(locs, e_locs)
        assert np.array_equal(p0s, e_p0s)
        assert np.array_equal(bts, e_bts)
        assert c2["data"] == data


def test_split_params_shapes():
    for inner in (1, BUCKETS):
        locs, p0s, bts = _params(inner)
        shape = (N_GROUPS,) if inner == 1 else (N_GROUPS, BUCKETS)
        rl, rp, rb = split_params(params_bytes((locs, p0s, bts)), N_GROUPS)
        assert rl.shape == shape and rp.shape == shape and rb.shape == shape


def test_prefix_byte_length_and_truncated_read(tmp_path):
    path, header, blob, chunks, _ = _fixture(tmp_path)
    raw = path.read_bytes()
    _, offsets = chunk_end_offsets(path)
    assert len(offsets) == len(chunks)

    # every level prefix must be a contiguous head of the file; the parsed
    # header still declares ALL chunks of the original file (the header is
    # part of the prefix), but only len(expect) complete bodies remain
    for p in (0, 1):
        cut = prefix_byte_length(path, p)
        expect = [c for c in header["chunks"] if c["level"] <= p]
        (tmp_path / f"p{p}.bin").write_bytes(raw[:cut])
        h2, b2, c2 = read_container(tmp_path / f"p{p}.bin")
        assert len(c2) == len(expect)
        assert ([c["level"] for c in h2["chunks"]][:len(expect)]
                == [c["level"] for c in expect])
        assert b2 == blob  # blob sits before all chunk data

    # a cut INSIDE the last chunk's data drops that chunk, never corrupts
    # earlier ones (reader skips trailing partial frames)
    inside = offsets[-1] - 10
    (tmp_path / "mid.bin").write_bytes(raw[:inside])
    _, _, c_mid = read_container(tmp_path / "mid.bin")
    assert len(c_mid) == len(chunks) - 1
