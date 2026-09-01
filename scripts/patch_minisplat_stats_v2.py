"""Second MiniSplat stat-sync fix: avoid double-pruning already-sliced stats."""

from pathlib import Path


path = Path("hacplus/scene/gaussian_model.py")
s = path.read_text()
old = "        if self.offset_gradient_accum.numel() > 0:\n"
new = (
    "        if (\n"
    "            self.offset_gradient_accum.numel() > 0\n"
    "            and self.offset_gradient_accum.shape[0] != n_target * self.n_offsets\n"
    "        ):\n"
)
assert s.count(old) == 1, f"count={s.count(old)}"
s = s.replace(old, new, 1)
path.write_text(s)
print("REMOTE_PATCH_V2")
