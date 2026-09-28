"""Fourth MiniSplat fix: do not prune max_radii2D inside prune_anchor."""

from pathlib import Path


path = Path("hacplus/scene/gaussian_model.py")
s = path.read_text()
old = (
    "        if (\n"
    "            self.max_radii2D.numel() > 0\n"
    "            and self.max_radii2D.shape[0] != self._anchor.shape[0]\n"
    "        ):\n"
    "            self.max_radii2D = self.max_radii2D[valid_points_mask]\n"
)
assert s.count(old) == 1, f"count={s.count(old)}"
path.write_text(s.replace(old, "", 1))
print("REMOTE_PATCH_V4")
