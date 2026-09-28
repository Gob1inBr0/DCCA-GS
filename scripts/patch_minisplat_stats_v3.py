"""Third MiniSplat stat-sync fix: only prune stats that are not yet aligned."""

from pathlib import Path


path = Path("hacplus/scene/gaussian_model.py")
s = path.read_text()
old = (
    "        # Direct prune calls (e.g. MiniSplat-full) bypass adjust_anchor's\n"
    "        # manual slicing; keep growth statistics aligned here too.\n"
    "        if (\n"
    "            self.offset_gradient_accum.numel() > 0\n"
    "            and self.offset_gradient_accum.shape[0] != n_target * self.n_offsets\n"
    "        ):\n"
    "            keep_stats = valid_points_mask.repeat_interleave(self.n_offsets)\n"
    "            self.offset_gradient_accum = self.offset_gradient_accum[keep_stats]\n"
    "            self.offset_denom = self.offset_denom[keep_stats]\n"
    "        if self.opacity_accum.numel() > 0:\n"
    "            self.opacity_accum = self.opacity_accum[valid_points_mask]\n"
    "        if self.anchor_demon.numel() > 0:\n"
    "            self.anchor_demon = self.anchor_demon[valid_points_mask]\n"
    "        if self.max_radii2D.numel() > 0:\n"
    "            self.max_radii2D = self.max_radii2D[valid_points_mask]\n"
)
new = (
    "        # Direct prune calls (e.g. MiniSplat-full) bypass adjust_anchor's\n"
    "        # manual slicing; keep growth statistics aligned here too.\n"
    "        if (\n"
    "            self.offset_gradient_accum.numel() > 0\n"
    "            and self.offset_gradient_accum.shape[0]\n"
    "            != self._anchor.shape[0] * self.n_offsets\n"
    "        ):\n"
    "            keep_stats = valid_points_mask.repeat_interleave(self.n_offsets)\n"
    "            self.offset_gradient_accum = self.offset_gradient_accum[keep_stats]\n"
    "            self.offset_denom = self.offset_denom[keep_stats]\n"
    "        if (\n"
    "            self.opacity_accum.numel() > 0\n"
    "            and self.opacity_accum.shape[0] != self._anchor.shape[0]\n"
    "        ):\n"
    "            self.opacity_accum = self.opacity_accum[valid_points_mask]\n"
    "        if (\n"
    "            self.anchor_demon.numel() > 0\n"
    "            and self.anchor_demon.shape[0] != self._anchor.shape[0]\n"
    "        ):\n"
    "            self.anchor_demon = self.anchor_demon[valid_points_mask]\n"
    "        if (\n"
    "            self.max_radii2D.numel() > 0\n"
    "            and self.max_radii2D.shape[0] != self._anchor.shape[0]\n"
    "        ):\n"
    "            self.max_radii2D = self.max_radii2D[valid_points_mask]\n"
)
assert s.count(old) == 1, f"count={s.count(old)}"
path.write_text(s.replace(old, new, 1))
print("REMOTE_PATCH_V3")
