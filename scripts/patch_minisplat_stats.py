"""Patch HAC++ core so MiniSplat growth/prune keeps training stats aligned."""

from pathlib import Path


path = Path("hacplus/scene/gaussian_model.py")
s = path.read_text()

if "MiniSplat reinit changes the anchor count" in s:
    print("ALREADY_PATCHED")
else:
    old_append = (
        "        self.anchor_demon = torch.cat([self.anchor_demon, ext], dim=0)\n"
        "        self.opacity_accum = torch.cat([self.opacity_accum, ext], dim=0)\n"
    )
    block_append = old_append + (
        "        # MiniSplat reinit changes the anchor count outside the regular\n"
        "        # adjust_anchor path; these optimizer/stat buffers must stay aligned\n"
        "        # with the new N*K size or the next training_statis / adjust_anchor\n"
        "        # overflows (observed as CUDA device-side assert / IndexError on 1-78).\n"
        "        stat_ext = torch.zeros(\n"
        "            (n * self.n_offsets, 1),\n"
        "            dtype=self.offset_gradient_accum.dtype,\n"
        "            device=\"cuda\",\n"
        "        )\n"
        "        self.offset_gradient_accum = torch.cat(\n"
        "            [self.offset_gradient_accum, stat_ext], dim=0\n"
        "        )\n"
        "        self.offset_denom = torch.cat([self.offset_denom, stat_ext], dim=0)\n"
    )
    assert s.count(old_append) == 1, f"append anchor count={s.count(old_append)}"
    s = s.replace(old_append, block_append, 1)

    old_prune = (
        "            tensor = getattr(self, name)\n"
        "            if tensor.numel() > 0:\n"
        "                setattr(self, name, tensor[valid_points_mask])\n"
        "        if self.semantic_target.numel() > 0:\n"
    )
    block_prune = (
        "            tensor = getattr(self, name)\n"
        "            if tensor.numel() > 0:\n"
        "                setattr(self, name, tensor[valid_points_mask])\n"
        "        # Direct prune calls (e.g. MiniSplat-full) bypass adjust_anchor's\n"
        "        # manual slicing; keep growth statistics aligned here too.\n"
        "        if self.offset_gradient_accum.numel() > 0:\n"
        "            keep_stats = valid_points_mask.repeat_interleave(self.n_offsets)\n"
        "            self.offset_gradient_accum = self.offset_gradient_accum[keep_stats]\n"
        "            self.offset_denom = self.offset_denom[keep_stats]\n"
        "        if self.opacity_accum.numel() > 0:\n"
        "            self.opacity_accum = self.opacity_accum[valid_points_mask]\n"
        "        if self.anchor_demon.numel() > 0:\n"
        "            self.anchor_demon = self.anchor_demon[valid_points_mask]\n"
        "        if self.max_radii2D.numel() > 0:\n"
        "            self.max_radii2D = self.max_radii2D[valid_points_mask]\n"
        "        if self.semantic_target.numel() > 0:\n"
    )
    assert s.count(old_prune) == 1, f"prune anchor count={s.count(old_prune)}"
    s = s.replace(old_prune, block_prune, 1)
    path.write_text(s)
    print("REMOTE_PATCHED")
