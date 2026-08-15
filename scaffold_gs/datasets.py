"""COLMAP dataset loading and gsplat camera conversion.

The loader mirrors the conventions used by the gsplat examples
(``examples/datasets/colmap.py``): images are sorted by name, every
``test_every``-th image is held out for evaluation, intrinsics are divided by
``data_factor``, and world coordinates are kept in the original COLMAP frame
(no normalization), matching the official Scaffold-GS behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .utils import camera_extent_radius

def _image_w2c(image) -> np.ndarray:
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    w2c = np.eye(4)
    w2c[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)
    return w2c


def _as_dict(map_like) -> Dict[int, object]:
    return {int(k): v for k, v in map_like.items()}


def max_width_size(width: int, height: int, max_width: int):
    """Official HAC++ ``resolution=-1`` resize rule (int truncation)."""
    if width > max_width:
        scale = width / float(max_width)
        return max_width, int(height / scale)
    return width, height


@dataclass
class SceneCamera:
    """A single camera with everything needed for gsplat rasterization."""

    uid: int
    colmap_id: int
    image_name: str
    image_path: Path
    c2w: np.ndarray  # [4, 4] camera-to-world (COLMAP convention)
    K: np.ndarray  # [3, 3] intrinsics, already scaled by data_factor
    width: int
    height: int
    split: str  # "train" or "val"
    appearance_id: int  # index into the appearance embedding (train cameras)
    near_plane: float = 0.01
    far_plane: float = 1e10

    def to_gsplat(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(viewmats, Ks)`` shaped ``[1, 4, 4]`` / ``[1, 3, 3]``."""
        c2w = torch.from_numpy(self.c2w).float().to(device)
        viewmat = torch.linalg.inv_ex(c2w).inverse
        K = torch.from_numpy(self.K).float().to(device)
        return viewmat[None], K[None]

    def camera_center(self, device: torch.device) -> torch.Tensor:
        c2w = torch.from_numpy(self.c2w).float().to(device)
        return c2w[:3, 3]

    def load_image(self, device: torch.device) -> torch.Tensor:
        """Load and resize the image to ``(width, height)``, returning CHW."""
        with Image.open(self.image_path) as img:
            img = img.convert("RGB")
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor.to(device)

    def load_image_cpu_uint8(self) -> torch.Tensor:
        """Resized image as uint8 CHW on CPU (cheap to cache in RAM)."""
        with Image.open(self.image_path) as img:
            img = img.convert("RGB")
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), Image.BICUBIC)
            arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ColmapDataset:
    """Loads a COLMAP scene and exposes train/val cameras and SfM points."""

    def __init__(
        self,
        data_dir: str,
        data_factor: int = 4,
        test_every: int = 8,
        white_background: bool = False,
        preload_images: bool = True,
        max_width: Optional[int] = None,
        cache_images_cpu: bool = True,
        device: str = "cuda",
    ) -> None:
        # Imported lazily: pycolmap must load *after* torch_scatter on the
        # HAC++ env, otherwise the process segfaults at dlopen time.
        try:
            import pycolmap
        except ImportError:  # pragma: no cover
            pycolmap = None
        if pycolmap is None:
            raise ImportError(
                "pycolmap is required to load COLMAP scenes. "
                "Install it with: pip install pycolmap"
            )
        self.data_dir = Path(data_dir)
        self.data_factor = max(1, int(data_factor))
        self.test_every = max(1, int(test_every))
        self.white_background = white_background
        self.max_width = max_width
        self.cache_images_cpu = cache_images_cpu and not preload_images
        self.device = torch.device(device)

        colmap_dir = self.data_dir / "sparse" / "0"
        if not colmap_dir.exists():
            colmap_dir = self.data_dir / "sparse"
        if not colmap_dir.exists():
            raise FileNotFoundError(
                f"No COLMAP sparse dir found under {self.data_dir}"
            )

        reconstruction = pycolmap.Reconstruction(str(colmap_dir))
        cameras = _as_dict(reconstruction.cameras)
        images = _as_dict(reconstruction.images)
        image_ids = [int(i) for i in reconstruction.reg_image_ids()]
        if len(image_ids) == 0:
            raise ValueError(f"No registered images in {colmap_dir}")

        # Build per-image extrinsics / intrinsics / sizes, then sort by name.
        records = []
        for image_id in image_ids:
            im = images[image_id]
            cam = cameras[int(im.camera_id)]
            K = np.asarray(cam.calibration_matrix(), dtype=np.float64).copy()
            K[:2, :] /= self.data_factor
            width = int(cam.width) // self.data_factor
            height = int(cam.height) // self.data_factor
            if self.max_width is not None and width > self.max_width:
                width, height = max_width_size(width, height, self.max_width)
                K[:2, :] *= float(self.max_width) / (int(cam.width) // self.data_factor)
            w2c = _image_w2c(im)
            c2w = np.linalg.inv(w2c)
            records.append(
                {
                    "colmap_id": image_id,
                    "name": im.name,
                    "c2w": c2w,
                    "K": K,
                    "width": width,
                    "height": height,
                }
            )
        records.sort(key=lambda r: r["name"])

        # Resolve the image folder. Prefer images_<factor> when present.
        image_root = self.data_dir / "images"
        if self.data_factor > 1 and (self.data_dir / f"images_{self.data_factor}").exists():
            image_root = self.data_dir / f"images_{self.data_factor}"

        self.cameras: List[SceneCamera] = []
        for idx, rec in enumerate(records):
            split = "val" if idx % self.test_every == 0 else "train"
            self.cameras.append(
                SceneCamera(
                    uid=idx,
                    colmap_id=rec["colmap_id"],
                    image_name=rec["name"],
                    image_path=image_root / rec["name"],
                    c2w=rec["c2w"],
                    K=rec["K"],
                    width=rec["width"],
                    height=rec["height"],
                    split=split,
                    appearance_id=0,
                )
            )

        # Re-number appearance ids over train cameras only (0-based).
        train_appearance = {}
        next_id = 0
        for cam in self.cameras:
            if cam.split == "train":
                train_appearance[cam.uid] = next_id
                next_id += 1
        for cam in self.cameras:
            if cam.split == "train":
                cam.appearance_id = train_appearance[cam.uid]

        self.train_cameras = [c for c in self.cameras if c.split == "train"]
        self.val_cameras = [c for c in self.cameras if c.split == "val"]
        if len(self.train_cameras) == 0:
            raise ValueError("No training cameras after train/val split.")
        print(
            f"[Dataset] {len(self.cameras)} images: "
            f"{len(self.train_cameras)} train, {len(self.val_cameras)} val."
        )

        # SfM points and colors.
        points3d = _as_dict(reconstruction.points3D)
        point_ids = sorted(points3d)
        self.points = np.array(
            [points3d[pid].xyz for pid in point_ids], dtype=np.float32
        ).reshape(-1, 3)
        self.points_rgb = np.array(
            [points3d[pid].color for pid in point_ids], dtype=np.uint8
        ).reshape(-1, 3)
        print(f"[Dataset] {len(self.points)} SfM points.")

        self.scene_scale = camera_extent_radius(
            np.stack([c.c2w for c in self.train_cameras])
        )
        self.num_cameras = len(self.train_cameras)
        self.background = (
            torch.ones(3, device=self.device)
            if self.white_background
            else torch.zeros(3, device=self.device)
        )

        self._images: Dict[int, torch.Tensor] = {}
        self._images_cpu: Dict[int, torch.Tensor] = {}
        if preload_images:
            print("[Dataset] Preloading images ...")
            for cam in self.cameras:
                self._images[cam.uid] = cam.load_image(self.device)

    def get_image(self, cam: SceneCamera) -> torch.Tensor:
        if cam.uid in self._images:
            return self._images[cam.uid]
        if self.cache_images_cpu:
            if cam.uid not in self._images_cpu:
                self._images_cpu[cam.uid] = cam.load_image_cpu_uint8()
            return (self._images_cpu[cam.uid].to(self.device) / 255.0).float()
        return cam.load_image(self.device)
