"""Compression codec interface (HAC / HAC++ extension point).

V1 only defines the protocol and a ``none`` codec that writes the raw
uncompressed attribute export. HAC (hash-grid assisted context) and HAC++
will later register implementations under ``hac`` / ``hac_pp`` and produce
entropy-coded bitstreams without changing the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, Type

import torch

from .model import BaseGaussianModel


class CompressionCodec(ABC):
    """Interface for compressing a trained anchor-based Gaussian model."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def encode(
        self, model: BaseGaussianModel, output_dir: Path
    ) -> Dict[str, Any]:
        """Compress ``model`` and write artifacts into ``output_dir``.

        Returns metadata (e.g. bitstream sizes) for logging.
        """

    @abstractmethod
    def decode(self, artifact_dir: Path) -> BaseGaussianModel:
        """Rebuild a renderable model from the encoded artifacts."""

    def rate(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Optional: return bit-rate / size statistics."""
        return {}


class RawAttributeCodec(CompressionCodec):
    """Uncompressed baseline: writes the official attribute export."""

    name: ClassVar[str] = "none"

    def encode(
        self, model: BaseGaussianModel, output_dir: Path
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        attrs = model.export_attributes()
        torch.save(attrs, output_dir / "attributes.pth")
        total_bytes = _tensor_bytes(attrs)
        return {
            "codec": self.name,
            "num_anchors": int(attrs["anchor"].shape[0]),
            "raw_attribute_bytes": total_bytes,
        }

    def decode(self, artifact_dir: Path) -> BaseGaussianModel:
        from .model import MODELS

        attrs = torch.load(artifact_dir / "attributes.pth")
        model_cls = MODELS[attrs["config"]["model_name"]]
        return model_cls.from_attributes(attrs, "cuda")


CODECS: Dict[str, Type[CompressionCodec]] = {
    RawAttributeCodec.name: RawAttributeCodec,
}


def register_codec(codec: Type[CompressionCodec]) -> Type[CompressionCodec]:
    """Register a codec (used by HAC / HAC++ extensions)."""
    CODECS[codec.name] = codec
    return codec


def _tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.nelement() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0
