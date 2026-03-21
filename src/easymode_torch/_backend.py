"""Backend implementation for torch-segment-tomogram.

Registers easymode-torch as a segmentation backend so it can be
discovered and used via the unified torch-segment-tomogram API.
"""

from __future__ import annotations

from typing import Any


class _Feature:
    def __init__(self, name: str, dim: str):
        self.name = name
        self.dim = dim
        self.description = ""


class EasymodeBackend:
    @property
    def name(self) -> str:
        return "easymode"

    def list_features(self) -> list[_Feature]:
        from ._distribution import list_models
        models = list_models(silent=True)
        features = []
        for m in models:
            if m["has_3d"] and m["has_2d"]:
                dim = "3D/2D"
            elif m["has_3d"]:
                dim = "3D"
            else:
                dim = "2D"
            features.append(_Feature(m["title"], dim))
        return features

    def segment(
        self,
        feature: str,
        data: Any,
        output: str = "segmented",
        **kwargs: Any,
    ) -> None:
        use_2d = kwargs.pop("use_2d", False)
        if use_2d:
            from . import segment_2d
            segment_2d(
                feature=feature,
                data_directory=data,
                output_directory=output,
                **kwargs,
            )
        else:
            from . import segment
            segment(
                feature=feature,
                data_directory=data,
                output_directory=output,
                **kwargs,
            )


backend = EasymodeBackend()
