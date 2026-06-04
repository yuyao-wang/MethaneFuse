# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

__all__ = [
    "DatasetWithEnumeratedTargets",
    "make_data_loader",
    "make_dataset",
    "SamplerType",
    "collate_data_and_cast",
    "MaskingGenerator",
    "make_augmentation",
]


def __getattr__(name):
    if name == "DatasetWithEnumeratedTargets":
        from .adapters import DatasetWithEnumeratedTargets

        return DatasetWithEnumeratedTargets
    if name in {"make_data_loader", "make_dataset", "SamplerType"}:
        from .loaders import make_data_loader, make_dataset, SamplerType

        return {"make_data_loader": make_data_loader, "make_dataset": make_dataset, "SamplerType": SamplerType}[name]
    if name == "collate_data_and_cast":
        from .collate import collate_data_and_cast

        return collate_data_and_cast
    if name == "MaskingGenerator":
        from .masking import MaskingGenerator

        return MaskingGenerator
    if name == "make_augmentation":
        from .augmentations import make_augmentation

        return make_augmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
