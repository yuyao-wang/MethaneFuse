#!/usr/bin/env python3
"""Losslessly recover a TIFF whose IFD was never finalized.

The legacy crop writer first reserved a 272-byte TIFF header/IFD region and
then wrote contiguous float32 image planes.  A small number of interrupted
writes contain the complete pixel payload but an all-zero reserved IFD.  This
utility accepts only that exact forensic signature, reconstructs a valid TIFF
in the hashed local cache, and verifies pixel-for-pixel equality before an
atomic publish.  It never modifies the source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile


HEADER_BYTES = 272
EXPECTED_SHAPE = (12, 224, 224)
EXPECTED_DTYPE = np.dtype("<f4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def cache_path(source: Path, cache_root: Path) -> Path:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    suffix = "".join(source.suffixes[-2:]) if source.suffixes else ""
    if len(suffix) > 24:
        suffix = source.suffix
    return cache_root / digest[:2] / f"{digest}{suffix}"


def recover(source: Path, cache_root: Path) -> dict[str, object]:
    source = source.expanduser().absolute()
    cache_root = cache_root.expanduser().absolute()
    if not source.is_file():
        raise FileNotFoundError(source)

    expected_payload_bytes = int(np.prod(EXPECTED_SHAPE)) * EXPECTED_DTYPE.itemsize
    expected_source_bytes = HEADER_BYTES + expected_payload_bytes
    if source.stat().st_size != expected_source_bytes:
        raise ValueError(
            f"source size is not the exact recoverable contract: "
            f"{source.stat().st_size} != {expected_source_bytes}"
        )
    with source.open("rb") as stream:
        header = stream.read(HEADER_BYTES)
    if header[:8] != b"II*\x00\x08\x00\x00\x00":
        raise ValueError("source is not the expected little-endian TIFF header")
    if any(header[8:]):
        raise ValueError("reserved IFD region is not entirely zero")

    pixels = np.fromfile(
        source, dtype=EXPECTED_DTYPE, count=int(np.prod(EXPECTED_SHAPE)),
        offset=HEADER_BYTES,
    ).reshape(EXPECTED_SHAPE)
    if not np.isfinite(pixels).all():
        raise ValueError("pixel payload contains non-finite values")
    if float((pixels[11] == 0).mean()) >= 0.20:
        raise ValueError("recovered S2 band 11 fails the legacy quality gate")

    destination = cache_path(source, cache_root)
    audit_path = Path(str(destination) + ".zero_ifd_recovery.json")
    if destination.exists() or audit_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing recovery: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".partial",
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        tifffile.imwrite(temporary_path, pixels)
        reread = np.asarray(tifffile.imread(temporary_path))
        if reread.shape != EXPECTED_SHAPE or reread.dtype != np.dtype("float32"):
            raise RuntimeError(
                f"reconstructed TIFF contract differs: {reread.shape}, {reread.dtype}"
            )
        if not np.array_equal(reread, pixels, equal_nan=True):
            raise RuntimeError("reconstructed TIFF is not pixel-exact")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

    payload_sha = hashlib.sha256(pixels.tobytes(order="C")).hexdigest()
    audit: dict[str, object] = {
        "schema_version": "legacy360-zero-ifd-tiff-cache-recovery-v1",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_bytes": source.stat().st_size,
        "forensic_contract": {
            "byte_order": "little-endian",
            "ifd_offset": 8,
            "zero_reserved_ifd_bytes": [8, HEADER_BYTES],
            "payload_offset": HEADER_BYTES,
            "shape": list(EXPECTED_SHAPE),
            "dtype": str(EXPECTED_DTYPE),
            "payload_sha256": payload_sha,
        },
        "destination": str(destination),
        "destination_sha256": sha256_file(destination),
        "destination_bytes": destination.stat().st_size,
        "pixel_exact_roundtrip": True,
        "source_modified": False,
        "labels_consulted": False,
    }
    atomic_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.source, args.cache_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
