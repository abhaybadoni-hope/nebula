"""Build a fixed-device SKY130 library for the Stage 1 NFET-only CTLE."""

from __future__ import annotations

import argparse
from pathlib import Path

from simulator.provenance import sha256_file


CORNERS = ("tt", "ss", "ff", "sf", "fs")
DEVICE = "sky130_fd_pr__nfet_01v8"


def build_compact_sky130(
    full_library: str | Path, output: str | Path | None = None,
) -> Path:
    """Write a library containing the exact official NFET model at each PVT corner."""
    source = Path(full_library).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"full SKY130 library does not exist: {source}")
    destination = (
        Path(output).expanduser().resolve()
        if output is not None else source.with_name("sky130.nebula_nfet.lib.spice")
    )
    device_root = (source.parent / "../../libs.ref/sky130_fd_pr/spice").resolve()
    lines = [
        "* Generated fixed-device library for Nebula Stage 1",
        f"* Full library SHA256: {sha256_file(source)}",
        "* Contains only the official sky130_fd_pr__nfet_01v8 models used by the CTLE.",
        "",
    ]
    for corner in CORNERS:
        model = device_root / f"{DEVICE}__{corner}.pm3.spice"
        mismatch = device_root / f"{DEVICE}__mismatch.corner.spice"
        if not model.is_file():
            raise FileNotFoundError(f"SKY130 {corner} NFET model does not exist: {model}")
        if not mismatch.is_file():
            raise FileNotFoundError(f"SKY130 NFET mismatch parameters do not exist: {mismatch}")
        relative = Path("../../libs.ref/sky130_fd_pr/spice") / model.name
        mismatch_relative = Path("../../libs.ref/sky130_fd_pr/spice") / mismatch.name
        lines.extend((
            f".lib {corner}",
            ".option scale=1.0u",
            ".param mc_mm_switch=0",
            ".param mc_pr_switch=0",
            f'* Official model SHA256: {sha256_file(model)}',
            f'.include "{relative.as_posix()}"',
            f'* Official mismatch-parameter SHA256: {sha256_file(mismatch)}',
            f'.include "{mismatch_relative.as_posix()}"',
            f".endl {corner}",
            "",
        ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_library", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_compact_sky130(args.full_library, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
