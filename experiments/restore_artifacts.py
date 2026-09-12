"""Restore packaged historical results without overwriting existing files."""
from pathlib import Path
import zipfile

def restore(root=Path(__file__).resolve().parents[1]):
    root = Path(root).resolve()
    destination = root / "results"
    count = 0
    with zipfile.ZipFile(root / "artifacts/NEBULA_phase2_experimental_artifacts.zip") as archive:
        for name in archive.namelist():
            prefix = "NEBULA_phase2_experimental_artifacts/results/"
            if not name.startswith(prefix) or name.endswith("/"): continue
            path = (destination / name[len(prefix):]).resolve()
            if not path.is_relative_to(destination): raise ValueError("unsafe archive path")
            if path.exists(): continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
            count += 1
    return count

if __name__ == "__main__":
    print(f"Restored {restore()} missing artifact files")
