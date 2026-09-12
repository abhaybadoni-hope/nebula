"""Read-only reproducibility and simulation readiness check."""
import importlib.metadata
import json
import platform
import sys
from simulator.config import Sky130Config
from simulator.ngspice import NgSpiceConfig, ngspice_identity
from simulator.provenance import spice_dependency_fingerprint

def preflight():
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}, "errors": []}
    for package in ("numpy", "torch"):
        try: result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: result["errors"].append(f"missing dependency: {package}")
    try: result["ngspice"] = ngspice_identity(NgSpiceConfig())
    except (FileNotFoundError, OSError, ValueError) as exc: result["errors"].append(str(exc))
    try:
        model = Sky130Config().resolve_model_library()
        result["model"] = str(model)
        result["model_checksum"] = spice_dependency_fingerprint(model)
    except (FileNotFoundError, OSError, ValueError) as exc: result["errors"].append(str(exc))
    result["ready"] = not result["errors"]
    return result

if __name__ == "__main__":
    result = preflight()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)
