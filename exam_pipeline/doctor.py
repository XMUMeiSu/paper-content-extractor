"""Lightweight environment diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="检查试卷提取环境")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checks = {
        "dataset": args.dataset.is_dir(),
        "opencv": importlib.util.find_spec("cv2") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
    }
    print(json.dumps(checks, ensure_ascii=False))
    return 0 if checks["dataset"] and checks["opencv"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
