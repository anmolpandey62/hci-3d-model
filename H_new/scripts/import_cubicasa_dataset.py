#!/usr/bin/env python3
"""
CLI: import a frozen Cubicasa YOLO-seg dataset into an H-managed artifact.

Example:
  python scripts/import_cubicasa_dataset.py ^
    --source C:\\cubicasa_converter\\output\\runs\\20260807_024316

Does not modify the source. Does not touch gdrive_dataset or production weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure H root is on sys.path when run as a script
_H_ROOT = Path(__file__).resolve().parent.parent
if str(_H_ROOT) not in sys.path:
    sys.path.insert(0, str(_H_ROOT))

from logic.cubicasa_dataset_import import (  # noqa: E402
    CubicasaImportError,
    import_cubicasa_dataset,
    validate_source_dataset,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Import Cubicasa freeze dataset into H")
    p.add_argument(
        "--source",
        required=True,
        help="Path to Cubicasa freeze run (contains dataset.yaml, images/, labels/)",
    )
    p.add_argument(
        "--version",
        default=None,
        help="H import version id (default: source folder name)",
    )
    p.add_argument(
        "--dest-root",
        default=None,
        help="Optional explicit destination directory for this version",
    )
    p.add_argument(
        "--imports-base",
        default=None,
        help="Override HCI_CUBICASA_IMPORTS_ROOT parent for versions",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; do not copy files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing import version (explicit; default refuses overwrite)",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Print validation report JSON and exit (0 if ok)",
    )
    args = p.parse_args(argv)

    source = Path(args.source)
    if args.validate_only:
        report = validate_source_dataset(source)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok else 2

    try:
        result = import_cubicasa_dataset(
            source,
            version=args.version,
            dest_root=args.dest_root,
            imports_base=args.imports_base,
            dry_run=args.dry_run,
            force=args.force,
        )
    except CubicasaImportError as e:
        print(f"IMPORT FAILED: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    if args.dry_run:
        print("Dry-run OK — no files copied.", file=sys.stderr)
    else:
        print(f"Imported → {result.dest_root}", file=sys.stderr)
        print(f"Train with dataset.yaml: {result.dataset_yaml}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
