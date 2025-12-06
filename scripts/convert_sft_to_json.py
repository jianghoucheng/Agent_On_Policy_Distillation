#!/usr/bin/env python3
"""
Utility to preview SFT parquet data as JSONL without relying on datasets metadata.
"""

import argparse
import json
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is not installed in this environment. Please install pyarrow (e.g., pip install pyarrow) "
        "before running this script."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SFT parquet data to JSONL preview.")
    parser.add_argument("--input", required=True, help="Path to the parquet file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination JSONL file. Existing file will be overwritten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of samples to export (default: 20, set -1 for all rows).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    parquet_path = Path(args.input)
    pf = pq.ParquetFile(parquet_path)

    limit = args.limit
    exported = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches():
            for row in batch.to_pylist():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                exported += 1
                if limit > 0 and exported >= limit:
                    break
            if limit > 0 and exported >= limit:
                break
    print(f"Exported {exported} samples to {args.output}")


if __name__ == "__main__":
    main()
