from __future__ import annotations

import argparse
from pathlib import Path

from qgan_latent.workflows.qgan_expectation_values.training.stage8 import (
    aggregate_stage8_results,
    write_stage8_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Stage 8 QGAN/surrogate comparison artifacts.")
    parser.add_argument("output_root", type=Path, help="Stage 8 output directory containing condition subdirectories.")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = aggregate_stage8_results(args.output_root, run_id=args.run_id)
    json_path, csv_path, md_path = write_stage8_summary(summary, args.output_root)
    print(f"STAGE8_SUMMARY_JSON={json_path}")
    print(f"STAGE8_SUMMARY_CSV={csv_path}")
    print(f"STAGE8_SUMMARY_MD={md_path}")
    print(f"STAGE8_RESULT_COUNT={len(summary.get('results', []))}")


if __name__ == "__main__":
    main()
