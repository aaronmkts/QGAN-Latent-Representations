from __future__ import annotations

import argparse
from pathlib import Path

from qgan_latent.workflows.qgan_expectation_values.training.stage8 import (
    Stage8SweepConfig,
    preflight_conditions,
    run_stage8_sweep,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or preflight the Stage 8 observable-bank QGAN/surrogate sweep.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--frozen-encoder", type=Path, default=None)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and write preflight only; do not train.")
    parser.add_argument("--conditions", default=None, help="Comma-separated condition names to run.")
    parser.add_argument("--rerun-completed", action="store_true", help="Ignore completed run_record.json files and rerun.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = Stage8SweepConfig.default(
        args.repo_root,
        run_id=args.run_id,
        frozen_encoder=args.frozen_encoder,
        output_root=args.output_root,
        device=args.device,
        smoke_test=args.smoke_test,
    )
    cfg.dry_run = bool(args.dry_run)
    cfg.skip_completed = not bool(args.rerun_completed)
    if args.conditions:
        cfg.selected_conditions = {item.strip() for item in args.conditions.split(",") if item.strip()}
    if args.dry_run:
        rows = preflight_conditions(cfg)
        for row in rows:
            print(f"{row.condition}: latent_dim={row.latent_dim} output_dim={row.output_dim} valid={row.valid} {row.message}")
    summary = run_stage8_sweep(cfg)
    print(f"STAGE8_OUTPUT_ROOT={cfg.output_root}")
    if not args.dry_run:
        print(f"STAGE8_SUMMARY_JSON={cfg.output_root / 'stage8_summary.json'}")
    else:
        print(f"STAGE8_PREFLIGHT_JSON={cfg.output_root / 'preflight.json'}")
    print(f"STAGE8_RESULT_COUNT={len(summary.get('results', []))}")


if __name__ == "__main__":
    main()
