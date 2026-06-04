from __future__ import annotations

from pathlib import Path

from qgan_latent.stage5 import (
    LATENT_QGAN_REPRESENTATIONS,
    TENSOR_PRIOR_REPRESENTATION,
    ExperimentStatus,
    build_stage5_plan,
    main,
    render_obsidian_note,
    render_summary_markdown,
    run_stage5_plan,
)


def test_stage5_latent_qgan_matrix_excludes_vqvae_variants() -> None:
    assert LATENT_QGAN_REPRESENTATIONS == ("autoencoder", "vae", "sinkhorn_ae")
    assert "vqvae" not in LATENT_QGAN_REPRESENTATIONS
    assert "spatial_vqvae" not in LATENT_QGAN_REPRESENTATIONS


def test_stage5_tensor_prior_uses_only_spatial_vqvae() -> None:
    assert TENSOR_PRIOR_REPRESENTATION == "spatial_vqvae"

    plan = build_stage5_plan(run_id="stage5-test", epochs=5)
    tensor_runs = [run for run in plan.runs if run.workflow == "tensor_prior_vqvae"]

    assert [run.name for run in tensor_runs] == [
        "spatial_vqvae_pretrain_5ep",
        "mps_prior_train_5ep",
    ]
    assert all(run.representation == "spatial_vqvae" for run in tensor_runs)


def test_stage5_plan_builds_expected_ordered_matrix_and_output_dirs() -> None:
    plan = build_stage5_plan(run_id="stage5-test", epochs=5)

    assert plan.output_root == Path("outputs/experiments/stage5_test_run/stage5-test")
    assert [run.name for run in plan.runs] == [
        "latent_autoencoder_pretrain_5ep",
        "latent_autoencoder_qgan_5ep",
        "latent_vae_pretrain_5ep",
        "latent_vae_qgan_5ep",
        "latent_sinkhorn_ae_pretrain_5ep",
        "latent_sinkhorn_ae_qgan_5ep",
        "spatial_vqvae_pretrain_5ep",
        "mps_prior_train_5ep",
    ]
    assert all("vqvae" not in run.name for run in plan.runs[:6])
    assert plan.runs[0].output_dir == plan.output_root / "latent_qgan" / "autoencoder" / "pretrain"
    assert plan.runs[1].output_dir == plan.output_root / "latent_qgan" / "autoencoder" / "qgan_train"
    assert plan.runs[-1].output_dir == plan.output_root / "tensor_prior_vqvae" / "mps_prior_train"


def test_stage5_commands_use_project_clis_and_checkpoint_wiring() -> None:
    plan = build_stage5_plan(run_id="stage5-test", epochs=5)
    by_name = {run.name: run for run in plan.runs}

    auto_pretrain = by_name["latent_autoencoder_pretrain_5ep"]
    assert auto_pretrain.command[0] == "qgan-latent-pretrain"
    assert "shared/representations@model.autoencoder=autoencoder" in auto_pretrain.command
    assert "pretrain_epochs=5" in auto_pretrain.command
    assert "device=gpu" in auto_pretrain.command
    assert "wandb_mode=disabled" in auto_pretrain.command

    auto_qgan = by_name["latent_autoencoder_qgan_5ep"]
    assert auto_qgan.command[0] == "qgan-latent-train"
    assert "epochs=5" in auto_qgan.command
    assert "device=gpu" in auto_qgan.command
    assert any(
        item == "checkpoints.dir=outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/qgan_train/checkpoints"
        for item in auto_qgan.command
    )
    assert any(
        item == "checkpoints.autoencoder=outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/pretrain/checkpoints/autoencoder.ckpt"
        for item in auto_qgan.command
    )

    mps_prior = by_name["mps_prior_train_5ep"]
    assert mps_prior.command[0] == "qgan-vqvae-train-prior"
    assert "model.mps_prior.epochs=5" in mps_prior.command
    assert "shared/representations@model.vqvae=spatial_vqvae" in mps_prior.command
    assert "device=gpu" in mps_prior.command
    assert any(
        item == "checkpoints.dir=outputs/experiments/stage5_test_run/stage5-test/tensor_prior_vqvae/mps_prior_train/checkpoints"
        for item in mps_prior.command
    )
    assert any(
        item == "checkpoints.vqvae=outputs/experiments/stage5_test_run/stage5-test/tensor_prior_vqvae/spatial_vqvae_pretrain/checkpoints/spatial_vqvae.ckpt"
        for item in mps_prior.command
    )


def test_stage5_summary_and_obsidian_note_are_compact_and_traceable() -> None:
    plan = build_stage5_plan(run_id="stage5-test", epochs=5)
    statuses = [
        ExperimentStatus(run=run, returncode=0, log_path=run.log_path, elapsed_seconds=1.25)
        for run in plan.runs
    ]

    summary = render_summary_markdown(plan, statuses, git_commit="abc123")
    assert "# Stage 5 Test Run Summary" in summary
    assert "Run ID: stage5-test" in summary
    assert "Git commit: abc123" in summary
    assert "latent_autoencoder_pretrain_5ep: PASS" in summary
    assert "mps_prior_train_5ep: PASS" in summary
    assert "latent_vqvae" not in summary

    note = render_obsidian_note(plan, statuses, git_commit="abc123")
    assert "# QGAN Latent Representations - Stage 5 Test Run" in note
    assert "[[QGAN Latent Representations]]" in note
    assert "No VQ-VAE variants are used for latent-QGAN runs" in note
    assert "Output root: outputs/experiments/stage5_test_run/stage5-test" in note


def test_stage5_runner_stops_after_first_failed_experiment(monkeypatch) -> None:
    plan = build_stage5_plan(run_id="stage5-test", epochs=5)
    calls = []

    def fake_run_experiment(run, *, dry_run=False):
        calls.append(run.name)
        return ExperimentStatus(run=run, returncode=23, log_path=run.log_path, elapsed_seconds=0.5)

    monkeypatch.setattr("qgan_latent.stage5.run_experiment", fake_run_experiment)

    statuses = run_stage5_plan(plan)

    assert calls == ["latent_autoencoder_pretrain_5ep"]
    assert len(statuses) == 1
    assert statuses[0].returncode == 23


def test_stage5_main_dry_run_writes_summary_logs_and_obsidian_note(tmp_path) -> None:
    output_root = tmp_path / "outputs"
    obsidian_vault = tmp_path / "obsidian"

    exit_code = main([
        "--run-id",
        "stage5-test",
        "--epochs",
        "5",
        "--output-root",
        str(output_root),
        "--dry-run",
        "--write-obsidian",
        "--obsidian-vault",
        str(obsidian_vault),
    ])

    assert exit_code == 0
    run_root = output_root / "stage5-test"
    summary = (run_root / "stage5_test_run_summary.md").read_text(encoding="utf-8")
    assert "latent_autoencoder_pretrain_5ep: PASS" in summary
    assert "latent_vqvae" not in summary

    first_log = run_root / "latent_qgan" / "autoencoder" / "pretrain" / "stage5.log"
    assert first_log.is_file()
    assert first_log.read_text(encoding="utf-8").startswith("qgan-latent-pretrain")

    obsidian_note = obsidian_vault / "Research" / "PhD" / "Experiments" / "QGAN Latent Representations - Stage 5 Test Run.md"
    assert obsidian_note.is_file()
    note_text = obsidian_note.read_text(encoding="utf-8")
    assert "No VQ-VAE variants are used for latent-QGAN runs" in note_text
    assert "latent_vqvae" not in note_text


def test_stage5_commands_default_to_gpu_where_possible_and_keep_train_checkpoints_separate() -> None:
    plan = build_stage5_plan(run_id="stage5-test", epochs=5)
    by_name = {run.name: run for run in plan.runs}

    auto_pretrain = by_name["latent_autoencoder_pretrain_5ep"]
    assert "device=gpu" in auto_pretrain.command

    auto_qgan = by_name["latent_autoencoder_qgan_5ep"]
    assert "device=gpu" in auto_qgan.command
    assert any(
        item == "checkpoints.dir=outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/qgan_train/checkpoints"
        for item in auto_qgan.command
    )
    assert any(
        item == "checkpoints.autoencoder=outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/pretrain/checkpoints/autoencoder.ckpt"
        for item in auto_qgan.command
    )
    assert auto_qgan.expected_outputs == (
        Path("outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/qgan_train/checkpoints/qgan_gen.ckpt"),
        Path("outputs/experiments/stage5_test_run/stage5-test/latent_qgan/autoencoder/qgan_train/checkpoints/qgan_disc.ckpt"),
    )

    mps_prior = by_name["mps_prior_train_5ep"]
    assert "device=gpu" in mps_prior.command
    assert any(
        item == "checkpoints.dir=outputs/experiments/stage5_test_run/stage5-test/tensor_prior_vqvae/mps_prior_train/checkpoints"
        for item in mps_prior.command
    )
    assert any(
        item == "checkpoints.vqvae=outputs/experiments/stage5_test_run/stage5-test/tensor_prior_vqvae/spatial_vqvae_pretrain/checkpoints/spatial_vqvae.ckpt"
        for item in mps_prior.command
    )
    assert mps_prior.expected_outputs == (
        Path("outputs/experiments/stage5_test_run/stage5-test/tensor_prior_vqvae/mps_prior_train/checkpoints/mps_prior.ckpt"),
    )


def test_stage5_main_can_export_obsidian_handoff_for_vps_filing(tmp_path) -> None:
    output_root = tmp_path / "outputs"

    exit_code = main([
        "--run-id",
        "stage5-test",
        "--epochs",
        "5",
        "--output-root",
        str(output_root),
        "--dry-run",
        "--obsidian-handoff",
    ])

    assert exit_code == 0
    run_root = output_root / "stage5-test"
    handoff = run_root / "obsidian_handoff" / "QGAN Latent Representations - Stage 5 Test Run.md"
    assert handoff.is_file()
    handoff_text = handoff.read_text(encoding="utf-8")
    assert "# QGAN Latent Representations - Stage 5 Test Run" in handoff_text
    assert "stage5-test" in handoff_text
