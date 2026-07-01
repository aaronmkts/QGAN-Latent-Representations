from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.models.surrogate import (
    build_fourier_apply,
    fourier_features,
    fourier_parameter_count,
    init_fourier_params,
    matched_rff_dim,
)
from qgan_latent.workflows.qgan_expectation_values.training.surrogate_loop import run_surrogate_gan


def _compose_train(overrides: list[str]):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="projects/qgan_expectation_values/train", overrides=overrides)


def _smoke_overrides(output_dir: str = "outputs/surrogate_smoke") -> list[str]:
    return [
        "smoke_test=true",
        "device=cpu",
        "wandb_mode=disabled",
        "metrics.active_metrics=[]",
        f"outputs.dir={output_dir}",
        "checkpoints.dir=checkpoints",
        "model.autoencoder.latent_dim=4",
        "model.autoencoder.encoder_channels=[4,8]",
        "model.autoencoder.decoder_channels=[8,4]",
        "model.autoencoder.mlp_dim=16",
        "model.quantum_generator.n_qubits=2",
        "model.quantum_generator.noise_dim=2",
        "model.quantum_generator.depth=1",
        "surrogate.enabled=true",
        "surrogate.hidden_dim=8",
    ]


def test_fourier_surrogate_output_shape_matches_bank() -> None:
    rng = jax.random.PRNGKey(0)
    params = init_fourier_params(rng, noise_dim=2, output_dim=4, rff_dim=8, hidden_dim=8)
    apply = build_fourier_apply(noise_dim=2, output_dim=4, rff_dim=8, hidden_dim=8, sigma=1.0, seed=7)
    output = apply(params, jnp.ones((3, 2)))
    assert output.shape == (3, 4)


def test_fourier_features_are_deterministic_given_seed() -> None:
    noise = jnp.ones((2, 3))
    a = fourier_features(noise, rff_dim=6, sigma=1.0, seed=11)
    b = fourier_features(noise, rff_dim=6, sigma=1.0, seed=11)
    c = fourier_features(noise, rff_dim=6, sigma=1.0, seed=12)
    assert jnp.allclose(a, b)
    assert not jnp.allclose(a, c)


def test_matched_rff_dim_equals_quantum_angle_count() -> None:
    assert matched_rff_dim(4, 3) == 24


def test_fourier_parameter_count_has_expected_shape() -> None:
    counts = fourier_parameter_count(noise_dim=2, output_dim=4, rff_dim=8, hidden_dim=8)
    assert set(counts) == {"rff_weights", "rff_bias", "w1", "b1", "w2", "b2", "total"}
    assert counts["total"] > 0
    assert counts["rff_weights"] == 16


def test_surrogate_smoke_train_runs_without_pretrained_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose_train(_smoke_overrides())

    run_surrogate_gan(cfg, matched_run_id="qgan-parent")

    out = tmp_path / "outputs" / "surrogate_smoke"
    assert (out / "metrics.json").exists()
    assert (out / "evs_diagnostics.json").exists()
    record_path = out / "run_record.json"
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["workflow"] == "fourier_surrogate"
    assert record["parent_run_id"] == "qgan-parent"
    assert record["resource_counts"]["surrogate_rff_dim"] == 4


def test_surrogate_runs_with_mixed_pauli_bank(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _compose_train(
        _smoke_overrides("outputs/surrogate_mixed")
        + [
            "model.quantum_generator.n_qubits=4",
            "model.quantum_generator.noise_dim=4",
            "model.autoencoder.latent_dim=14",
            "model.quantum_generator.observable_bank.name=mixed_pauli",
        ]
    )

    run_surrogate_gan(cfg)

    diagnostics = json.loads((tmp_path / "outputs" / "surrogate_mixed" / "evs_diagnostics.json").read_text(encoding="utf-8"))
    assert len(diagnostics) == 1
    assert len(diagnostics[0]["parseval_norm_sq"]) == 14
