from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import (
    FixedPauliBank,
    build_generator_apply,
    init_generator_params,
)
from qgan_latent.workflows.qgan_expectation_values.training.gan_loop import run_gan


def test_fixed_pauli_bank_output_dim_and_validation() -> None:
    assert FixedPauliBank(n_qubits=10, paulis=("X", "Z")).output_dim == 20
    assert FixedPauliBank(n_qubits=10, paulis=("X",)).output_dim == 10

    with pytest.raises(ValueError, match="Invalid Pauli label"):
        FixedPauliBank(n_qubits=10, paulis=("A",))


def test_default_fixed_pauli_bank_preserves_x_then_z_readout() -> None:
    bank = FixedPauliBank(n_qubits=10)

    assert bank.output_dim == 20
    assert bank.paulis == ("X", "Z")

    measurements = bank.measurements()
    assert [m.obs.name for m in measurements[:10]] == ["PauliX"] * 10
    assert [m.obs.wires.tolist() for m in measurements[:10]] == [[i] for i in range(10)]
    assert [m.obs.name for m in measurements[10:]] == ["PauliZ"] * 10
    assert [m.obs.wires.tolist() for m in measurements[10:]] == [[i] for i in range(10)]


def test_custom_fixed_pauli_bank_changes_generator_output_dim() -> None:
    rng = jax.random.PRNGKey(0)
    params = init_generator_params(rng, n_qubits=10, depth=1, noise_dim=2)
    gen_apply = build_generator_apply(
        n_qubits=10,
        depth=1,
        observable_bank=FixedPauliBank(n_qubits=10, paulis=("X",)),
    )

    fake_features = gen_apply(params, jnp.zeros((3, 2)))

    assert fake_features.shape == (3, 10)


def test_quantum_generator_config_declares_default_observable_bank() -> None:
    config_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="projects/qgan_expectation_values/train")

    bank_cfg = cfg.model.quantum_generator.observable_bank
    assert bank_cfg.name == "fixed_pauli"
    assert list(bank_cfg.paulis) == ["X", "Z"]
    assert FixedPauliBank(
        n_qubits=cfg.model.quantum_generator.n_qubits,
        paulis=tuple(bank_cfg.paulis),
    ).output_dim == cfg.model.autoencoder.latent_dim == 20


def test_gan_validation_uses_observable_bank_output_dim(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="projects/qgan_expectation_values/train",
            overrides=[
                "smoke_test=true",
                "device=cpu",
                "wandb_mode=disabled",
                "metrics.active_metrics=[]",
                "outputs.dir=outputs/train_smoke",
                "checkpoints.dir=checkpoints",
                "model.autoencoder.latent_dim=20",
                "model.autoencoder.encoder_channels=[4,8]",
                "model.autoencoder.decoder_channels=[8,4]",
                "model.autoencoder.mlp_dim=16",
                "model.quantum_generator.n_qubits=10",
                "model.quantum_generator.noise_dim=2",
                "model.quantum_generator.depth=1",
                "model.quantum_generator.observable_bank.paulis=[X]",
            ],
        )

    with pytest.raises(ValueError, match=r"observable_bank.output_dim \(10\)"):
        run_gan(cfg)
