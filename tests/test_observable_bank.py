from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

from qgan_latent.workflows.qgan_expectation_values.models.quantum_generator import (
    FixedPauliBank,
    LocalPauliBank,
    MixedPauliBank,
    TwoBodyPauliBank,
    build_generator_apply,
    build_observable_bank,
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


def test_local_pauli_bank_terms_and_metadata() -> None:
    bank = LocalPauliBank(n_qubits=3, paulis=("X", "Y", "Z"))

    assert bank.output_dim == 9
    assert bank.num_terms == 9
    assert bank.max_locality == 1
    assert bank.terms == (
        "X_0",
        "X_1",
        "X_2",
        "Y_0",
        "Y_1",
        "Y_2",
        "Z_0",
        "Z_1",
        "Z_2",
    )
    assert bank.topology == "local"
    assert bank.measurement_groups_estimate == 3
    metadata = bank.metadata()
    assert metadata["terms"] == list(bank.terms)
    assert metadata["topology"] == "local"


def test_two_body_pauli_bank_line_topology_ordering() -> None:
    bank = TwoBodyPauliBank(n_qubits=4, two_body_paulis=("XX", "ZZ"), topology="line")

    assert bank.output_dim == 6
    assert bank.num_terms == 6
    assert bank.max_locality == 2
    assert bank.terms == (
        "XX_0_1",
        "XX_1_2",
        "XX_2_3",
        "ZZ_0_1",
        "ZZ_1_2",
        "ZZ_2_3",
    )
    assert [measurement.obs.wires.tolist() for measurement in bank.measurements()] == [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 1],
        [1, 2],
        [2, 3],
    ]


def test_mixed_pauli_bank_local_then_two_body_order() -> None:
    bank = MixedPauliBank(
        n_qubits=4,
        paulis=("X", "Z"),
        two_body_paulis=("XX", "ZZ"),
        topology="line",
    )

    assert bank.output_dim == 14
    assert bank.max_locality == 2
    assert bank.terms == (
        "X_0",
        "X_1",
        "X_2",
        "X_3",
        "Z_0",
        "Z_1",
        "Z_2",
        "Z_3",
        "XX_0_1",
        "XX_1_2",
        "XX_2_3",
        "ZZ_0_1",
        "ZZ_1_2",
        "ZZ_2_3",
    )


def test_bank_metadata_schema() -> None:
    banks = [
        FixedPauliBank(n_qubits=3),
        LocalPauliBank(n_qubits=3, paulis=("X", "Y", "Z")),
        TwoBodyPauliBank(n_qubits=3, two_body_paulis=("XX", "YZ")),
        MixedPauliBank(n_qubits=3, paulis=("X",), two_body_paulis=("ZZ",)),
    ]

    expected_keys = {
        "name",
        "output_dim",
        "n_qubits",
        "num_terms",
        "max_locality",
        "terms",
        "topology",
        "measurement_groups_estimate",
    }
    for bank in banks:
        assert set(bank.metadata()) == expected_keys
        assert bank.metadata()["output_dim"] == bank.output_dim


def test_invalid_pauli_label_raises() -> None:
    with pytest.raises(ValueError, match="Invalid Pauli label"):
        LocalPauliBank(n_qubits=2, paulis=("A",))


def test_malformed_two_body_term_raises() -> None:
    with pytest.raises(ValueError, match="two-body Pauli"):
        TwoBodyPauliBank(n_qubits=2, two_body_paulis=("X",))


def test_unsupported_topology_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported topology"):
        TwoBodyPauliBank(n_qubits=3, topology="ring")


def test_build_observable_bank_dispatch() -> None:
    assert isinstance(build_observable_bank({"name": "fixed_pauli"}, n_qubits=3), FixedPauliBank)
    assert isinstance(build_observable_bank({"name": "local_pauli"}, n_qubits=3), LocalPauliBank)
    assert isinstance(build_observable_bank({"name": "two_body_pauli"}, n_qubits=3), TwoBodyPauliBank)
    assert isinstance(build_observable_bank({"name": "mixed_pauli"}, n_qubits=3), MixedPauliBank)
    with pytest.raises(ValueError, match="Unsupported observable bank"):
        build_observable_bank({"name": "unknown"}, n_qubits=3)


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


def test_generator_output_dim_matches_bank() -> None:
    rng = jax.random.PRNGKey(0)
    bank = TwoBodyPauliBank(n_qubits=4, two_body_paulis=("XX", "ZZ"), topology="line")
    params = init_generator_params(rng, n_qubits=4, depth=1, noise_dim=2)
    gen_apply = build_generator_apply(n_qubits=4, depth=1, observable_bank=bank)

    fake_features = gen_apply(params, jnp.zeros((3, 2)))

    assert fake_features.shape == (3, bank.output_dim)
    assert bank.terms == (
        "XX_0_1",
        "XX_1_2",
        "XX_2_3",
        "ZZ_0_1",
        "ZZ_1_2",
        "ZZ_2_3",
    )


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
