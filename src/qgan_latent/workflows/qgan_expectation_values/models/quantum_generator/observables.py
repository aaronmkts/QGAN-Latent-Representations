from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pennylane as qml
from omegaconf import DictConfig, OmegaConf

_VALID_PAULIS = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
}


@dataclass(frozen=True)
class FixedPauliBank:
    """Fixed per-qubit Pauli expectation-value readout bank.

    Measurements are ordered by Pauli label first, then qubit index. The default
    bank reproduces the existing LaSt-style readout:
    [X_0, ..., X_{n-1}, Z_0, ..., Z_{n-1}].
    """

    n_qubits: int
    paulis: tuple[str, ...] = ("X", "Z")

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}.")
        normalised = tuple(str(pauli).upper() for pauli in self.paulis)
        if not normalised:
            raise ValueError("FixedPauliBank requires at least one Pauli label.")
        invalid = [pauli for pauli in normalised if pauli not in _VALID_PAULIS]
        if invalid:
            valid = ", ".join(sorted(_VALID_PAULIS))
            raise ValueError(
                f"Invalid Pauli label(s) for FixedPauliBank: {invalid}. "
                f"Expected labels from {{{valid}}}."
            )
        object.__setattr__(self, "paulis", normalised)

    @property
    def output_dim(self) -> int:
        return len(self.paulis) * self.n_qubits

    def measurements(self) -> list[Any]:
        measurements = []
        for pauli in self.paulis:
            op = _VALID_PAULIS[pauli]
            measurements.extend(qml.expval(op(wire)) for wire in range(self.n_qubits))
        return measurements


def build_observable_bank(config: Any, *, n_qubits: int) -> FixedPauliBank:
    """Build the configured observable bank.

    Stage 1 intentionally supports only the fixed Pauli bank.
    """

    if config is None:
        return FixedPauliBank(n_qubits=n_qubits)

    if isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = {
            key: getattr(config, key)
            for key in ("name", "paulis")
            if hasattr(config, key)
        }

    name = str(cfg.get("name", "fixed_pauli")).lower()
    if name != "fixed_pauli":
        raise ValueError(
            f"Unsupported observable bank '{name}'. Stage 1 supports only 'fixed_pauli'."
        )

    paulis = cfg.get("paulis", ("X", "Z"))
    return FixedPauliBank(n_qubits=n_qubits, paulis=tuple(paulis))
