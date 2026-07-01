from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import jax.numpy as jnp
import pennylane as qml

from .observables import FixedPauliBank

@dataclass(frozen=True)
class CircuitConfig:
    n_qubits: int
    depth: int
    observable_bank: FixedPauliBank | None = None

    def __post_init__(self) -> None:
        if self.observable_bank is None:
            object.__setattr__(self, "observable_bank", FixedPauliBank(self.n_qubits))
        elif self.observable_bank.n_qubits != self.n_qubits:
            object.__setattr__(self, "observable_bank", FixedPauliBank(self.n_qubits, self.observable_bank.paulis))

def make_style_based_circuit(config: CircuitConfig) -> Callable:
    """
    Creates a style-based quantum circuit using PennyLane's JAX-compatible device.
    """
    # default.qubit is JAX-compatible and avoids NumPy conversion inside JIT.
    dev = qml.device("default.qubit", wires=config.n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(angles: jnp.ndarray) -> List[jnp.ndarray]:
        # angles shape: (depth, n_qubits, 2)

        for d in range(config.depth):
            for i in range(config.n_qubits):
                theta_y = angles[d, i, 0]
                theta_z = angles[d, i, 1]
                qml.RY(theta_y, wires=i)
                qml.RZ(theta_z, wires=i)

            for i in range(config.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Return configured observable-bank measurements.
        return config.observable_bank.measurements()

    return circuit
