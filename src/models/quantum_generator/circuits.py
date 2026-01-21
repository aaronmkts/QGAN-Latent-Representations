from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import pennylane as qml


@dataclass(frozen=True)
class CircuitConfig:
    n_qubits: int
    depth: int


def make_circuit(architecture_name: str, config: CircuitConfig) -> Callable:
    name = architecture_name.lower()
    if name == "fullyentangling":
        entangle = _fully_entangling
    elif name == "simpleentangling":
        entangle = _simple_entangling
    else:
        raise ValueError(f"Unknown circuit architecture: {architecture_name}")

    dev = qml.device("default.qubit", wires=config.n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(q_params: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
        for i in range(config.n_qubits):
            qml.RY(angles[i], wires=i)
        for layer in range(config.depth):
            for i in range(config.n_qubits):
                qml.Rot(
                    q_params[layer, i, 0],
                    q_params[layer, i, 1],
                    q_params[layer, i, 2],
                    wires=i,
                )
            entangle(config.n_qubits)
        return [qml.expval(qml.PauliZ(i)) for i in range(config.n_qubits)]

    return circuit


def _simple_entangling(n_qubits: int) -> None:
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def _fully_entangling(n_qubits: int) -> None:
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            qml.CNOT(wires=[i, j])
