from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pennylane as qml
from omegaconf import DictConfig, OmegaConf

_VALID_PAULIS = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
}


@runtime_checkable
class ObservableBank(Protocol):
    @property
    def n_qubits(self) -> int: ...

    @property
    def output_dim(self) -> int: ...

    @property
    def num_terms(self) -> int: ...

    @property
    def max_locality(self) -> int: ...

    @property
    def terms(self) -> tuple[str, ...]: ...

    @property
    def topology(self) -> str: ...

    @property
    def measurement_groups_estimate(self) -> int: ...

    def measurements(self) -> list[Any]: ...

    def metadata(self) -> dict[str, Any]: ...


def _normalise_paulis(paulis: tuple[str, ...], *, bank_name: str) -> tuple[str, ...]:
    normalised = tuple(str(pauli).upper() for pauli in paulis)
    if not normalised:
        raise ValueError(f"{bank_name} requires at least one Pauli label.")
    invalid = [pauli for pauli in normalised if pauli not in _VALID_PAULIS]
    if invalid:
        valid = ", ".join(sorted(_VALID_PAULIS))
        raise ValueError(
            f"Invalid Pauli label(s) for {bank_name}: {invalid}. "
            f"Expected labels from {{{valid}}}."
        )
    return normalised


def _normalise_two_body_paulis(
    two_body_paulis: tuple[str, ...], *, bank_name: str
) -> tuple[str, ...]:
    normalised = tuple(str(pauli).upper() for pauli in two_body_paulis)
    if not normalised:
        raise ValueError(f"{bank_name} requires at least one two-body Pauli label.")
    malformed = [pauli for pauli in normalised if len(pauli) != 2]
    if malformed:
        raise ValueError(
            f"Malformed two-body Pauli label(s) for {bank_name}: {malformed}. "
            "Each term must contain exactly two Pauli labels."
        )
    for term in normalised:
        _normalise_paulis(tuple(term), bank_name=bank_name)
    return normalised


def _line_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    return tuple((wire, wire + 1) for wire in range(n_qubits - 1))


def _single_terms(paulis: tuple[str, ...], n_qubits: int) -> tuple[str, ...]:
    return tuple(f"{pauli}_{wire}" for pauli in paulis for wire in range(n_qubits))


def _two_body_terms(
    two_body_paulis: tuple[str, ...], pairs: tuple[tuple[int, int], ...]
) -> tuple[str, ...]:
    return tuple(
        f"{pauli}_{left}_{right}"
        for pauli in two_body_paulis
        for left, right in pairs
    )


def _term_measurements(terms: tuple[str, ...]) -> list[Any]:
    measurements: list[Any] = []
    for term in terms:
        parts = term.split("_")
        pauli_label = parts[0]
        wires = tuple(int(wire) for wire in parts[1:])
        if len(wires) == 1:
            measurements.append(qml.expval(_VALID_PAULIS[pauli_label](wires[0])))
        elif len(wires) == 2 and len(pauli_label) == 2:
            left = _VALID_PAULIS[pauli_label[0]](wires[0])
            right = _VALID_PAULIS[pauli_label[1]](wires[1])
            measurements.append(qml.expval(left @ right))
        else:
            raise ValueError(f"Unsupported observable term format: {term!r}.")
    return measurements


def _metadata(bank: ObservableBank, *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "output_dim": bank.output_dim,
        "n_qubits": bank.n_qubits,
        "num_terms": bank.num_terms,
        "max_locality": bank.max_locality,
        "terms": list(bank.terms),
        "topology": bank.topology,
        "measurement_groups_estimate": bank.measurement_groups_estimate,
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
        object.__setattr__(
            self,
            "paulis",
            _normalise_paulis(self.paulis, bank_name="FixedPauliBank"),
        )

    @property
    def output_dim(self) -> int:
        return len(self.paulis) * self.n_qubits

    @property
    def name(self) -> str:
        return "fixed_pauli"

    @property
    def num_terms(self) -> int:
        return self.output_dim

    @property
    def max_locality(self) -> int:
        return 1

    @property
    def terms(self) -> tuple[str, ...]:
        return _single_terms(self.paulis, self.n_qubits)

    @property
    def topology(self) -> str:
        return "local"

    @property
    def measurement_groups_estimate(self) -> int:
        return len(set(self.paulis))

    def measurements(self) -> list[Any]:
        measurements = []
        for pauli in self.paulis:
            op = _VALID_PAULIS[pauli]
            measurements.extend(qml.expval(op(wire)) for wire in range(self.n_qubits))
        return measurements

    def metadata(self) -> dict[str, Any]:
        return _metadata(self, name=self.name)


@dataclass(frozen=True)
class LocalPauliBank:
    """Configurable per-qubit Pauli readout bank."""

    n_qubits: int
    paulis: tuple[str, ...] = ("X", "Y", "Z")

    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {self.n_qubits}.")
        object.__setattr__(
            self,
            "paulis",
            _normalise_paulis(self.paulis, bank_name="LocalPauliBank"),
        )

    @property
    def output_dim(self) -> int:
        return len(self.terms)

    @property
    def name(self) -> str:
        return "local_pauli"

    @property
    def num_terms(self) -> int:
        return self.output_dim

    @property
    def max_locality(self) -> int:
        return 1

    @property
    def terms(self) -> tuple[str, ...]:
        return _single_terms(self.paulis, self.n_qubits)

    @property
    def topology(self) -> str:
        return "local"

    @property
    def measurement_groups_estimate(self) -> int:
        return len(set(self.paulis))

    def measurements(self) -> list[Any]:
        return _term_measurements(self.terms)

    def metadata(self) -> dict[str, Any]:
        return _metadata(self, name=self.name)


@dataclass(frozen=True)
class TwoBodyPauliBank:
    """Nearest-neighbour two-body Pauli readout bank on line topology."""

    n_qubits: int
    two_body_paulis: tuple[str, ...] = ("XX", "ZZ")
    topology: str = "line"

    def __post_init__(self) -> None:
        if self.n_qubits <= 1:
            raise ValueError(f"n_qubits must be greater than one, got {self.n_qubits}.")
        topology = str(self.topology).lower()
        if topology != "line":
            raise ValueError(f"Unsupported topology '{self.topology}'. Only 'line' is supported.")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self,
            "two_body_paulis",
            _normalise_two_body_paulis(
                self.two_body_paulis,
                bank_name="TwoBodyPauliBank",
            ),
        )

    @property
    def output_dim(self) -> int:
        return len(self.terms)

    @property
    def name(self) -> str:
        return "two_body_pauli"

    @property
    def num_terms(self) -> int:
        return self.output_dim

    @property
    def max_locality(self) -> int:
        return 2

    @property
    def terms(self) -> tuple[str, ...]:
        return _two_body_terms(self.two_body_paulis, _line_pairs(self.n_qubits))

    @property
    def measurement_groups_estimate(self) -> int:
        return len(set(self.two_body_paulis))

    def measurements(self) -> list[Any]:
        return _term_measurements(self.terms)

    def metadata(self) -> dict[str, Any]:
        return _metadata(self, name=self.name)


@dataclass(frozen=True)
class MixedPauliBank:
    """Local Pauli block followed by nearest-neighbour two-body Pauli block."""

    n_qubits: int
    paulis: tuple[str, ...] = ("X", "Z")
    two_body_paulis: tuple[str, ...] = ("XX", "ZZ")
    topology: str = "line"

    def __post_init__(self) -> None:
        if self.n_qubits <= 1:
            raise ValueError(f"n_qubits must be greater than one, got {self.n_qubits}.")
        topology = str(self.topology).lower()
        if topology != "line":
            raise ValueError(f"Unsupported topology '{self.topology}'. Only 'line' is supported.")
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self,
            "paulis",
            _normalise_paulis(self.paulis, bank_name="MixedPauliBank"),
        )
        object.__setattr__(
            self,
            "two_body_paulis",
            _normalise_two_body_paulis(
                self.two_body_paulis,
                bank_name="MixedPauliBank",
            ),
        )

    @property
    def output_dim(self) -> int:
        return len(self.terms)

    @property
    def name(self) -> str:
        return "mixed_pauli"

    @property
    def num_terms(self) -> int:
        return self.output_dim

    @property
    def max_locality(self) -> int:
        return 2

    @property
    def terms(self) -> tuple[str, ...]:
        return _single_terms(self.paulis, self.n_qubits) + _two_body_terms(
            self.two_body_paulis,
            _line_pairs(self.n_qubits),
        )

    @property
    def measurement_groups_estimate(self) -> int:
        return len(set(self.paulis) | set(self.two_body_paulis))

    def measurements(self) -> list[Any]:
        return _term_measurements(self.terms)

    def metadata(self) -> dict[str, Any]:
        return _metadata(self, name=self.name)


def build_observable_bank(config: Any, *, n_qubits: int) -> ObservableBank:
    """Build the configured observable bank."""

    if config is None:
        return FixedPauliBank(n_qubits=n_qubits)

    if isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = {
            key: getattr(config, key)
            for key in ("name", "paulis", "two_body_paulis", "topology")
            if hasattr(config, key)
        }

    name = str(cfg.get("name", "fixed_pauli")).lower()
    paulis = tuple(cfg.get("paulis", ("X", "Z")))
    two_body_paulis = tuple(cfg.get("two_body_paulis", ("XX", "ZZ")))
    topology = str(cfg.get("topology", "line"))

    if name == "fixed_pauli":
        return FixedPauliBank(n_qubits=n_qubits, paulis=paulis)
    if name == "local_pauli":
        return LocalPauliBank(n_qubits=n_qubits, paulis=paulis)
    if name == "two_body_pauli":
        return TwoBodyPauliBank(
            n_qubits=n_qubits,
            two_body_paulis=two_body_paulis,
            topology=topology,
        )
    if name == "mixed_pauli":
        return MixedPauliBank(
            n_qubits=n_qubits,
            paulis=paulis,
            two_body_paulis=two_body_paulis,
            topology=topology,
        )

    raise ValueError(
        f"Unsupported observable bank '{name}'. Expected one of "
        "{'fixed_pauli', 'local_pauli', 'two_body_pauli', 'mixed_pauli'}."
    )
