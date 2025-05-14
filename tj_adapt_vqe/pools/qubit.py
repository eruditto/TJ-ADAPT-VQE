from itertools import combinations

from openfermion import MolecularData, QubitOperator
from qiskit.quantum_info.operators.linear_op import LinearOp  # type: ignore
from typing_extensions import Any, Self, override

from ..utils import openfermion_to_qiskit
from .pool import Pool


class QubitPool(Pool):
    """
    The qubit pool, which consists of the individual Pauli strings
    of the jordan wigner form of the operators in the GSD/QEB pools.
    """

    def __init__(self: Self, molecule: MolecularData, n_excitations: int) -> None:
        super().__init__("Qubit Pool", molecule)

        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.n_excitations = n_excitations

        self.operators, self.labels = self.make_operators_and_labels()

    def make_operators_and_labels(self: Self) -> tuple[list[LinearOp], list[str]]:
        """
        The method that generates the pool operators for the molecule as well as a label for each operator
        Should return a tuple of two equal length lists, where each element in the first list
        is the pool operator and each element in the second list is the label for that operator
        """
        operators = []
        labels = []
        
        for i in range(self.n_qubits):
            for j in range(i+1,self.n_qubits):
                s1 = openfermion_to_qiskit(QubitOperator(((i, 'Y'), (j, 'X')), 1j),self.n_qubits)
                s2 = openfermion_to_qiskit(QubitOperator(((i, 'X'), (j, 'Y')), 1j),self.n_qubits)
                operators.append(s1)
                operators.append(s2)

                for k in range(j+1,self.n_qubits):
                    for l in range(k+1,self.n_qubits):
                        d1 = openfermion_to_qiskit(QubitOperator(((i, 'X'),(k, 'X'), (l,'X'), (j, 'Y')), 1j),self.n_qubits)
                        d2 = openfermion_to_qiskit(QubitOperator(((j, 'X'),(k, 'X'), (l,'X'), (i, 'Y')), 1j),self.n_qubits)
                        d3 = openfermion_to_qiskit(QubitOperator(((i, 'Y'),(j, 'Y'), (k,'Y'), (l, 'X')), 1j),self.n_qubits)
                        d4 = openfermion_to_qiskit(QubitOperator(((i, 'Y'),(j, 'Y'), (l,'Y'), (k, 'X')), 1j),self.n_qubits)
                        d5 = openfermion_to_qiskit(QubitOperator(((i, 'X'),(j, 'X'), (l,'X'), (k, 'Y')), 1j),self.n_qubits)
                        d6 = openfermion_to_qiskit(QubitOperator(((i, 'X'),(j, 'X'), (k,'X'), (l, 'Y')), 1j),self.n_qubits)
                        d7 = openfermion_to_qiskit(QubitOperator(((i, 'Y'),(k, 'Y'), (l,'Y'), (j, 'X')), 1j),self.n_qubits)
                        d8 = openfermion_to_qiskit(QubitOperator(((j, 'Y'),(k, 'Y'), (l,'Y'), (i, 'X')), 1j),self.n_qubits)
                        operators.append(d1)
                        operators.append(d2)
                        operators.append(d3)
                        operators.append(d4)
                        operators.append(d5)
                        operators.append(d6)
                        operators.append(d7)
                        operators.append(d8)

        return operators, labels

    def _concat_ops(self: Self, ops: list[QubitOperator]) -> QubitOperator:
        k = ops[0]
        for i in range(1, len(ops)):
            k = k * ops[i]
        return k

    @override
    def get_op(self: Self, idx: int) -> LinearOp:
        return self.operators[idx]

    @override
    def get_label(self: Self, idx: int) -> str:
        return self.labels[idx]

    @override
    def to_config(self: Self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_electrons": self.n_electrons,
            "n_excitations": self.n_excitations,
        }

    @override
    def __len__(self: Self) -> int:
        return len(self.operators)