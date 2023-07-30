import sys
from typing import Any, Sequence

import numpy as np
from openfermion import QubitOperator
from openfermion.transforms import jordan_wigner, normal_ordered
from openfermion.utils import load_operator, hermitian_conjugated
from openfermion.ops.operators.fermion_operator import FermionOperator
from quri_parts.algo.optimizer import NFT
from quri_parts.circuit import QuantumGate, UnboundParametricQuantumCircuit
from quri_parts.core.estimator import ConcurrentParametricQuantumEstimator
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.operator import Operator, SinglePauli, PauliLabel, PAULI_IDENTITY
from quri_parts.core.sampling.shots_allocator import create_equipartition_shots_allocator
from quri_parts.core.state import ComputationalBasisState, ParametricCircuitQuantumState
from quri_parts.openfermion.operator import operator_from_openfermion_op

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, TimeExceededError

challenge_sampling = ChallengeSampling(noise=True)


class ADAPT_VQE:
    def __init__(self, hamiltonian: Operator, n_qubits):
        self.shots_allocator = None
        self.measurement_factory = None
        self.parametric_state = None
        self.qubit_pool: list[Operator] = []
        self.fermion_pool: list[FermionOperator] = []
        self.combined_operators: list[Operator] = []
        self.combined_operators_len: list[int] = []
        self.ansatz_operators: list[Operator] = []
        self.ansatz_circuit: UnboundParametricQuantumCircuit = None
        self.hf_gates: Sequence[QuantumGate] = None
        self.sampling_estimator: ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState] = None
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.optimizer: NFT = None
        self.params = np.asarray([])
        self.estimate_result = float("inf")

    def init_fermion_pool(self, orbitalNumber):
        singlet_gsd = []

        for p in range(0, orbitalNumber):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, orbitalNumber):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)
                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    singlet_gsd.append(termA)

        pq = -1
        for p in range(0, orbitalNumber):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, orbitalNumber):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, orbitalNumber):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, orbitalNumber):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            singlet_gsd.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            singlet_gsd.append(termB)

        self.fermion_pool = singlet_gsd

    def init_qubit_pool(self):
        pool = self.fermion_pool
        qubitPool = []

        for fermionOp in pool:
            qubitOp = jordan_wigner(fermionOp)
            for pauli in qubitOp.terms:
                new_pauli = ()
                for qubit, gate in pauli:
                    if gate != 'Z':
                        new_pauli += ((qubit, gate),)
                qubitOp = QubitOperator(new_pauli, 1j)
                found = False
                for exist_op in qubitPool:
                    if exist_op.terms == qubitOp.terms:
                        found = True
                        break
                if not found:
                    qubitPool.append(qubitOp)

        self.qubit_pool = [operator_from_openfermion_op(qubitOp) for qubitOp in qubitPool]

    def prepare(self):
        initial_bits = 0b00000000
        if self.n_qubits == 4:
            initial_bits = 0b0011
        elif self.n_qubits == 8:
            initial_bits = 0b00001111
        self.hf_gates = ComputationalBasisState(self.n_qubits, bits=initial_bits).circuit.gates
        self.init_fermion_pool(self.n_qubits // 2)
        self.init_qubit_pool()
        self.combined_operators = [self.hamiltonian * op for op in self.qubit_pool]
        self.shots_allocator = create_equipartition_shots_allocator()
        self.measurement_factory = bitwise_commuting_pauli_measurement
        for op in self.combined_operators:
            measurements = self.measurement_factory(op)
            measurements = [m for m in measurements if m.pauli_set != {PAULI_IDENTITY}]
            pauli_sets = tuple(m.pauli_set for m in measurements)
            self.combined_operators_len.append(len(pauli_sets))
        measurements = self.measurement_factory(self.hamiltonian)
        measurements = [m for m in measurements if m.pauli_set != {PAULI_IDENTITY}]
        pauli_sets = tuple(m.pauli_set for m in measurements)
        self.sampling_estimator = challenge_sampling.create_concurrent_parametric_sampling_estimator(
            len(pauli_sets) * 12, self.measurement_factory, self.shots_allocator, "it"
        )
        self.params = []
        self.construct_parametric_circuit()
        self.parametric_state = ParametricCircuitQuantumState(self.n_qubits, self.ansatz_circuit)
        self.optimizer = NFT(randomize=True)

    def get_operator_gradient(self, index):
        op_grad_estimator = challenge_sampling.create_parametric_sampling_estimator(
            self.combined_operators_len[index], self.measurement_factory, self.shots_allocator, "it"
        )
        est_value = op_grad_estimator(
            self.combined_operators[index], self.parametric_state, self.params
        )
        return 2 * est_value.value.real

    def select_operator(self) -> Operator:
        selected_gradient_abs = 0
        selected_index = None
        for i in range(len(self.qubit_pool)):
            gradient_abs = np.abs(self.get_operator_gradient(i))
            if gradient_abs > selected_gradient_abs:
                selected_gradient_abs = gradient_abs
                selected_index = i
        return self.qubit_pool[selected_index]

    def construct_parametric_circuit(self):
        self.ansatz_circuit = UnboundParametricQuantumCircuit(self.n_qubits)
        self.ansatz_circuit.extend(self.hf_gates)
        if len(self.ansatz_operators) == 0:
            return
        pauliLabel: PauliLabel = None
        for op in self.ansatz_operators:
            for pLabel in op:
                pauliLabel = pLabel
                break
            involved_qubits = [index for index, gate in pauliLabel]
            involved_qubits.sort()
            for index, gate in pauliLabel:
                match gate:
                    case SinglePauli.X:
                        self.ansatz_circuit.add_H_gate(index)
                    case SinglePauli.Y:
                        self.ansatz_circuit.add_RX_gate(index, np.pi / 2)
            for i in range(len(involved_qubits) - 1):
                self.ansatz_circuit.add_CNOT_gate(involved_qubits[i], involved_qubits[i + 1])
            last_qubit = involved_qubits[-1]
            self.ansatz_circuit.add_ParametricRZ_gate(last_qubit)
            for i in range(len(involved_qubits) - 1, 0, -1):
                self.ansatz_circuit.add_CNOT_gate(involved_qubits[i - 1], involved_qubits[i])
            for index, gate in pauliLabel:
                match gate:
                    case SinglePauli.X:
                        self.ansatz_circuit.add_H_gate(index)
                    case SinglePauli.Y:
                        self.ansatz_circuit.add_RX_gate(index, -np.pi / 2)

    def cost_fn(self, param_values):
        estimate = self.sampling_estimator(self.hamiltonian, self.parametric_state, [param_values])
        return estimate[0].value.real

    def g_fn(self, param_values):
        grad = parameter_shift_gradient_estimates(
            self.hamiltonian, self.parametric_state, param_values, self.sampling_estimator
        )
        return np.asarray([i.real for i in grad.values])

    def run(self):
        n_iter = 0
        while True:
            try:
                print(f"STEP 1: {challenge_sampling.total_quantum_circuit_time}")
                selected_operator = self.select_operator()
                print(f"STEP 2: {challenge_sampling.total_quantum_circuit_time}")
                self.ansatz_operators.append(selected_operator)
                self.construct_parametric_circuit()
                self.parametric_state = ParametricCircuitQuantumState(self.n_qubits, self.ansatz_circuit)
                self.params = np.append(self.params, 0.0)
                opt_state = self.optimizer.get_init_state(self.params)
                opt_state = self.optimizer.step(opt_state, self.cost_fn, self.g_fn)
                print(f"STEP 3: {challenge_sampling.total_quantum_circuit_time}")
                self.params = opt_state.params
                n_iter += 1
                print(f"iteration {n_iter}")
                print(opt_state.cost)
                if opt_state.cost < self.estimate_result:
                    self.estimate_result = opt_state.cost
                    print("Update estimate result")
            except TimeExceededError:
                return


class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H_1",
            data_directory="../hamiltonian/hamiltonian_samples",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        adapt_vqe = ADAPT_VQE(hamiltonian, n_qubits)
        adapt_vqe.prepare()
        adapt_vqe.run()

        return adapt_vqe.estimate_result


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
