"""
Page 10: Academic Research Guide and Theoretical Foundations
"""
import streamlit as st

st.title("📚 Academic Research Guide & Educational Theory")
st.markdown("Comprehensive pedagogical and theoretical documentation for evaluators, researchers, and students.")

st.markdown("""
---

### 1. Research Overview & Problem Context
Logistics route planning (Traveling Salesperson Problem and Vehicle Routing Problem) is a cornerstone of operations research and global supply chain management. These problems belong to the class **NP-hard**: the computational complexity of finding an exact global optimum scales factorially $(n-1)!$ with the number of delivery stops.

In the **Noisy Intermediate-Scale Quantum (NISQ)** era, the **Quantum Approximate Optimization Algorithm (QAOA)** has emerged as the leading variational hybrid algorithm for solving combinatorial problems on quantum processors.

---

### 2. Core Concepts Explained from First Principles

#### A. What is a Qubit?
Unlike a classical bit which is strictly deterministic (0 or 1), a quantum bit (**qubit**) can exist in a linear superposition of both orthogonal basis states:
$$|\\psi\\rangle = \\alpha |0\\rangle + \\beta |1\\rangle, \\quad |\\alpha|^2 + |\\beta|^2 = 1$$
In Q-LogiOpt, an $N$-qubit quantum register represents $2^N$ simultaneous route assignments in a single unified quantum state.

#### B. What is QUBO?
**Quadratic Unconstrained Binary Optimization (QUBO)** is a mathematical problem format:
$$\\min_{x \\in \\{0, 1\\}^N} x^T Q x + c$$
where $Q$ is an upper-triangular real matrix. Constrained optimization problems are mapped into QUBO by converting hard constraints (e.g. visiting each customer once) into soft penalty functions added to the objective.

#### C. What is the Cost Hamiltonian ($H_C$)?
By substituting $x_i = \\frac{I - Z_i}{2}$, the binary quadratic program is transformed into an **Ising Spin Glass Hamiltonian**:
$$H_C = \\sum_{i} h_i Z_i + \\sum_{i < j} J_{ij} Z_i Z_j + \\text{offset}$$
Because $Z |0\\rangle = +1 |0\\rangle$ and $Z |1\\rangle = -1 |1\\rangle$, the energy eigenvalue of $H_C$ for any computational basis state $|z\\rangle$ equals the penalized logistics route cost of bitstring $z$. The ground state (lowest energy state) encodes the optimal route.

#### D. What is the Mixer Hamiltonian ($H_M$)?
To explore the search space, QAOA applies a transverse magnetic field:
$$H_M = \\sum_{i=0}^{N-1} X_i$$
The mixer driver non-commutatively flips qubit states ($X|0\\rangle = |1\\rangle$), enabling quantum tunneling through high-energy penalty barriers that trap classical local search heuristics.

#### E. What does QAOA Depth ($p$) Mean?
QAOA approximates the adiabatic quantum evolution by alternating $p$ layers of Cost and Mixer unitaries:
$$|\\psi(\\vec{\\gamma}, \\vec{\\beta})\\rangle = \\left( \\prod_{l=1}^p e^{-i \\beta_l H_M} e^{-i \\gamma_l H_C} \\right) |+\\rangle^{\\otimes N}$$
* For $p=1$: A shallow circuit with 2 variational angles $(\\gamma_1, \\beta_1)$.
* For $p \\ge 2$: Greater entanglement and expressibility, allowing the quantum state to concentrate higher probability onto optimal tours, at the expense of circuit depth and optimizer parameter landscape complexity.

---

### 3. Empirical Findings & Honest NISQ Limitations

#### A. Simulation vs. Physical Hardware
This project runs on local high-performance simulators (`Qiskit AerSimulator`). Statevector simulation maintains a statevector of $2^N$ complex amplitudes:
* $N = 9$ qubits ($n=4$ stops): $2^9 = 512$ amplitudes (instantaneous, < 0.1 MB).
* $N = 16$ qubits ($n=5$ stops): $2^{16} = 65,536$ amplitudes (~1 MB).
* $N = 25$ qubits ($n=6$ stops): $2^{25} = 33.5$ million amplitudes (~512 MB).
* $N = 36$ qubits ($n=7$ stops): $2^{36} = 68.7$ billion amplitudes (~1 TB RAM, infeasible on laptops).

#### B. The Infeasible Sampling Challenge
Because QAOA explores an unconstrained Hilbert space of dimension $2^{(n-1)^2}$, only $(n-1)!$ of those states correspond to valid permutations. For $n=4$, there are $3! = 6$ valid tours out of $2^9 = 512$ total states (only 1.17% of state space is feasible!).
* If penalty $\\lambda$ is too low, the quantum state collapses into invalid bitstrings.
* If penalty $\\lambda$ is too high, the routing cost landscape is drowned out.
* **Q-LogiOpt resolves this** by combining calibrated penalty tuning with hybrid classical Hungarian repair, ensuring valid routes while tracking raw constraint satisfaction rates (CSR).

#### C. Quantum Advantage Reality Check
No current NISQ algorithm demonstrates quantum advantage for small logistics instances over classical algorithms (Nearest Neighbor and 2-Opt run in sub-millisecond time). The research significance of QAOA lies in studying parameterized quantum state preparation, barren plateau avoidance, and preparing hybrid algorithms for fault-tolerant quantum computers.

---

### 4. Proposed Formulation: QLDO (Quantum Logistics Distribution Objective)

Standard QAOA optimizes the expectation value of the Hamiltonian $J_{\\text{standard}} = \\sum p_{\\theta}(r) C(r)$. In this research, we propose and investigate an alternative multi-moment objective:

$$\\boxed{J_{\\text{QLDO}}(\\boldsymbol{\\theta}) = \\mathbb{E}_{p_{\\boldsymbol{\\theta}}}[C] + \\alpha \\operatorname{Var}_{p_{\\boldsymbol{\\theta}}}(C) - \\beta G_{\\boldsymbol{\\theta}} - \\gamma \\log(P_F(\\boldsymbol{\\theta}) + \\epsilon)}$$

#### Mathematical Decomposition:
1. **$\\mathbb{E}[C]$ (Expected Cost)**: Center of mass driving the distribution towards the ground state.
2. **$+\\alpha \\operatorname{Var}(C)$ (Variance Penalty)**: Second central moment penalizing statistical dispersion and multimodal scatter.
3. **$-\\beta G_{\\boldsymbol{\\theta}}$ (Good-Route Concentration)**: Herfindahl-Hirschman index $G = \\sum_{\\mathcal{G}} p(r)^2$ over the high-utility feasible set $\\mathcal{G}$, rewarding peaked probability distributions on optimal tours.
4. **$-\\gamma \\log(P_F + \\epsilon)$ (Logarithmic Feasibility Barrier)**: Interior-point barrier that dynamically imposes steep penalty walls when $P_F \\to 0$, forcing optimizers into the valid permutation manifold.

#### Novelty & Literature Comparison Matrix:

| Literature Reference | Objective Paradigm | Feasibility-Aware | Variance Regularization | Distribution Concentration | Logistics Routing Domain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Farhi et al. (2014)** | Expectation $\\langle H_C \\rangle$ | No | No | No | Max-Cut |
| **Barkoutsos et al. (2020)** | CVaR $\\alpha$-tail | No | No | No | Portfolio / Max-Cut |
| **Hadfield et al. (2019)** | Subspace XY-Mixer | Yes (Circuit) | No | No | Graph Coloring / TSP |
| **Egger et al. (2021)** | Mean-Variance | No | Yes | No | Financial Portfolio |
| **Proposed QLDO-QAOA** | **Distribution-Aware** | **Yes (Log Barrier)** | **Yes** | **Yes (Quadratic $G$)** | **Constrained Logistics (TSP/VRP)** |

---

### 5. Future Research Directions
1. **Cloud Execution on IBM Quantum**: Connecting Q-LogiOpt to IBM Quantum Heron/Eagle processors via `qiskit-ibm-runtime`.
2. **Warm-Starting QAOA**: Initializing quantum superposition biased towards classical 2-Opt heuristic routes rather than uniform $|+\\rangle$.
3. **Multi-Vehicle CVRP Decomposition**: Partitioning customer clusters into vehicle sub-problems before mapping to quantum registers.
""")
