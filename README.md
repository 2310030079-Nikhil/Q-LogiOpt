# Q-LogiOpt: Quantum Approximate Optimization for Intelligent Logistics Route and Delivery Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 2.x](https://img.shields.io/badge/qiskit-2.5+-613394.svg)](https://qiskit.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest Passed](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()

> **Final-Year Major Project**: A pure software quantum computing research platform introducing the **Quantum Logistics Distribution Objective (QLDO)** and evaluating QAOA against classical heuristics for constrained last-mile logistics routing.

---

## 1. Project Overview

Last-mile delivery and vehicle routing problems—classically formalized as the Traveling Salesperson Problem (TSP) and Vehicle Routing Problem (VRP)—are NP-hard combinatorial challenges that dominate supply chain logistics costs. While classical heuristics (Nearest Neighbor, 2-Opt local search, Simulated Annealing) offer quick approximations, they frequently settle in suboptimal local minima.

**Q-LogiOpt** is an end-to-end hybrid quantum-classical optimization platform that:
1. Ingests or synthetically generates geographic logistics networks with depots, customer demands, and delivery priorities.
2. Converts routing constraints into an **Anchored Depot Quadratic Unconstrained Binary Optimization (QUBO)** matrix, reducing the qubit footprint from $n^2$ to $(n-1)^2$.
3. Maps the QUBO into an **Ising Cost Hamiltonian** solved via **QAOA** on local **Qiskit Aer** simulators.
4. Optimizes variational parameters $(\vec{\gamma}, \vec{\beta})$ using classical optimizers (COBYLA, Nelder-Mead, BFGS).
5. Samples measurement bitstrings, evaluates Constraint Satisfaction Rates (CSR), and decodes routes using both strict filtering and minimum-conflict Hungarian repair.
6. Delivers an interactive **Python + Streamlit** research dashboard with Plotly route maps, convergence curves, and automated batch benchmarking.

---

## 2. System Architecture

```text
Logistics Network (CSV / Synthetic Generator)
                      │
                      ▼
       Distance & Cost Matrix Calculation
     (Haversine Distance + Priority Penalty)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
Classical Optimization        Anchored Depot QUBO Formulation
├── Nearest Neighbor          ├── Variables: (n-1)² qubits
├── 2-Opt Local Search        ├── Node & Step Penalties (λ)
├── Simulated Annealing       └── Objective: min x^T Q x + c
└── Exact Brute Force                     │
          │                               ▼
          │                   Ising Cost Hamiltonian
          │                   H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j
          │                               │
          │                               ▼
          │                   QAOA Parameterized Circuit
          │                   U(γ, β) = e^-iβH_M e^-iγH_C
          │                               │
          │                               ▼
          │                   Hybrid Variational Loop
          │                   Classical Optimizer (COBYLA)
          │                               │
          │                               ▼
          │                   Aer Simulation & Sampling
          │                   Measurement Counts {z : N_z}
          │                               │
          │                               ▼
          │                   Route Decoding & Feasibility
          │                   ├── Strict Permutation Filter
          │                   └── Hungarian Repair Projector
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
            Empirical Comparison & Metrics
            ├── Approximation Ratio (α = C / C*)
            ├── Constraint Satisfaction Rate (CSR)
            ├── Qubit Scaling & Circuit Depth
            └── Wall-Clock Execution Latency
                          │
                          ▼
          Streamlit Interactive Research UI
```

---

## 3. Technology Stack

* **Programming Language**: Python 3.11+ / Python 3.14
* **Quantum Computing Framework**: `Qiskit 2.5.2`, `Qiskit Aer 0.17.2`, `Qiskit Optimization 0.7.0`
* **Mathematical Modeling**: `docplex 2.32`, `scipy 1.17.1`, `numpy 2.4.2`
* **Frontend Web Application**: `Streamlit 1.54.0`
* **Data Visualization**: `Plotly 6.6.0`, `NetworkX 3.6.1`, `Matplotlib 3.10.8`
* **Automated Testing**: `Pytest 9.0.3`

---

## 4. Mathematical Formulation

### 4.1 Anchored Depot Qubit Reduction
In standard TSP, binary decision variable $x_{i,t} \in \{0, 1\}$ indicates if node $i$ is visited at step $t \in \{0, \dots, n-1\}$, requiring $n^2$ qubits (e.g. $n=4 \implies 16$ qubits).
**Q-LogiOpt anchors the depot at step 0:**
$$x_{0,0} = 1, \quad x_{0,t} = 0 \; (\forall t > 0), \quad x_{i,0} = 0 \; (\forall i > 0)$$
This reduces the variable space strictly to customer locations $u \in \{0, \dots, m-1\}$ and customer tour steps $s \in \{0, \dots, m-1\}$ (where $m = n - 1$).
$$\text{Required Qubits } N = (n - 1)^2$$
* For $n = 4$ locations: **9 qubits** (instead of 16).
* For $n = 5$ locations: **16 qubits** (instead of 25).

### 4.2 Constraints & Cost Functions
1. **Customer Visitation Constraint:**
   $$H_{\text{node}} = \lambda \sum_{u=0}^{m-1} \left( 1 - \sum_{s=0}^{m-1} x_{u,s} \right)^2$$
2. **Step Occupancy Constraint:**
   $$H_{\text{time}} = \lambda \sum_{s=0}^{m-1} \left( 1 - \sum_{u=0}^{m-1} x_{u,s} \right)^2$$
3. **Routing Cost:**
   $$H_{\text{cost}} = \sum_{u=0}^{m-1} C_{0, u+1} x_{u,0} + \sum_{s=0}^{m-2} \sum_{u \ne v} C_{u+1, v+1} x_{u,s} x_{v,s+1} + \sum_{u=0}^{m-1} C_{u+1, 0} x_{u, m-1}$$
4. **Ising Spin Glass Hamiltonian:**
   Substituting $x_k = \frac{I - Z_k}{2}$:
   $$H_C = \sum_{k=0}^{N-1} h_k Z_k + \sum_{j < k} J_{jk} Z_j Z_k + \text{offset}$$

---

## 5. Installation and Setup

### Step 1: Clone or Navigate to the Workspace
```bash
cd "e:/Q LogiOpt"
```

### Step 2: Set Up Virtual Environment (Optional but Recommended)
On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 4: Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 6. Project Directory Structure

```text
e:/Q LogiOpt/
├── app.py                      # Main Streamlit Application Entrypoint
├── requirements.txt            # Dependency manifest
├── README.md                   # Academic-grade documentation
├── .gitignore                  # Git exclusions
├── data/
│   ├── sample_routes.csv       # Preloaded Hyderabad metropolitan dataset
│   └── generated/              # Benchmark CSV exports
├── config/
│   └── settings.py             # Global defaults and model parameters
├── classical/
│   ├── nearest_neighbor.py     # Greedy construction heuristic
│   ├── two_opt.py              # 2-Opt edge-swapping local search
│   ├── simulated_annealing.py  # Probabilistic metaheuristic
│   └── exact_solver.py         # Ground truth brute force
├── quantum/
│   ├── qubo.py                 # Anchored QUBO matrix generator
│   ├── hamiltonian.py          # Ising mapping (SparsePauliOp)
│   ├── circuit.py              # Parameterized QAOA ansatz circuit
│   ├── simulator.py            # Aer statevector & shot sampler
│   └── qaoa_optimizer.py       # Hybrid variational parameter loop
├── optimization/
│   ├── objective.py            # Logistics cost and constraint evaluator
│   ├── constraints.py          # Matrix & tour feasibility checker
│   └── route_decoder.py        # Bitstring parser & Hungarian repair
├── data_processing/
│   ├── loader.py               # CSV reader and validator
│   ├── generator.py            # Synthetic cluster generator
│   └── distance_matrix.py      # Haversine & Euclidean calculators
├── qldo/
│   ├── probability.py          # Empirical distribution & raw/central moments
│   ├── feasibility.py          # Constraint checking & -γ log(P_F + ε) barrier
│   ├── concentration.py        # Good-route sets G and quadratic concentration G_θ
│   ├── objective.py            # J_QLDO master evaluation and term decomposition
│   └── metrics.py              # Multi-metric dispersion & CSR telemetry
├── visualization/
│   ├── maps.py                 # Plotly geographic route overlays
│   ├── charts.py               # Convergence & distribution charts
│   └── circuit_visualizer.py   # Circuit metrics and gate tables
├── experiments/
│   ├── qldo_benchmark.py       # Head-to-Head: Standard QAOA vs QLDO-QAOA
│   ├── parameter_sweep.py      # Sweeps for α, β, γ, ε, depth p, and size n
│   ├── noise_experiment.py     # Realistic NISQ depolarizing & readout error
│   └── benchmark_runner.py     # Automated batch experiment suite
├── utils/
│   └── helpers.py              # State management and session init
├── pages/
│   ├── 1_📊_Dashboard.py       # Executive overview & KPI cards
│   ├── 2_📍_Dataset.py         # Dataset upload & synthetic generation
│   ├── 3_⚙️_Route_Configuration.py # Objectives & capacity calibration
│   ├── 4_🏃_Classical.py       # Classical baselines inspection
│   ├── 5_🧮_QUBO_Matrix.py     # QUBO heatmap & Ising viewer
│   ├── 6_⚛️_QAOA_Optimization.py # Live QLDO & Standard QAOA runner
│   ├── 7_🔬_Quantum_Circuit.py # Circuit schematic & gate analysis
│   ├── 8_🧪_Experiments.py     # 4 automated batch experiment tabs
│   ├── 9_📈_Performance.py     # Tripartite comparison & 8 analytical graphs
│   └── 10_📚_About_Research.py # Academic literature review & novelty analysis
└── tests/
    ├── test_classical.py       # Classical solver tests
    ├── test_data_processing.py # Haversine & loader tests
    ├── test_qubo.py            # QUBO energy match tests
    ├── test_quantum.py         # Circuit, Aer & decoder tests
    ├── test_qldo.py            # QLDO mathematical moments & barrier tests
    └── test_experiments.py     # Automated experiment runners tests
```

---

## 7. Experimental Capabilities & Results

The system implements automated batch experiments across 4 dedicated research tabs:
1. **Experiment 1 (Controlled Head-to-Head)**: Under identical seeds, instances, and circuit depths, compares Standard QAOA (expectation $\mathbb{E}[C]$) against QLDO-QAOA (distribution-aware).
2. **Experiments 4–6 (QLDO Hyperparameter Sweeps)**: Evaluates variance regularization $\alpha$, good-route concentration $\beta$, and logarithmic barrier parameters $\gamma, \epsilon$.
3. **Experiments 2 & 3 (Scaling Analysis)**: Evaluates depth scaling $p \in \{1, 2, 3\}$ and problem sizes $n \in \{3, 4, 5\}$ stops.
4. **Experiment 7 (NISQ Noise Robustness)**: Simulates 1-qubit/2-qubit depolarizing gate noise and readout bit-flip error models to quantify degradation under realistic hardware noise.

All experimental tables can be exported as CSV directly from the dashboard.

---

## 8. Academic Rigor & Honest Limitations

* **No False Claims of Quantum Advantage**: On small problem instances ($n \le 8$), classical heuristics (2-opt, SA) run in sub-millisecond time and achieve near-perfect solution quality. QAOA on local simulators introduces classical parameter optimization and statevector simulation overhead.
* **Statevector Exponential Scaling**: Simulating $N$ qubits requires tracking $2^N$ complex amplitudes ($N=16 \implies 65,536$ amplitudes; $N=25 \implies 33.5 \text{ million}$; $N=36 \implies \approx 68.7 \text{ billion}$). Consumer laptops can comfortably simulate up to $N \approx 20$.
* **The Infeasible Sampling Phenomenon**: Because QAOA explores an unconstrained Hilbert space, only $(n-1)!$ of $2^{(n-1)^2}$ states correspond to valid routing permutations. Q-LogiOpt transparently reports this Constraint Satisfaction Rate (CSR) and demonstrates how QLDO and hybrid classical repair (Hungarian matching) stabilize NISQ outputs.

---

## 9. Verification & Automated Testing

All modules have been verified using Pytest:
```bash
$ python -m pytest tests/ -v
======================== 26 passed, 1 warning in 6.46s ========================
```

---

## 10. License & Citation

Distributed under the MIT License. Developed as a major university final-year project demonstrating hybrid quantum-classical optimization for intelligent logistics route engineering.
