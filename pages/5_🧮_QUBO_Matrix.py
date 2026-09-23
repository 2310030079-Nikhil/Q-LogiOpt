"""
Page 5: QUBO Formulation & Matrix Inspector
"""
import streamlit as st
import numpy as np
import pandas as pd
from quantum import QuboBuilder, HamiltonianConverter
from visualization.charts import ChartVisualizer
from utils.helpers import initialize_session_state

initialize_session_state()

st.title("🧮 QUBO Matrix & Ising Hamiltonian Inspector")
st.markdown("Inspect the mathematical mapping from the constrained logistics routing problem into a Quadratic Unconstrained Binary Optimization (QUBO) matrix.")

nodes = st.session_state.nodes
cmat = st.session_state.cost_matrix
pen_lambda = st.session_state.penalty_lambda

# Build QUBO and Ising
qubo = QuboBuilder.build_qubo(nodes, cmat, penalty_lambda=pen_lambda)
ising = HamiltonianConverter.qubo_to_ising(qubo)
st.session_state.qubo = qubo

# Resource KPI Cards
q_cols = st.columns(5)
q_cols[0].metric("Locations (n)", len(nodes), "1 Depot + Customers")
q_cols[1].metric("Customers (m)", len(nodes)-1)
q_cols[2].metric("QUBO Variables", qubo.num_qubits, f"m² = ({len(nodes)-1})²")
q_cols[3].metric("Required Qubits", ising.num_qubits, "1 Qubit / Variable")
q_cols[4].metric("Penalty Parameter λ", f"{qubo.penalty_lambda:.1f}", "Enforces Feasibility")

st.markdown("---")

tab_heatmap, tab_table, tab_ising, tab_math = st.tabs([
    "🔥 QUBO Heatmap",
    "📊 QUBO Matrix Table",
    "⚛️ Ising Hamiltonian (Pauli-Z)",
    "📐 Mathematical Derivation"
])

with tab_heatmap:
    st.subheader(f"Interactive Heatmap of Upper-Triangular Matrix Q ({qubo.num_qubits} × {qubo.num_qubits})")
    fig_heat = ChartVisualizer.plot_qubo_heatmap(qubo)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab_table:
    st.subheader("Numeric QUBO Matrix Q")
    st.caption("Rows and columns correspond to binary decision variables x[Customer, Step]. Off-diagonal entries represent transition costs and mutual exclusion penalties.")
    df_qubo = QuboBuilder.to_dataframe(qubo)
    st.dataframe(df_qubo, use_container_width=True)
    st.write(f"**Constant Offset (Energy Baseline):** `{qubo.constant_offset:.3f}`")

with tab_ising:
    st.subheader("Ising Cost Hamiltonian: H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j + Offset")
    st.caption("Mapped from QUBO via algebraic substitution x_i = (I - Z_i) / 2.")
    
    col_h, col_j = st.columns(2)
    with col_h:
        st.markdown("#### Single-Qubit Coefficients (h_i Z_i)")
        df_h = pd.DataFrame([
            {"Qubit Index (i)": k, "Variable": qubo.variable_names[k], "Coefficient (h_i)": v}
            for k, v in ising.linear_coeffs.items()
        ])
        st.dataframe(df_h, use_container_width=True)
        
    with col_j:
        st.markdown("#### Two-Qubit Entangling Terms (J_ij Z_i Z_j)")
        df_j = pd.DataFrame([
            {"Qubit Pair (i, j)": f"({i}, {j})", "Variables": f"{qubo.variable_names[i]} ⊗ {qubo.variable_names[j]}", "Coupling (J_ij)": v}
            for (i, j), v in list(ising.quadratic_coeffs.items())[:25]
        ])
        st.dataframe(df_j, use_container_width=True)
        if len(ising.quadratic_coeffs) > 25:
            st.caption(f"Showing first 25 of {len(ising.quadratic_coeffs)} total ZZ coupling pairs.")

with tab_math:
    st.markdown("""
    ### 🔬 Complete Mathematical Transformation
    
    #### 1. Anchored Depot Decision Variables
    Standard Traveling Salesperson formulations require $n^2$ binary variables $x_{i,t} \\in \\{0,1\\}$ denoting whether node $i \\in \\{0, \\dots, n-1\\}$ is visited at step $t \\in \\{0, \\dots, n-1\\}$.
    For $n=4$, this requires $16$ qubits.
    
    **Q-LogiOpt reduces the qubit space by anchoring the depot at step 0:**
    $$x_{0,0} = 1, \\quad x_{0,t} = 0 \\; (\\forall t > 0), \\quad x_{i,0} = 0 \\; (\\forall i > 0)$$
    
    This leaves only $(n-1)$ customer locations $u \\in \\{0, \\dots, m-1\\}$ and $(n-1)$ tour steps $s \\in \\{0, \\dots, m-1\\}$, reducing the required quantum bits to:
    $$N = (n-1)^2$$
    *(Saving $2n - 1$ physical qubits).*
    
    #### 2. Permutation Constraint Penalties
    Every customer must be visited exactly once:
    $$H_{\\text{node}} = \\lambda \\sum_{u=0}^{m-1} \\left( 1 - \\sum_{s=0}^{m-1} x_{u,s} \\right)^2$$
    
    Every tour step must have exactly one customer:
    $$H_{\\text{time}} = \\lambda \\sum_{s=0}^{m-1} \\left( 1 - \\sum_{u=0}^{m-1} x_{u,s} \\right)^2$$
    
    #### 3. Routing Cost Objective
    * Departure leg (Depot to first customer): $\\sum_{u=0}^{m-1} C_{0, u+1} x_{u,0}$
    * Intermediate legs: $\\sum_{s=0}^{m-2} \\sum_{u \\ne v} C_{u+1, v+1} x_{u,s} x_{v,s+1}$
    * Return leg (Last customer back to Depot): $\\sum_{u=0}^{m-1} C_{u+1, 0} x_{u, m-1}$
    
    #### 4. Total QUBO and Ising Transformation
    Grouping linear and quadratic coefficients yields:
    $$\\min_{x \\in \\{0,1\\}^N} x^T Q x + c$$
    Substituting $x_k = \\frac{I - Z_k}{2}$ converts this into the diagonal Ising Cost Hamiltonian $H_C$ whose ground state corresponds to the minimum-cost feasible tour.
    """)
