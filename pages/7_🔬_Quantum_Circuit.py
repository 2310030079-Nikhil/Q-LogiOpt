"""
Page 7: Quantum Circuit Schematics and Gate Resource Accounting
"""
import streamlit as st
import pandas as pd
from quantum import QuboBuilder, HamiltonianConverter, QAOACircuitBuilder
from visualization.charts import ChartVisualizer
from visualization.circuit_visualizer import CircuitVisualizer
from utils.helpers import initialize_session_state

initialize_session_state()

st.title("🔬 Quantum Circuit Schematics & Resource Analysis")
st.markdown("Detailed gate-level accounting and schematic visualization of the parameterized QAOA ansatz circuit.")

nodes = st.session_state.nodes
cmat = st.session_state.cost_matrix
res = st.session_state.get("qaoa_result")
aq_res = st.session_state.get("aq_res")
qubo = st.session_state.qubo
if qubo is None:
    qubo = QuboBuilder.build_qubo(nodes, cmat)
    st.session_state.qubo = qubo
ising = HamiltonianConverter.qubo_to_ising(qubo)

# Preview depth selector
col_p, _ = st.columns([3, 7])
with col_p:
    preview_p = st.selectbox("Circuit Preview Depth (p)", options=[1, 2, 3], index=0)

# Build sample circuit for inspection
qc_sample, metrics = QAOACircuitBuilder.build_qaoa_circuit(
    hamiltonian=ising,
    p=preview_p,
    include_measurements=True
)

# Metric Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Qubit Count (N)", metrics.num_qubits, f"Anchored ({len(nodes)-1})²")
c2.metric("Circuit Depth", metrics.circuit_depth, f"p={preview_p} layers")
c3.metric("Total Quantum Gates", metrics.total_gates)
c4.metric("Entangling Two-Qubit Gates", metrics.two_qubit_gates, "RZZ / CX")

st.markdown("---")

tab_hist, tab_diag, tab_gates, tab_explain = st.tabs([
    "📊 Measurement Histogram",
    "🔲 Circuit Schematic (ASCII)",
    "⚙️ Gate Breakdown",
    "📚 Circuit Component Guide"
])

with tab_hist:
    st.subheader("Measurement Probability Distribution")
    active_sim_res = res.simulation_result if res else (aq_res.simulation_result if aq_res else None)
    if active_sim_res:
        valid_set = set([s.bitstring for s in (st.session_state.decoded_states or (aq_res.all_valid_states if aq_res else [])) if s.is_valid_permutation])
        fig_dist = ChartVisualizer.plot_measurement_distribution(
            counts=active_sim_res.counts,
            top_k=20,
            valid_bitstrings=valid_set
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.caption("🟢 Green: strictly valid permutation bitstrings. 🔴 Red: infeasible bitstrings (subject to penalty).")
    else:
        st.info("Run QAOA on the 'QAOA Optimization' page to generate measurement probability histograms.")

with tab_diag:
    st.subheader("Qiskit Circuit Schematic (Text Representation)")
    ascii_art = QAOACircuitBuilder.draw_ascii_circuit(qc_sample)
    st.code(ascii_art, language="text")

with tab_gates:
    st.subheader("Gate Breakdown and Resource Requirements")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### High-Level Metrics")
        st.dataframe(CircuitVisualizer.get_metrics_dataframe(metrics), use_container_width=True, hide_index=True)
    with col_g2:
        st.markdown("#### Gate Frequency by Instruction")
        st.dataframe(CircuitVisualizer.get_gate_distribution_dataframe(metrics), use_container_width=True, hide_index=True)

with tab_explain:
    st.markdown("""
    ### 🧩 Anatomy of the QAOA Circuit
    
    1. **Hadamard Initialization Layer ($H^{\\otimes N}$)**
       * Transforms the ground state $|0\\rangle^{\\otimes N}$ into an equal superposition of all $2^N$ computational basis states:
         $$|+\\rangle^{\\otimes N} = \\frac{1}{\\sqrt{2^N}} \\sum_{z \\in \\{0,1\\}^N} |z\\rangle$$
       * Ensures that all potential routing configurations begin with identical probability amplitudes.
       
    2. **Cost Unitary Layer ($e^{-i \\gamma_l H_C}$)**
       * Encodes problem constraints and distances into quantum phase shifts.
       * Linear terms $h_i Z_i$ are implemented via single-qubit $R_Z(2 \\gamma h_i)$ rotations.
       * Quadratic terms $J_{ij} Z_i Z_j$ are implemented via two-qubit entangling $R_{ZZ}(2 \\gamma J_{ij})$ gates.
       
    3. **Transverse Mixer Unitary ($e^{-i \\beta_l H_M}$)**
       * Applied via single-qubit $R_X(2 \\beta_l)$ rotations on all qubits.
       * Facilitates quantum tunneling and amplitude interference across the state space.
       
    4. **Computational Basis Readout (Measurement)**
       * Projects the evolved quantum state $|\psi(\\vec{\\gamma}, \\vec{\\beta})\\rangle$ into classical bitstrings $z \\in \\{0, 1\\}^N$.
    """)
