"""
Page 8: Automated Batch Experiments and Sensitivity Studies
Executes Controlled Experiments 1–7:
- Experiment 1: Standard QAOA vs. QLDO-QAOA (Head-to-Head under identical parameters)
- Experiment 2 & 3: Problem Size (n) and QAOA Depth (p) Scaling
- Experiment 4, 5, 6: Hyperparameter Sweeps (Alpha, Beta, Gamma, Epsilon)
- Experiment 7: Ideal vs. Noisy NISQ Simulation
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from experiments.benchmark_runner import BenchmarkRunner
from experiments.qldo_benchmark import QLDOBenchmarkRunner
from experiments.parameter_sweep import ParameterSweepRunner
from experiments.noise_experiment import NoiseExperimentRunner
from data_processing.generator import SyntheticDataGenerator
from data_processing.distance_matrix import DistanceMatrixCalculator
from quantum.qubo import QuboBuilder
from quantum.hamiltonian import HamiltonianConverter
from utils.helpers import initialize_session_state

initialize_session_state()

st.title("🧪 Automated Batch Experiments & Research Studies")
st.markdown("Execute controlled empirical experiments comparing **Standard QAOA** and the proposed **QLDO-QAOA** across problem dimensions, ansatz depths, hyperparameter landscapes, and simulated NISQ noise.")

nodes = st.session_state.nodes
cmat = st.session_state.cost_matrix
dmat = st.session_state.distance_matrix
pen_lambda = st.session_state.penalty_lambda

qubo = QuboBuilder.build_qubo(nodes, cmat, penalty_lambda=pen_lambda)
ising = HamiltonianConverter.qubo_to_ising(qubo)

tab_h2h, tab_sweeps, tab_scaling, tab_noise = st.tabs([
    "⚔️ Experiment 1: Standard vs. QLDO Head-to-Head",
    "🎛️ Experiments 4–6: QLDO Hyperparameter Sweeps (α, β, γ, ε)",
    "📈 Experiments 2 & 3: Depth (p) and Problem Size (n) Scaling",
    "📡 Experiment 7: Ideal vs. Noisy NISQ Simulation"
])

# ----------------- TAB 1: HEAD-TO-HEAD COMPARISON -----------------
with tab_h2h:
    st.subheader("Experiment 1: Controlled Standard QAOA vs. QLDO-QAOA Comparison")
    st.markdown("Tests the primary research hypothesis under identical datasets, QUBO mappings, initial angles, shots, and optimizers.")
    
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        h_p = st.selectbox("QAOA Depth p", options=[1, 2], index=0, key="h_p")
        h_shots = st.selectbox("Measurement Shots", options=[512, 1024, 2048], index=1, key="h_shots")
    with col_h2:
        h_opt = st.selectbox("Classical Optimizer", options=["COBYLA", "Nelder-Mead"], index=0, key="h_opt")
        h_iters = st.slider("Max Iterations", min_value=10, max_value=50, value=25, step=5, key="h_iters")
    with col_h3:
        h_alpha = st.number_input("Alpha (Variance)", value=0.05, step=0.01, key="h_alpha")
        h_beta = st.number_input("Beta (Concentration)", value=5.0, step=1.0, key="h_beta")
        h_gamma = st.number_input("Gamma (Barrier)", value=5.0, step=1.0, key="h_gamma")
        
    if st.button("🚀 Run Controlled Head-to-Head Benchmark", type="primary", use_container_width=True):
        prog = st.progress(0, text="Initializing head-to-head benchmark...")
        def prog_cb(curr, tot, msg):
            prog.progress(int((curr / tot) * 100), text=msg)
            
        with st.spinner("Executing controlled head-to-head comparison..."):
            record, std_res, qldo_res = QLDOBenchmarkRunner.run_head_to_head_comparison(
                nodes=nodes,
                cost_matrix=cmat,
                distance_matrix=dmat,
                p=h_p,
                optimizer_type=h_opt,
                max_iterations=h_iters,
                shots=h_shots,
                alpha=h_alpha,
                beta=h_beta,
                gamma=h_gamma,
                progress_callback=prog_cb
            )
            
        st.session_state.h2h_record = record
        st.session_state.h2h_std = std_res
        st.session_state.h2h_qldo = qldo_res
        st.success("Head-to-head benchmark completed successfully!")
        
    if "h2h_record" in st.session_state:
        rec = st.session_state.h2h_record
        st.markdown("#### 📑 Comparative Research Metric Card")
        
        # Comparison Table
        df_comp = pd.DataFrame([
            {"Metric": "Optimal Ground Truth Cost (C*)", "Standard QAOA": f"{rec.ground_truth_cost:.2f}" if rec.ground_truth_cost else "N/A", "QLDO-QAOA": f"{rec.ground_truth_cost:.2f}" if rec.ground_truth_cost else "N/A", "Delta": "0.0"},
            {"Metric": "Best Measured Tour Cost (C_best)", "Standard QAOA": f"{rec.std_best_cost:.2f}", "QLDO-QAOA": f"{rec.qldo_best_cost:.2f}", "Delta": f"{rec.delta_best_cost:+.2f}"},
            {"Metric": "Feasible Permutation Probability (P_F)", "Standard QAOA": f"{rec.std_feasible_prob * 100:.2f}%", "QLDO-QAOA": f"{rec.qldo_feasible_prob * 100:.2f}%", "Delta": f"{rec.delta_feasible_prob * 100:+.2f}%"},
            {"Metric": "Best Route Sampling Probability P(r*)", "Standard QAOA": f"{rec.std_best_prob * 100:.2f}%", "QLDO-QAOA": f"{rec.qldo_best_prob * 100:.2f}%", "Delta": f"{rec.delta_best_prob * 100:+.2f}%"},
            {"Metric": "Expected Logistics Cost E[C]", "Standard QAOA": f"{rec.std_expected_cost:.2f}", "QLDO-QAOA": f"{rec.qldo_expected_cost:.2f}", "Delta": f"{rec.qldo_expected_cost - rec.std_expected_cost:+.2f}"},
            {"Metric": "Approximation Ratio (C* / C_best)", "Standard QAOA": f"{rec.std_approx_ratio:.4f}" if rec.std_approx_ratio else "N/A", "QLDO-QAOA": f"{rec.qldo_approx_ratio:.4f}" if rec.qldo_approx_ratio else "N/A", "Delta": f"{rec.qldo_approx_ratio - rec.std_approx_ratio:+.4f}" if (rec.qldo_approx_ratio and rec.std_approx_ratio) else "N/A"},
            {"Metric": "Execution Runtime (ms)", "Standard QAOA": f"{rec.std_runtime_ms:.1f} ms", "QLDO-QAOA": f"{rec.qldo_runtime_ms:.1f} ms", "Delta": f"{rec.qldo_runtime_ms - rec.std_runtime_ms:+.1f} ms"}
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        
        # Side-by-side Bar Charts
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Standard QAOA",
            x=["Feasible Probability (P_F %)", "Best Route Prob P(r*) %"],
            y=[rec.std_feasible_prob * 100.0, rec.std_best_prob * 100.0],
            marker_color="#FF9100"
        ))
        fig_bar.add_trace(go.Bar(
            name="QLDO-QAOA (Proposed)",
            x=["Feasible Probability (P_F %)", "Best Route Prob P(r*) %"],
            y=[rec.qldo_feasible_prob * 100.0, rec.qldo_best_prob * 100.0],
            marker_color="#00D4B2"
        ))
        fig_bar.update_layout(
            title="Feasibility Probability and Target Route Mass: Standard QAOA vs QLDO-QAOA",
            barmode="group",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1A1D24",
            font=dict(color="#FAFAFA")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- TAB 2: HYPERPARAMETER SWEEPS -----------------
with tab_sweeps:
    st.subheader("Experiments 4–6: QLDO Hyperparameter Sensitivity Analysis")
    st.markdown("Systematically test the individual influence of variance regularization (α), concentration (β), and the logarithmic barrier (γ, ε).")
    
    sweep_type = st.selectbox(
        "Select Hyperparameter Sweep:",
        options=[
            "Experiment 4: Alpha Sweep (Variance Regularization Effect)",
            "Experiment 5: Beta Sweep (Good-Route Concentration Effect)",
            "Experiment 6: Gamma & Epsilon Sweep (Feasibility Barrier Sensitivity)"
        ]
    )
    
    if st.button("🚀 Run Parameter Sweep", type="primary"):
        prog_sw = st.progress(0, text="Executing parameter sweep...")
        def p_cb(c, t, msg):
            prog_sw.progress(int((c / t) * 100), text=msg)
            
        with st.spinner("Sweeping parameter range..."):
            if "Alpha" in sweep_type:
                df_res = ParameterSweepRunner.sweep_alpha(
                    nodes, cmat, dmat, ising, alpha_values=[0.0, 0.01, 0.05, 0.1, 0.5], progress_cb=p_cb
                )
                st.session_state.sweep_alpha_df = df_res
            elif "Beta" in sweep_type:
                df_res = ParameterSweepRunner.sweep_beta(
                    nodes, cmat, dmat, ising, beta_values=[0.0, 1.0, 5.0, 10.0], progress_cb=p_cb
                )
                st.session_state.sweep_beta_df = df_res
            else:
                df_res = ParameterSweepRunner.sweep_gamma_epsilon(
                    nodes, cmat, dmat, ising, gamma_values=[0.0, 2.0, 5.0, 10.0], epsilon_values=[1e-2, 1e-4, 1e-6], progress_cb=p_cb
                )
                st.session_state.sweep_gamma_df = df_res
        st.success("Parameter sweep completed!")
        
    if "Alpha" in sweep_type and "sweep_alpha_df" in st.session_state:
        df_a = st.session_state.sweep_alpha_df
        st.dataframe(df_a, use_container_width=True, hide_index=True)
        fig_a = px.line(df_a, x="Alpha", y="Cost Variance Var(C)", markers=True, title="Cost Variance Suppression vs. Alpha")
        fig_a.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
        st.plotly_chart(fig_a, use_container_width=True)
        
    elif "Beta" in sweep_type and "sweep_beta_df" in st.session_state:
        df_b = st.session_state.sweep_beta_df
        st.dataframe(df_b, use_container_width=True, hide_index=True)
        fig_b = px.line(df_b, x="Beta", y="Good Concentration G", markers=True, title="Probability Concentration G vs. Beta")
        fig_b.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
        st.plotly_chart(fig_b, use_container_width=True)
        
    elif "Gamma" in sweep_type and "sweep_gamma_df" in st.session_state:
        df_g = st.session_state.sweep_gamma_df
        st.dataframe(df_g, use_container_width=True, hide_index=True)

# ----------------- TAB 3: SCALING -----------------
with tab_scaling:
    st.subheader("Experiments 2 & 3: Depth (p) and Problem Size (n) Scaling")
    
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        loc_options = st.multiselect("Select Location Sizes", options=[3, 4, 5], default=[3, 4])
        exp_p = st.selectbox("QAOA Depth p", options=[1, 2], index=0)
    with col_sc2:
        exp_shots = st.selectbox("Shots per instance", options=[512, 1024], index=0)
        
    if st.button("🚀 Run Scaling Study", type="primary"):
        prog_sc = st.progress(0, text="Running scaling benchmarks...")
        def p_sc(curr, tot, msg):
            prog_sc.progress(int((curr / tot) * 100), text=msg)
            
        with st.spinner("Benchmarking instance sizes..."):
            df_scale = BenchmarkRunner.run_scaling_experiment(
                location_counts=loc_options,
                qaoa_p=exp_p,
                shots=exp_shots,
                progress_callback=p_sc
            )
            st.session_state.df_scaling = df_scale
        st.success("Scaling study completed!")
        
    if "df_scaling" in st.session_state:
        df_s = st.session_state.df_scaling
        st.dataframe(df_s, use_container_width=True, hide_index=True)
        
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(x=df_s["num_locations"], y=df_s["classical_2opt_distance"], name="Classical 2-Opt (km)", marker_color="#FF9100"))
        fig_s.add_trace(go.Bar(x=df_s["num_locations"], y=df_s["qaoa_repaired_distance"], name="Quantum Repaired (km)", marker_color="#00D4B2"))
        fig_s.add_trace(go.Bar(x=df_s["num_locations"], y=df_s["exact_optimal_distance"], name="Exact Optimal (km)", marker_color="#2979FF"))
        fig_s.update_layout(title="Distance vs Number of Locations", barmode="group", paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
        st.plotly_chart(fig_s, use_container_width=True)

# ----------------- TAB 4: NOISE EXPERIMENT -----------------
with tab_noise:
    st.subheader("Experiment 7: Ideal vs. Noisy NISQ Simulation")
    st.markdown("Evaluates algorithm robustness in the presence of 1-qubit depolarizing gate noise, 2-qubit entangling noise, and measurement readout errors.")
    
    col_n1, col_n2, col_n3 = st.columns(3)
    with col_n1:
        p1_gate = st.number_input("1-Qubit Gate Error (p1)", value=0.001, format="%.4f", step=0.0005)
    with col_n2:
        p2_gate = st.number_input("2-Qubit Entangling Error (p2)", value=0.010, format="%.4f", step=0.005)
    with col_n3:
        p_ro = st.number_input("Readout Bit-Flip Error", value=0.020, format="%.4f", step=0.005)
        
    if st.button("🚀 Run Noisy Simulation Benchmark", type="primary"):
        with st.spinner("Simulating noisy quantum circuit execution with Aer NoiseModel..."):
            ideal_m, noisy_m = NoiseExperimentRunner.evaluate_circuit_under_noise(
                hamiltonian=ising,
                opt_gamma=[0.2],
                opt_beta=[0.4],
                nodes=nodes,
                distance_matrix=dmat,
                cost_matrix=cmat,
                p1_gate=p1_gate,
                p2_gate=p2_gate,
                p_readout=p_ro,
                shots=1024
            )
            st.session_state.noise_results = (ideal_m, noisy_m)
        st.success("Noisy simulation completed!")
        
    if "noise_results" in st.session_state:
        id_m, ny_m = st.session_state.noise_results
        
        st.markdown("#### 📑 Noise Robustness Summary")
        df_noise = pd.DataFrame([
            {"Environment": "Ideal Simulation (Zero Noise)", "Feasibility P_F": f"{id_m['feasible_prob'] * 100:.1f}%", "Best Distance (km)": f"{id_m['best_dist']:.2f}", "Expected Cost": f"{id_m['expected_cost']:.2f}", "Valid Tours Found": id_m['unique_valid_tours']},
            {"Environment": "Noisy Simulation (NISQ Model)", "Feasibility P_F": f"{ny_m['feasible_prob'] * 100:.1f}%", "Best Distance (km)": f"{ny_m['best_dist']:.2f}", "Expected Cost": f"{ny_m['expected_cost']:.2f}", "Valid Tours Found": ny_m['unique_valid_tours']}
        ])
        st.dataframe(df_noise, use_container_width=True, hide_index=True)
        st.caption(f"Noise Profile: p1={p1_gate}, p2={p2_gate}, readout_error={p_ro}. Executed entirely on classical simulator via Qiskit Aer NoiseModel.")
