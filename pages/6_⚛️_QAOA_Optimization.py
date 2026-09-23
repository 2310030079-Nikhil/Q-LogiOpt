"""
Page 6: Quantum Route Optimization Engine
Supports:
1. QLDO-QAOA (Quantum Logistics Distribution Objective — Proposed Novel Formulation)
2. Standard QAOA (Conventional Expectation E[C] Minimization)
3. AQ-LogiQAOA (Adaptive Constraint-Aware QAOA with Feedback)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantum import (
    QuboBuilder,
    HamiltonianConverter,
    QAOAOptimizer,
    AQLogiQAOAOptimizer,
    QLDOQAOAOptimizer,
    QLDOExecutionResult
)
from qldo import (
    ProbabilityDistribution,
    FeasibilityEngine,
    ConcentrationEngine,
    GoodRouteDefinition,
    QLDOObjectiveEngine,
    MetricsEvaluator
)
from optimization import RouteDecoder
from visualization.charts import ChartVisualizer
from utils.helpers import initialize_session_state

initialize_session_state()

st.title("⚛️ Quantum Route Optimization Engine")
st.markdown("Execute either the proposed **QLDO-QAOA (Quantum Logistics Distribution Objective)**, **Standard QAOA**, or **AQ-LogiQAOA** with closed-loop feedback.")

nodes = st.session_state.nodes
cmat = st.session_state.cost_matrix
dmat = st.session_state.distance_matrix
pen_lambda = st.session_state.penalty_lambda

# 1. Algorithm Selection Banner
algo_choice = st.radio(
    "Select Quantum Optimization Architecture:",
    options=[
        "🎯 QLDO-QAOA (Quantum Logistics Distribution Objective — Proposed Novel Formulation)",
        "⚙️ Standard QAOA (Conventional Expectation E[C] Minimization)",
        "🚀 AQ-LogiQAOA (Adaptive Constraint Penalty Feedback Loop)"
    ],
    index=0,
    help="QLDO-QAOA optimizes an empirical distribution-aware loss balancing expected cost, cost variance, feasibility barrier, and good-route concentration."
)
is_qldo = "QLDO" in algo_choice
is_adaptive = "AQ-LogiQAOA" in algo_choice
is_standard = "Standard QAOA" in algo_choice

# 2. Parameter Controls
st.markdown("### 🎛️ Simulation Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    qaoa_p = st.selectbox(
        "QAOA Depth (p)",
        options=[1, 2, 3],
        index=0,
        help="Number of alternating Cost and Mixer layers. Higher p expands ansatz expressibility."
    )

with col2:
    shots = st.selectbox(
        "Measurement Shots",
        options=[512, 1024, 2048, 4096],
        index=1,
        help="Number of quantum projective measurement samples in computational basis."
    )

with col3:
    if is_adaptive:
        num_rounds = st.slider(
            "Adaptive Rounds (T)",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            help="Number of closed-loop outer adaptation iterations."
        )
    else:
        optimizer_type = st.selectbox(
            "Classical Optimizer",
            options=["COBYLA", "Nelder-Mead", "BFGS"],
            index=0
        )

with col4:
    if is_adaptive:
        adaptation_eta = st.slider(
            "Adaptation Rate (η)",
            min_value=0.1,
            max_value=1.5,
            value=0.5,
            step=0.1,
            help="Penalty amplification sensitivity when feasibility is low."
        )
    else:
        max_iters = st.slider("Max Iterations", min_value=10, max_value=80, value=30, step=5)

# Dedicated QLDO Hyperparameters
if is_qldo:
    with st.expander("⚙️ QLDO Hyperparameters (Distribution-Aware Loss Controls)", expanded=True):
        st.markdown(r"Configure weights for: $J_{\text{QLDO}} = E[C] + \alpha \operatorname{Var}(C) - \beta G_{\theta} - \gamma \log(P_F + \epsilon)$")
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        
        with q_c1:
            qldo_alpha = st.slider(
                "Variance Weight (α)",
                min_value=0.0,
                max_value=0.5,
                value=0.05,
                step=0.01,
                help="Penalizes multimodal spread and output variance across candidate routes."
            )
            
        with q_c2:
            qldo_beta = st.slider(
                "Concentration Weight (β)",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Rewards quadratic probability concentration G = sum_{G} p(r)^2 over high-quality feasible routes."
            )
            
        with q_c3:
            qldo_gamma = st.slider(
                "Feasibility Barrier (γ)",
                min_value=0.0,
                max_value=15.0,
                value=5.0,
                step=0.5,
                help="Logarithmic barrier penalty weight: -gamma * log(P_F + epsilon)."
            )
            
        with q_c4:
            qldo_eps = st.selectbox(
                "Stability Epsilon (ε)",
                options=[1e-2, 1e-4, 1e-6, 1e-8],
                index=1,
                format_func=lambda x: f"{x:.0e}",
                help="Numerical barrier stabilization constant to prevent log(0)."
            )
            
        def_c1, def_c2 = st.columns([6, 4])
        with def_c1:
            good_def_choice = st.selectbox(
                "Good-Route Set Definition (𝒢)",
                options=[
                    GoodRouteDefinition.RELATIVE_BEST.value,
                    GoodRouteDefinition.PERCENTILE.value,
                    GoodRouteDefinition.CLASSICAL_BENCHMARK.value
                ],
                index=0,
                help="Criteria defining high-utility feasible routes G subseteq F."
            )
        with def_c2:
            if "Relative to Best" in good_def_choice:
                good_delta = st.slider("Tolerance δ", min_value=0.05, max_value=0.40, value=0.15, step=0.05, help="Within (1 + delta) * C_best")
                good_percentile = 0.25
            else:
                good_percentile = st.slider("Quantile Cutoff q", min_value=0.10, max_value=0.50, value=0.25, step=0.05)
                good_delta = 0.15

# Hardware scaling note
if len(nodes) >= 6:
    st.warning(f"⚠️ Problem size is {len(nodes)} locations (requires {(len(nodes)-1)**2} qubits). Quantum statevector simulation scales exponentially.")

# Action Button
if is_qldo:
    btn_label = "🎯 RUN QLDO-QAOA OPTIMIZATION"
elif is_adaptive:
    btn_label = "🚀 RUN AQ-LogiQAOA OPTIMIZATION"
else:
    btn_label = "⚙️ RUN STANDARD QAOA OPTIMIZATION"

if st.button(btn_label, type="primary", use_container_width=True):
    # Ensure QUBO & Ising Hamiltonian are built
    qubo = QuboBuilder.build_qubo(nodes, cmat, penalty_lambda=pen_lambda)
    ising = HamiltonianConverter.qubo_to_ising(qubo)
    st.session_state.qubo = qubo

    if is_qldo:
        # Run QLDO-QAOA
        prog_bar = st.progress(0, text="Initializing QLDO-QAOA distribution-aware optimization loop...")
        def progress_cb(curr, tot, msg):
            pct = min(100, int((curr / max(1, tot)) * 100))
            prog_bar.progress(pct, text=msg)

        # Ground truth if small
        c_star = min([s.total_distance for s in st.session_state.solutions.values() if s.is_feasible]) if st.session_state.solutions else None

        with st.spinner("Executing QLDO-QAOA optimization with multi-moment loss..."):
            qldo_res = QLDOQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cmat,
                distance_matrix=dmat,
                hamiltonian=ising,
                p=qaoa_p,
                optimizer_type=optimizer_type,
                max_iterations=max_iters,
                shots=shots,
                alpha=qldo_alpha,
                beta=qldo_beta,
                gamma=qldo_gamma,
                epsilon=qldo_eps,
                good_route_def=GoodRouteDefinition(good_def_choice),
                percentile_q=good_percentile,
                delta=good_delta,
                vehicle_capacity=st.session_state.vehicle_capacity,
                max_distance=st.session_state.max_distance,
                seed=42,
                optimal_cost=c_star,
                progress_callback=progress_cb
            )

            st.session_state.qldo_res = qldo_res
            st.session_state.aq_res = None
            st.session_state.qaoa_result = None
            st.session_state.active_quantum_mode = "QLDO-QAOA"

            if qldo_res.best_feasible_solution:
                st.session_state.solutions["QAOA (Strict)"] = qldo_res.best_feasible_solution
            st.session_state.solutions["QAOA (Repaired)"] = qldo_res.best_repaired_solution

            prog_bar.progress(100, text="QLDO-QAOA Simulation Complete!")
            st.toast(f"QLDO-QAOA Complete! P_F: {qldo_res.final_metrics.feasible_probability * 100:.1f}%, G: {qldo_res.final_metrics.good_route_concentration:.4f}", icon="🎯")
            st.rerun()

    elif is_adaptive:
        # Run AQ-LogiQAOA
        prog_bar = st.progress(0, text="Initializing AQ-LogiQAOA adaptive loop...")
        def progress_cb_aq(curr, tot, msg):
            prog_bar.progress(int((curr / tot) * 100), text=msg)

        with st.spinner("Executing AQ-LogiQAOA closed-loop optimization..."):
            aq_res = AQLogiQAOAOptimizer.optimize(
                nodes=nodes,
                cost_matrix=cmat,
                distance_matrix=dmat,
                p=qaoa_p,
                initial_lambda=pen_lambda,
                adaptation_rate_eta=adaptation_eta,
                num_rounds=num_rounds,
                shots=shots,
                progress_callback=progress_cb_aq
            )
            st.session_state.aq_res = aq_res
            st.session_state.qldo_res = None
            st.session_state.qaoa_result = None
            st.session_state.active_quantum_mode = "AQ-LogiQAOA"

            if aq_res.best_feasible_solution:
                st.session_state.solutions["QAOA (Strict)"] = aq_res.best_feasible_solution
            st.session_state.solutions["QAOA (Repaired)"] = aq_res.best_repaired_solution

            prog_bar.progress(100, text="AQ-LogiQAOA Simulation Complete!")
            st.toast(f"AQ-LogiQAOA complete! Found {len(aq_res.all_valid_states)} valid routes.", icon="🚀")
            st.rerun()

    else:
        # Run Standard QAOA
        prog_bar = st.progress(0, text="Optimizing standard QAOA variational parameters...")
        with st.spinner(f"Optimizing standard QAOA (p={qaoa_p}) with {optimizer_type}..."):
            qaoa_res = QAOAOptimizer.optimize(
                hamiltonian=ising,
                p=qaoa_p,
                optimizer_type=optimizer_type,
                max_iterations=max_iters,
                shots=shots
            )
            prog_bar.progress(80, text="Decoding quantum measurement bitstrings...")

            decoded_states, summary = RouteDecoder.evaluate_measurement_counts(
                counts=qaoa_res.simulation_result.counts,
                nodes=nodes,
                distance_matrix=dmat,
                cost_matrix=cmat,
                vehicle_capacity=st.session_state.vehicle_capacity,
                max_distance=st.session_state.max_distance
            )

            st.session_state.qaoa_result = qaoa_res
            st.session_state.decoded_states = decoded_states
            st.session_state.quantum_summary = summary
            st.session_state.qldo_res = None
            st.session_state.aq_res = None
            st.session_state.active_quantum_mode = "Standard QAOA"

            valid_states = [s for s in decoded_states if s.is_valid_permutation]
            if valid_states:
                st.session_state.solutions["QAOA (Strict)"] = valid_states[0].strict_solution
                st.session_state.solutions["QAOA (Strict)"].runtime_ms = round(qaoa_res.total_runtime_ms, 2)
            st.session_state.solutions["QAOA (Repaired)"] = decoded_states[0].repaired_solution
            st.session_state.solutions["QAOA (Repaired)"].runtime_ms = round(qaoa_res.total_runtime_ms, 2)

            prog_bar.progress(100, text="Standard QAOA Complete!")
            st.toast(f"Standard QAOA complete! CSR: {summary['constraint_satisfaction_rate']}%.", icon="⚛️")
            st.rerun()

st.markdown("---")

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
qldo_res = st.session_state.get("qldo_res")
aq_res = st.session_state.get("aq_res")
std_res = st.session_state.get("qaoa_result")

# ----------------- CASE 1: QLDO-QAOA RESULTS -----------------
if qldo_res is not None:
    st.subheader("🎯 QLDO-QAOA Execution Telemetry & Distribution Analysis")
    
    # KPI Metrics Row
    m = qldo_res.final_metrics
    c = qldo_res.final_components
    
    k_cols = st.columns(6)
    k_cols[0].metric("Final J_QLDO", f"{qldo_res.final_qldo:.2f}", f"Initial: {qldo_res.initial_qldo:.2f}")
    k_cols[1].metric("Feasible Prob (P_F)", f"{m.feasible_probability * 100:.1f}%", f"{len(qldo_res.all_valid_states)} Valid Permutations")
    k_cols[2].metric("Best Route Cost", f"{m.best_route_cost:.2f} km", "Strict Feasible" if qldo_res.best_feasible_solution else "Repaired")
    k_cols[3].metric("Best Route Prob P(r*)", f"{m.best_route_probability * 100:.2f}%", f"G: {m.good_route_concentration:.5f}")
    k_cols[4].metric("Expected Cost E[C]", f"{m.expected_cost:.2f}", f"Var: {m.cost_variance:.1f}")
    k_cols[5].metric("Runtime", f"{m.total_runtime_ms / 1000.0:.2f} s", f"{m.total_iterations} Iterations")

    st.markdown("<br>", unsafe_allow_html=True)

    # SECTION 31: DEDICATED QLDO COMPONENT BREAKDOWN PANEL
    st.markdown("### 🧩 QLDO Objective Component Breakdown Panel")
    st.markdown(r"Numerical contributions of every component to the master loss: $J_{\text{QLDO}} = E[C] + \alpha \operatorname{Var}(C) - \beta G - \gamma \log(P_F + \epsilon)$")
    
    comp_col1, comp_col2 = st.columns([5, 5])
    with comp_col1:
        st.markdown(f"""
        <div style="background: #1A1D24; border: 1px solid #2E3440; border-radius: 10px; padding: 18px; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Expected Route Cost E[C]:</span>
                <span style="color: #FAFAFA; font-weight: 700;">+{c.expected_cost:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Variance Regularization (+α·Var):</span>
                <span style="color: #FF9100; font-weight: 700;">+{c.scaled_variance:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Concentration Reward (-β·G):</span>
                <span style="color: #00D4B2; font-weight: 700;">{c.scaled_concentration:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                <span>Feasibility Barrier (-γ·log(P_F+ε)):</span>
                <span style="color: #B388FF; font-weight: 700;">+{c.feasibility_barrier:.2f}</span>
            </div>
            <hr style="border-color: #2E3440; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 1.15rem;">
                <span style="color: #00E676; font-weight: 700;">Total QLDO Objective J_QLDO:</span>
                <span style="color: #00E676; font-weight: 700;">={c.total_qldo_objective:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with comp_col2:
        # Plotly Waterfall / Stacked Bar of Components
        waterfall_fig = go.Figure(go.Waterfall(
            name="QLDO Loss Decomposition",
            orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Expected Cost", "+α·Var(C)", "-β·G_θ", "Barrier", "J_QLDO"],
            textposition="outside",
            text=[f"{c.expected_cost:.1f}", f"+{c.scaled_variance:.1f}", f"{c.scaled_concentration:.1f}", f"+{c.feasibility_barrier:.1f}", f"{c.total_qldo_objective:.1f}"],
            y=[c.expected_cost, c.scaled_variance, c.scaled_concentration, c.feasibility_barrier, c.total_qldo_objective],
            connector={"line": {"color": "#8F9CAE"}},
            decreasing={"marker": {"color": "#00D4B2"}},
            increasing={"marker": {"color": "#FF9100"}},
            totals={"marker": {"color": "#00E676"}}
        ))
        waterfall_fig.update_layout(
            title="Component Contribution Waterfall",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#1A1D24",
            font=dict(color="#FAFAFA"),
            margin=dict(l=10, r=10, t=35, b=10)
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    st.markdown("---")

    # SECTION 32: CONVERGENCE GRAPHS
    st.markdown("### 📈 Multi-Trajectory Variational Convergence")
    df_hist = qldo_res.iteration_history
    
    if len(df_hist) > 1:
        conv_tabs = st.tabs([
            "🎯 J_QLDO Convergence",
            "📊 Expected Cost E[C] vs Iteration",
            "🛡️ Feasible Probability P_F vs Iteration",
            "⚡ Good-Route Concentration G vs Iteration"
        ])
        
        with conv_tabs[0]:
            fig_j = px.line(
                df_hist, x="Iteration", y="J_QLDO",
                title="Master QLDO Objective J_QLDO vs Optimization Iteration",
                markers=True, color_discrete_sequence=["#00E676"]
            )
            fig_j.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_j, use_container_width=True)
            
        with conv_tabs[1]:
            fig_e = px.line(
                df_hist, x="Iteration", y="Expected Cost E[C]",
                title="Expected Route Cost E[C] Trajectory",
                markers=True, color_discrete_sequence=["#FF9100"]
            )
            fig_e.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_e, use_container_width=True)
            
        with conv_tabs[2]:
            fig_pf = px.line(
                df_hist, x="Iteration", y="Barrier (-gamma*log)",
                title="Logarithmic Barrier (-γ·log(P_F + ε)) Trajectory (Penalty Wall)",
                markers=True, color_discrete_sequence=["#B388FF"]
            )
            fig_pf.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_pf, use_container_width=True)
            
        with conv_tabs[3]:
            fig_g = px.line(
                df_hist, x="Iteration", y="Good Concentration G",
                title="Good-Route Concentration G_θ Trajectory (Herfindahl-Hirschman Mass)",
                markers=True, color_discrete_sequence=["#00D4B2"]
            )
            fig_g.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
            st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")

    # SECTION 30: QUANTUM MEASUREMENT PROBABILITY DISTRIBUTION BAR CHART
    st.markdown("### 📊 Quantum Measurement Probability Distribution")
    st.markdown("Computational basis projective measurement frequencies with semantic classification:")
    
    counts = qldo_res.simulation_result.counts
    total_shots = qldo_res.simulation_result.shots
    valid_bitstrings = {s.bitstring for s in qldo_res.all_valid_states}
    best_bitstring = qldo_res.final_metrics.best_bitstring
    
    # Sort top 20 bitstrings by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:25]
    
    dist_data = []
    for bitstr, cnt in sorted_counts:
        prob = (cnt / total_shots) * 100.0
        if bitstr == best_bitstring:
            cat = "🌟 Best Route (r*)"
            color = "#FFD700"
        elif bitstr in valid_bitstrings:
            cat = "🟢 Feasible Permutation (F)"
            color = "#00E676"
        else:
            cat = "🔴 Infeasible Bitstring"
            color = "#FF3D71"
            
        dist_data.append({
            "Bitstring": bitstr,
            "Count": cnt,
            "Probability (%)": prob,
            "Classification": cat,
            "Color": color
        })
        
    df_dist = pd.DataFrame(dist_data)
    fig_prob = px.bar(
        df_dist,
        x="Bitstring",
        y="Probability (%)",
        color="Classification",
        color_discrete_map={
            "🌟 Best Route (r*)": "#FFD700",
            "🟢 Feasible Permutation (F)": "#00E676",
            "🔴 Infeasible Bitstring": "#FF3D71"
        },
        title="Top Measurement Probabilities Categorized by Feasibility & Quality"
    )
    fig_prob.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#1A1D24", font=dict(color="#FAFAFA"))
    st.plotly_chart(fig_prob, use_container_width=True)

    # Feasible Solutions Table
    st.markdown("---")
    st.subheader("🌟 Sampled Strictly Feasible Permutation Tours")
    if qldo_res.all_valid_states:
        feas_records = []
        for rank, s in enumerate(qldo_res.all_valid_states, start=1):
            s_count = counts.get(s.bitstring, 0)
            prob_pct = (s_count / total_shots) * 100.0
            feas_records.append({
                "Rank": rank,
                "Bitstring": s.bitstring,
                "Route Sequence": " ➔ ".join(s.strict_solution.route_ids),
                "Tour Distance (km)": f"{s.strict_solution.total_distance:.2f}",
                "Logistics Cost": f"{s.strict_solution.total_cost:.2f}",
                "Measurement Count": s_count,
                "Probability": f"{prob_pct:.2f}%",
                "Feasibility": "✅ STRICTLY VALID"
            })
        st.dataframe(pd.DataFrame(feas_records), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No strictly valid permutations were measured. Showing Hungarian repaired tour:")
        st.write(f"**Repaired Route:** `{' ➔ '.join(qldo_res.best_repaired_solution.route_ids)}` ({qldo_res.best_repaired_solution.total_distance:.2f} km)")

# ----------------- CASE 2: AQ-LogiQAOA RESULTS -----------------
elif aq_res is not None:
    st.subheader("🚀 AQ-LogiQAOA Adaptive Execution Telemetry")
    k_cols = st.columns(5)
    k_cols[0].metric("Final Feasibility (CSR)", f"{aq_res.final_csr_pct:.2f}%", f"{len(aq_res.all_valid_states)} Unique Tours")
    k_cols[1].metric("Penalty Adaptation", f"λ: {aq_res.initial_lambda:.0f} ➔ {aq_res.final_lambda:.1f}", f"+{aq_res.final_lambda - aq_res.initial_lambda:.1f}")
    k_cols[2].metric("Total Rounds", aq_res.total_rounds, f"p={aq_res.p} Depth")
    if aq_res.best_feasible_solution:
        k_cols[3].metric("Best Feasible Distance", f"{aq_res.best_feasible_solution.total_distance:.2f} km", "Strict Optimum")
    else:
        k_cols[3].metric("Best Repaired Distance", f"{aq_res.best_repaired_solution.total_distance:.2f} km", "Repaired")
    k_cols[4].metric("Total Runtime", f"{aq_res.total_runtime_ms / 1000.0:.2f} s")

    st.markdown("#### 📑 Closed-Loop Adaptation Progression")
    st.dataframe(aq_res.rounds_history, use_container_width=True, hide_index=True)

# ----------------- CASE 3: STANDARD QAOA RESULTS -----------------
elif std_res is not None:
    st.subheader("⚙️ Standard QAOA Results (Conventional Expectation)")
    summary = st.session_state.quantum_summary
    r_cols = st.columns(5)
    r_cols[0].metric("Final Ground Energy", f"{std_res.final_energy:.2f}", f"Initial: {std_res.initial_energy:.2f}")
    r_cols[1].metric("Constraint Satisfaction (CSR)", f"{summary['constraint_satisfaction_rate']:.2f}%", f"{summary['valid_shots']}/{summary['total_shots']} Shots")
    r_cols[2].metric("Total Iterations", std_res.iteration_count, f"Optimizer: {std_res.optimizer_name}")
    r_cols[3].metric("Circuit Depth", std_res.circuit_metrics.circuit_depth, f"{std_res.circuit_metrics.total_gates} Gates")
    r_cols[4].metric("Runtime", f"{std_res.total_runtime_ms/1000.0:.2f} s")

    # Plotly energy convergence
    fig_conv = ChartVisualizer.plot_convergence(std_res.energy_history, title="Standard QAOA: Ground Energy <H_C> Convergence")
    st.plotly_chart(fig_conv, use_container_width=True)

    # Measurement distribution and feasible route solutions
    counts = std_res.simulation_result.counts
    total_shots = std_res.simulation_result.shots
    decoded = st.session_state.decoded_states or []
    valid_states = [s for s in decoded if s.is_valid_permutation]

    st.markdown("---")
    st.subheader("📊 Top Quantum Measurement Bitstrings")
    valid_bs_set = {s.bitstring for s in valid_states}
    fig_dist = ChartVisualizer.plot_measurement_distribution(counts, top_k=15, valid_bitstrings=valid_bs_set)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.subheader("🌟 Sampled Strictly Feasible Permutation Tours")
    if valid_states:
        feas_records = []
        for rank, s in enumerate(valid_states, start=1):
            s_count = counts.get(s.bitstring, 0)
            prob_pct = (s_count / total_shots) * 100.0 if total_shots > 0 else 0.0
            feas_records.append({
                "Rank": rank,
                "Bitstring": s.bitstring,
                "Route Sequence": " ➔ ".join(s.strict_solution.route_ids),
                "Tour Distance (km)": f"{s.strict_solution.total_distance:.2f}",
                "Logistics Cost": f"{s.strict_solution.total_cost:.2f}",
                "Measurement Count": s_count,
                "Probability": f"{prob_pct:.2f}%",
                "Feasibility": "✅ STRICTLY VALID"
            })
        st.dataframe(pd.DataFrame(feas_records), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No strictly valid permutations were measured. Showing Hungarian repaired tour:")
        if decoded:
            st.write(f"**Repaired Route:** `{' ➔ '.join(decoded[0].repaired_solution.route_ids)}` ({decoded[0].repaired_solution.total_distance:.2f} km)")

else:
    st.info("💡 Select an architecture above and click **RUN OPTIMIZATION** to simulate quantum variational routing.")
