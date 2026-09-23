"""
Q-LogiOpt: Quantum Approximate Optimization for Intelligent Logistics Route and Delivery Optimization
Main Streamlit Application Entrypoint
"""
import streamlit as st
import pandas as pd
from config.settings import CONFIG
from utils.helpers import initialize_session_state, run_all_classical, run_baseline_quantum
from visualization.maps import RouteMapVisualizer
from visualization.charts import ChartVisualizer

# Set page config
st.set_page_config(
    page_title="Q-LogiOpt | Quantum Logistics Optimization",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism cards, and clean typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header hero banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(121, 40, 202, 0.25) 0%, rgba(0, 212, 178, 0.15) 100%);
        border: 1px solid rgba(0, 212, 178, 0.3);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .hero-title {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        color: #A0AEC0;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    .badge-quantum {
        background: rgba(0, 212, 178, 0.2);
        color: #00D4B2;
        border: 1px solid #00D4B2;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .badge-classical {
        background: rgba(255, 145, 0, 0.2);
        color: #FF9100;
        border: 1px solid #FF9100;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Metric Glass Cards */
    .metric-container {
        background: #1A1D24;
        border: 1px solid #2E3440;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #00D4B2;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FAFAFA;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-lbl {
        color: #8F9CAE;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data and state
initialize_session_state()

# Sidebar Navigation Info
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
        <span style="font-size: 2.2rem; filter: drop-shadow(0 0 10px rgba(0, 212, 178, 0.7));">⚛️</span>
        <div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px;">Q-LogiOpt</div>
            <div style="font-size: 0.72rem; color: #00D4B2; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Quantum Optimization</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Intelligent Logistics Route Optimization")
    st.markdown("---")
    
    st.markdown("### 📌 Active Problem")
    st.write(f"**Total Locations:** {len(st.session_state.nodes)}")
    st.write(f"**Customer Stops:** {len(st.session_state.nodes) - 1}")
    st.write(f"**QUBO Qubits:** {(len(st.session_state.nodes) - 1)**2}")
    st.write(f"**Distance Metric:** {st.session_state.distance_metric.capitalize()}")
    
    st.markdown("---")
    if st.button("⚡ Run Classical Baselines", use_container_width=True):
        with st.spinner("Running classical algorithms..."):
            run_all_classical()
        st.success("Classical baselines evaluated!")
        st.rerun()

    if st.button("⚛️ Run Quantum Optimization", use_container_width=True, type="primary"):
        with st.spinner("Executing quantum variational simulation..."):
            run_baseline_quantum()
        st.success("Quantum optimization complete!")
        st.rerun()

# Main Hero Banner
st.markdown("""
<div class="hero-card">
    <div class="hero-title">
        <span>⚛️ Q-LogiOpt</span>
        <span class="badge-quantum">QISKIT 2.x SIMULATION</span>
        <span class="badge-classical">CLASSICAL BENCHMARK</span>
    </div>
    <div class="hero-subtitle">
        Quantum Approximate Optimization Algorithm (QAOA) vs. Classical Heuristics for Constrained Last-Mile Logistics Routing.
        Transforming real-world logistics networks into Quadratic Unconstrained Binary Optimization (QUBO) matrices solved via variational quantum simulation.
    </div>
</div>
""", unsafe_allow_html=True)

# Ensure classical and baseline quantum solutions are present
if not st.session_state.solutions:
    run_all_classical()
    if len(st.session_state.nodes) <= 5:
        run_baseline_quantum()

nodes = st.session_state.nodes
solutions = st.session_state.solutions
qaoa_res = st.session_state.qaoa_result
aq_res = st.session_state.get("aq_res")
qaoa_summary = st.session_state.quantum_summary

# Top KPI Metric Cards
kpi_cols = st.columns(6)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val">{len(nodes)}</div>
        <div class="metric-lbl">Total Stops</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    num_qubits = (len(nodes) - 1) ** 2
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val">{num_qubits}</div>
        <div class="metric-lbl">QUBO Qubits</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    classical_sols = [s for k, s in solutions.items() if "QAOA" not in k]
    best_classical = min([s.total_distance for s in classical_sols]) if classical_sols else 0.0
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val" style="color: #FF9100;">{best_classical:.1f} km</div>
        <div class="metric-lbl">Classical Best</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    if "QAOA (Strict)" in solutions:
        q_dist = f"{solutions['QAOA (Strict)'].total_distance:.1f} km"
        q_color = "#00E676"
    elif "QAOA (Repaired)" in solutions:
        q_dist = f"{solutions['QAOA (Repaired)'].total_distance:.1f} km*"
        q_color = "#B388FF"
    else:
        q_dist = "Not Run"
        q_color = "#8F9CAE"
        
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val" style="color: {q_color};">{q_dist}</div>
        <div class="metric-lbl">QAOA Best</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    if aq_res:
        csr_text = f"{aq_res.final_csr_pct:.1f}%"
    elif qaoa_summary:
        csr_text = f"{qaoa_summary['constraint_satisfaction_rate']:.1f}%"
    else:
        csr_text = "N/A"
        
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val" style="color: #00D4B2;">{csr_text}</div>
        <div class="metric-lbl">Quantum CSR</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[5]:
    if aq_res:
        q_time = f"{aq_res.total_runtime_ms / 1000.0:.2f} s"
    elif qaoa_res:
        q_time = f"{qaoa_res.total_runtime_ms / 1000.0:.2f} s"
    elif "QAOA (Strict)" in solutions:
        q_time = f"{solutions['QAOA (Strict)'].runtime_ms / 1000.0:.2f} s"
    else:
        q_time = "N/A"
        
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-val">{q_time}</div>
        <div class="metric-lbl">QAOA Runtime</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Telemetry Banner & Rerun Trigger
sim_col1, sim_col2 = st.columns([8, 2])
with sim_col1:
    if aq_res:
        valid_cnt = len(aq_res.all_valid_states)
        st.info(f"🌟 **Active Quantum Telemetry (AQ-LogiQAOA):** Identified **{valid_cnt} strictly feasible permutation tours** directly from quantum sampling (CSR: **{aq_res.final_csr_pct:.1f}%** across {aq_res.total_rounds} rounds).")
    elif "QAOA (Strict)" in solutions:
        st.info(f"🌟 **Active Quantum Telemetry:** QAOA simulation complete. Optimal tour distance: **{solutions['QAOA (Strict)'].total_distance:.2f} km**.")
    else:
        st.info("💡 Click **Run Quantum Optimization** to simulate quantum variational routing for this network.")

with sim_col2:
    if st.button("🔄 Re-Simulate QAOA", use_container_width=True, type="secondary"):
        with st.spinner("Simulating quantum optimization..."):
            run_baseline_quantum()
        st.rerun()

# Main Section: Interactive Route Map & Filter
col_map, col_details = st.columns([7, 3])

with col_map:
    st.subheader("🗺️ Logistics Network & Delivery Routes")
    available_routes = list(solutions.keys())
    
    # Intelligently pre-select Exact, QAOA, and 2-Opt by default
    preferred = [r for r in ["Exact (Brute Force)", "QAOA (Strict)", "2-Opt", "Nearest Neighbor"] if r in available_routes]
    default_routes = preferred[:3] if preferred else available_routes[:2]
    
    selected_routes = st.multiselect(
        "Select routes to overlay on map:",
        options=available_routes,
        default=default_routes,
        placeholder="Choose algorithms to display on map..."
    )
    
    fig_map = RouteMapVisualizer.plot_routes(
        nodes=nodes,
        solutions=solutions,
        active_routes=selected_routes,
        title="Interactive Fleet Route Visualizer (OpenStreetMap)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_details:
    st.subheader("📋 Route Solutions")
    if solutions:
        records = []
        for name, sol in solutions.items():
            is_quantum = "QAOA" in name
            records.append({
                "Type": "⚛️ Quantum" if is_quantum else "💻 Classical",
                "Algorithm": name,
                "Distance (km)": f"{sol.total_distance:.2f}",
                "Cost": f"{sol.total_cost:.2f}",
                "Feasible": "✅ Yes" if sol.is_feasible else f"❌ ({sol.constraint_violations})",
                "Runtime": f"{sol.runtime_ms:.1f} ms" if sol.runtime_ms < 1000 else f"{sol.runtime_ms/1000.0:.2f} s"
            })
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        
        st.markdown("#### 🎯 Quick Tour Paths")
        for name, sol in solutions.items():
            path_str = " ➔ ".join(sol.route_ids)
            if "QAOA" in name:
                st.markdown(f"**⚛️ {name}:** `{path_str}`")
            else:
                st.markdown(f"**{name}:** `{path_str}`")
    else:
        st.info("Run classical or QAOA solvers to inspect routes.")

# Quick Action Panels
st.markdown("---")
st.subheader("🚀 Quick Navigation & Modules")

nav_cols = st.columns(4)

with nav_cols[0]:
    st.markdown("### 📍 Dataset & Network")
    st.caption("Upload custom delivery stops or generate synthetic clusters with customer demands and priorities.")
    if st.button("Go to Dataset Manager ➔", use_container_width=True):
        st.switch_page("pages/2_📍_Dataset.py")

with nav_cols[1]:
    st.markdown("### 🧮 QUBO Matrix")
    st.caption("Inspect the anchored depot QUBO penalty matrix, variable mappings, and Ising Hamiltonian terms.")
    if st.button("Inspect QUBO ➔", use_container_width=True):
        st.switch_page("pages/5_🧮_QUBO_Matrix.py")

with nav_cols[2]:
    st.markdown("### ⚛️ QAOA Optimization")
    st.caption("Configure circuit depth p, select classical optimizer, and simulate quantum variational optimization.")
    if st.button("Run QAOA ➔", use_container_width=True):
        st.switch_page("pages/6_⚛️_QAOA_Optimization.py")

with nav_cols[3]:
    st.markdown("### 📈 Research Evaluation")
    st.caption("Analyze approximation ratios, convergence curves, constraint satisfaction, and export benchmarks.")
    if st.button("View Performance ➔", use_container_width=True):
        st.switch_page("pages/9_📈_Performance.py")
