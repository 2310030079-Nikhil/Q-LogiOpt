"""
Page 1: Executive Dashboard & Route Network Overview
"""
import streamlit as st
import pandas as pd
from utils.helpers import initialize_session_state, run_all_classical
from visualization.maps import RouteMapVisualizer
from visualization.charts import ChartVisualizer

initialize_session_state()

st.title("📊 Logistics Operations Dashboard")
st.markdown("High-level overview of delivery demand, fleet configuration, and optimization performance.")

nodes = st.session_state.nodes
solutions = st.session_state.solutions
dmat = st.session_state.distance_matrix
cmat = st.session_state.cost_matrix

# Top KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Stops", len(nodes), f"{len(nodes)-1} Customers")
c2.metric("Qubits Required", (len(nodes)-1)**2, "Anchored Depot")
classical_sols = [s for k, s in solutions.items() if "QAOA" not in k]
best_class = min([s.total_distance for s in classical_sols]) if classical_sols else 0.0
c3.metric("Best Classical Tour", f"{best_class:.2f} km")

if "QAOA (Strict)" in solutions:
    c4.metric("QAOA Tour (Strict)", f"{solutions['QAOA (Strict)'].total_distance:.2f} km", "Optimal")
elif "QAOA (Repaired)" in solutions:
    c4.metric("QAOA Tour (Repaired)", f"{solutions['QAOA (Repaired)'].total_distance:.2f} km", "Hybrid Repair")
else:
    c4.metric("QAOA Status", "Not Simulated", "Go to QAOA page")

st.markdown("---")

col_left, col_right = st.columns([6, 4])

with col_left:
    st.subheader("🗺️ Geographic Delivery Tour Map")
    if solutions:
        preferred = [r for r in ["Exact (Brute Force)", "QAOA (Strict)", "2-Opt", "Nearest Neighbor"] if r in solutions]
        default_routes = preferred[:3] if preferred else list(solutions.keys())[:2]
        active = st.multiselect(
            "Toggle routes:",
            options=list(solutions.keys()),
            default=default_routes,
            placeholder="Choose algorithms to display..."
        )
        fig_map = RouteMapVisualizer.plot_routes(nodes, solutions, active_routes=active)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No routes computed yet.")

with col_right:
    st.subheader("📊 Comparative Algorithm Benchmark")
    if solutions:
        fig_comp = ChartVisualizer.plot_algorithm_comparison(solutions)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("#### 📑 Summary Table")
        df_sum = pd.DataFrame([
            {
                "Algorithm": name,
                "Distance (km)": f"{sol.total_distance:.2f}",
                "Cost": f"{sol.total_cost:.2f}",
                "Feasible": "✅" if sol.is_feasible else f"❌ ({sol.constraint_violations})",
                "Runtime": f"{sol.runtime_ms:.1f} ms"
            }
            for name, sol in solutions.items()
        ])
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
    else:
        st.button("Run Baselines", on_click=run_all_classical)
