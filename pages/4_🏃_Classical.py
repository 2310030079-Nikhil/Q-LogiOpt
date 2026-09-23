"""
Page 4: Classical Optimization Baselines
"""
import streamlit as st
import pandas as pd
from utils.helpers import initialize_session_state, run_all_classical
from visualization.charts import ChartVisualizer

initialize_session_state()

st.title("🏃 Classical Optimization Baseline Engines")
st.markdown("Rigorous benchmark solvers providing reference ground truth and heuristic bounds against which quantum solutions are compared.")

nodes = st.session_state.nodes
solutions = st.session_state.solutions

if not solutions:
    run_all_classical()
    solutions = st.session_state.solutions

# Action bar
col_act1, col_act2 = st.columns([8, 2])
with col_act1:
    st.info(f"Currently solving for **{len(nodes)} total stops** ({len(nodes)-1} customer delivery locations + 1 depot).")
with col_act2:
    if st.button("🔄 Re-run All Classical", use_container_width=True):
        with st.spinner("Re-solving..."):
            run_all_classical()
        st.success("Re-evaluated!")
        st.rerun()

st.markdown("---")

# Cards for each classical algorithm
algo_cols = st.columns(4 if len(nodes) <= 8 else 3)

# 1. Nearest Neighbor
with algo_cols[0]:
    st.subheader("1. Nearest Neighbor")
    st.caption("Greedy Heuristic • O(n²)")
    nn = solutions.get("Nearest Neighbor")
    if nn:
        st.metric("Tour Distance", f"{nn.total_distance:.2f} km")
        st.metric("Objective Cost", f"{nn.total_cost:.2f}")
        st.metric("Runtime", f"{nn.runtime_ms:.2f} ms")
        st.markdown(f"**Route:** `{' ➔ '.join(nn.route_ids)}`")
        st.markdown(f"**Feasible:** {'✅ Valid' if nn.is_feasible else '❌ Infeasible'}")

# 2. 2-Opt
with algo_cols[1]:
    st.subheader("2. 2-Opt Search")
    st.caption("Local Search Heuristic")
    two_opt = solutions.get("2-Opt")
    if two_opt:
        st.metric("Tour Distance", f"{two_opt.total_distance:.2f} km")
        st.metric("Objective Cost", f"{two_opt.total_cost:.2f}")
        st.metric("Runtime", f"{two_opt.runtime_ms:.2f} ms")
        st.markdown(f"**Route:** `{' ➔ '.join(two_opt.route_ids)}`")
        st.markdown(f"**Iterations:** `{two_opt.metadata.get('iterations', 0)}`")

# 3. Simulated Annealing
with algo_cols[2]:
    st.subheader("3. Simulated Annealing")
    st.caption("Probabilistic Metaheuristic")
    sa = solutions.get("Simulated Annealing")
    if sa:
        st.metric("Tour Distance", f"{sa.total_distance:.2f} km")
        st.metric("Objective Cost", f"{sa.total_cost:.2f}")
        st.metric("Runtime", f"{sa.runtime_ms:.2f} ms")
        st.markdown(f"**Route:** `{' ➔ '.join(sa.route_ids)}`")
        st.markdown(f"**Iterations:** `{sa.metadata.get('iterations', 0)}`")

# 4. Exact Brute Force (if n <= 8)
if len(nodes) <= 8 and len(algo_cols) == 4:
    with algo_cols[3]:
        st.subheader("4. Exact Ground Truth")
        st.caption("Exhaustive (n-1)! Permutations")
        ex = solutions.get("Exact (Brute Force)")
        if ex:
            st.metric("Tour Distance", f"{ex.total_distance:.2f} km", "Global Optimum")
            st.metric("Objective Cost", f"{ex.total_cost:.2f}")
            st.metric("Runtime", f"{ex.runtime_ms:.2f} ms")
            st.markdown(f"**Route:** `{' ➔ '.join(ex.route_ids)}`")
            st.markdown(f"**Evaluated:** `{ex.metadata.get('total_permutations_evaluated', 0)} tours`")

st.markdown("---")

# Classical Comparison Chart & Full Table
st.subheader("📊 Performance Comparison Table")
records = []
exact_dist = solutions.get("Exact (Brute Force)", solutions.get("2-Opt")).total_distance

for name, sol in solutions.items():
    if "QAOA" not in name:
        approx_ratio = round(sol.total_distance / exact_dist, 3) if exact_dist > 0 else 1.0
        records.append({
            "Algorithm": name,
            "Total Distance (km)": f"{sol.total_distance:.2f}",
            "Objective Cost": f"{sol.total_cost:.2f}",
            "Runtime (ms)": f"{sol.runtime_ms:.2f}",
            "Approximation Ratio (α)": f"{approx_ratio:.3f}",
            "Violations": sol.constraint_violations,
            "Feasibility": "✅ Feasible" if sol.is_feasible else "❌ Violations",
            "Route Sequence": " ➔ ".join(sol.route_ids)
        })

df_res = pd.DataFrame(records)
st.dataframe(df_res, use_container_width=True, hide_index=True)

# Visual comparison bar chart
st.subheader("📈 Classical Solvers Latency & Tour Length")
fig_class = ChartVisualizer.plot_algorithm_comparison({k: v for k, v in solutions.items() if "QAOA" not in k})
st.plotly_chart(fig_class, use_container_width=True)
