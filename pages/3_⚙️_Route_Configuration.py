"""
Page 3: Route Configuration & Constraints Parameterization
"""
import streamlit as st
import numpy as np
from data_processing import DistanceMatrixCalculator
from utils.helpers import (
    initialize_session_state,
    update_configuration_and_recalculate,
    run_all_classical
)

initialize_session_state()

st.title("⚙️ Route Configuration & Objective Weights")
st.markdown("Calibrate the multi-criteria logistics objective function, physical vehicle capacities, and penalty multipliers.")

# Display persistent flash messages if any
if "config_status_msg" in st.session_state and st.session_state.config_status_msg:
    st.success(st.session_state.config_status_msg)

nodes = st.session_state.nodes
total_demand = sum(n.demand for n in nodes if not n.is_depot)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚛 Vehicle & Physical Fleet Limits")
    num_vehicles = st.number_input(
        "Number of Delivery Vehicles",
        min_value=1,
        max_value=3,
        value=1,
        step=1,
        key="cfg_num_vehicles",
        help="For single-vehicle TSP/VRP routing, vehicle=1. Multi-vehicle assigns sub-tours."
    )
    
    cap_val = st.number_input(
        "Vehicle Capacity (units)",
        min_value=10.0,
        max_value=500.0,
        value=float(st.session_state.vehicle_capacity),
        step=5.0,
        key="cfg_cap"
    )
    
    max_d_val = st.number_input(
        "Maximum Route Distance (km)",
        min_value=20.0,
        max_value=1000.0,
        value=float(st.session_state.max_distance),
        step=10.0,
        key="cfg_max_dist"
    )
    
    dist_metric = st.selectbox(
        "Distance Metric",
        options=["haversine", "euclidean"],
        index=0 if st.session_state.distance_metric == "haversine" else 1,
        key="cfg_metric",
        help="Haversine uses spherical great-circle coordinates; Euclidean treats lat/lon as planar coordinates."
    )

with col2:
    st.subheader("⚖️ Objective Weights & QUBO Penalty")
    w_dist = st.slider(
        "Distance Weight (w_dist)",
        min_value=0.1,
        max_value=5.0,
        value=float(st.session_state.distance_weight),
        step=0.1,
        key="cfg_w_dist"
    )
    
    w_prio = st.slider(
        "Priority Weight (w_prio)",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state.priority_weight),
        step=0.1,
        key="cfg_w_prio"
    )
    
    max_edge = float(np.max(st.session_state.distance_matrix))
    recommended_lambda = round(max_edge * 2.0, 1)
    
    pen_lambda = st.number_input(
        f"Constraint Penalty Multiplier λ (Recommended: >= {recommended_lambda:.1f})",
        min_value=10.0,
        max_value=2000.0,
        value=max(10.0, float(st.session_state.penalty_lambda)),
        step=10.0,
        key="cfg_lambda",
        help="Penalty for violating permutation constraints in QUBO. Must exceed maximum travel cost to guarantee a valid ground state."
    )

# Save configurations
if st.button("💾 Apply Configuration & Recalculate Matrices", type="primary", use_container_width=True):
    with st.spinner("Applying new parameters and recalculating cost matrices & classical baselines..."):
        update_configuration_and_recalculate(
            vehicle_capacity=cap_val,
            max_distance=max_d_val,
            distance_metric=dist_metric,
            distance_weight=w_dist,
            priority_weight=w_prio,
            penalty_lambda=pen_lambda
        )
        
        best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
        msg = f"✅ Configuration updated successfully! Distance metric ({dist_metric}), cost matrix (w_dist={w_dist}, w_prio={w_prio}), QUBO matrix (λ={pen_lambda:.1f}), and classical baselines have all been re-evaluated (Best Classical: {best_classical_dist:.2f} km)."
        st.session_state.config_status_msg = msg
        st.toast("Configuration updated & matrices recalculated!", icon="💾")
        st.rerun()

st.markdown("---")

# Pre-Optimization Feasibility Audit
st.subheader("🔍 Pre-Optimization Feasibility Audit")
f_cols = st.columns(3)

with f_cols[0]:
    st.write(f"**Total Customer Demand:** `{total_demand:.1f}` units")
    st.write(f"**Vehicle Capacity:** `{st.session_state.vehicle_capacity:.1f}` units")
    if total_demand <= st.session_state.vehicle_capacity:
        st.success("✅ Capacity Constraint Feasible")
    else:
        st.warning(f"⚠️ Demand ({total_demand:.1f}) exceeds vehicle capacity ({st.session_state.vehicle_capacity:.1f}). A capacity penalty will apply.")

with f_cols[1]:
    st.write(f"**Configured Max Distance:** `{st.session_state.max_distance:.1f}` km")
    # Estimate minimum MST or NN tour distance
    est_min_dist = float(np.sum(np.min(st.session_state.distance_matrix + np.eye(len(nodes)) * 1e5, axis=1)))
    st.write(f"**Estimated Lower Bound:** `{est_min_dist:.1f}` km")
    if est_min_dist <= st.session_state.max_distance:
        st.success("✅ Distance Limit Feasible")
    else:
        st.warning(f"⚠️ Tour distance is likely to exceed max limit ({st.session_state.max_distance:.1f} km).")

with f_cols[2]:
    st.write(f"**QUBO Penalty λ:** `{st.session_state.penalty_lambda:.1f}`")
    if st.session_state.penalty_lambda >= recommended_lambda:
        st.success("✅ Penalty Sufficient for Feasibility")
    else:
        st.warning(f"⚠️ Low λ (< {recommended_lambda:.1f}) may cause QAOA to sample invalid permutations.")

st.markdown("---")

# Current Matrix Previews
st.subheader("📊 Active Cost Matrix Preview")
df_cost = DistanceMatrixCalculator.to_dataframe(st.session_state.cost_matrix, st.session_state.labels)
st.dataframe(df_cost, use_container_width=True)

st.markdown("""
### 📐 Logistics Objective Function Formulation
The operational cost of a closed logistics tour $\\pi = (v_0, v_{\\pi(1)}, \\dots, v_{\\pi(n-1)}, v_0)$ is evaluated as:
$$\\text{Cost}(\\pi) = \\sum_{k=0}^{n-1} \\left( w_{\\text{dist}} \\cdot D_{\\pi(k), \\pi(k+1)} + w_{\\text{prio}} \\cdot P_{\\pi(k+1)} \\right) + \\lambda_{\\text{cap}} \\cdot \\max(0, \\text{Demand} - Q_{\\text{cap}}) + \\lambda_{\\text{dist}} \\cdot \\max(0, \\text{Dist} - D_{\\text{max}})$$
where:
* $D_{i,j}$ is the distance between stops $i$ and $j$.
* $P_j$ is the priority delivery penalty of stop $j$ ($P_j = \\max(\\text{prio}) - \\text{prio}_j$).
* $Q_{\\text{cap}}$ is the vehicle payload capacity limit.
* $\\lambda$ represents the constraint penalty multiplier.
""")
