"""
Page 2: Logistics Dataset & Network Management
"""
import streamlit as st
import pandas as pd
from data_processing import DataLoader, SyntheticDataGenerator, DistanceMatrixCalculator
from visualization.maps import RouteMapVisualizer
from utils.helpers import (
    initialize_session_state,
    update_dataset_and_recalculate,
    run_all_classical
)

initialize_session_state()

st.title("📍 Logistics Network & Dataset Manager")
st.markdown("Inspect, upload, or synthetically generate logistics network stops, customer demands, and delivery priorities.")

# Display any flash messages
if "dataset_status_msg" in st.session_state and st.session_state.dataset_status_msg:
    st.success(st.session_state.dataset_status_msg)

tab_synthetic, tab_upload, tab_sample = st.tabs(["🎲 Synthetic Generator", "📤 Upload Custom CSV", "📁 Sample Dataset"])

# Tab 1: Synthetic Generator
with tab_synthetic:
    st.subheader("Generate Clustered Synthetic Logistics Network")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        num_customers = st.slider(
            "Number of Customer Stops",
            min_value=2,
            max_value=7,
            value=min(max(len(st.session_state.nodes) - 1, 2), 7),
            step=1,
            key="synth_num_cust",
            help="Total locations = Customers + 1 Depot. Required Qubits = Customers²."
        )
        seed = st.number_input(
            "Random Seed (Reproducibility)",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            key="synth_seed"
        )
        
    with col_s2:
        center_lat = st.number_input("Depot Latitude", value=17.3850, format="%.4f", key="synth_center_lat")
        center_lon = st.number_input("Depot Longitude", value=78.4867, format="%.4f", key="synth_center_lon")
        radius_km = st.slider("Cluster Radius (km)", min_value=2.0, max_value=25.0, value=8.0, step=0.5, key="synth_radius_km")
        
    with col_s3:
        demand_min, demand_max = st.slider("Demand Range (units)", 1, 20, (1, 10), key="synth_demand_range")
        prio_min, prio_max = st.slider("Priority Range (1-5)", 1, 5, (1, 5), key="synth_prio_range")
        
    if st.button("✨ Generate Synthetic Network", type="primary", use_container_width=True):
        with st.spinner("Generating logistics stops and computing matrices..."):
            new_nodes = SyntheticDataGenerator.generate(
                num_customers=num_customers,
                seed=int(seed),
                center_lat=float(center_lat),
                center_lon=float(center_lon),
                radius_km=float(radius_km),
                demand_range=(demand_min, demand_max),
                priority_range=(prio_min, prio_max)
            )
            update_dataset_and_recalculate(new_nodes)
            
            best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
            msg = f"✅ Successfully generated {len(new_nodes)} stops ({num_customers} customers + 1 depot)! Distance matrix, cost matrix, QUBO ({num_customers**2} qubits), and classical baselines have all been recalculated (Best Classical: {best_classical_dist:.2f} km)."
            st.session_state.dataset_status_msg = msg
            st.toast(f"Generated {len(new_nodes)} stops! Matrices and baselines updated.", icon="✨")
            st.rerun()

# Tab 2: Upload CSV
with tab_upload:
    st.subheader("Upload Custom Logistics Stops CSV")
    st.markdown("""
    Ensure your CSV adheres to the expected format:
    ```csv
    location_id,latitude,longitude,demand,priority
    DEPOT,17.3850,78.4867,0,0
    C1,17.4000,78.4800,5,3
    C2,17.4100,78.5000,8,2
    ```
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    if uploaded_file is not None:
        parsed_nodes, errors = DataLoader.from_csv_file(uploaded_file)
        if errors:
            for err in errors:
                st.error(f"Validation Error: {err}")
        else:
            cust_count = len(parsed_nodes) - 1
            st.info(f"Detected {len(parsed_nodes)} stops ({cust_count} customers + 1 depot). Required Qubits: {cust_count**2}.")
            if cust_count > 6:
                st.warning(f"⚠️ Note: {cust_count} customers requires {cust_count**2} qubits. Statevector simulation may experience latency on consumer hardware.")
            if st.button("📥 Apply Uploaded Dataset", type="primary", use_container_width=True):
                update_dataset_and_recalculate(parsed_nodes)
                best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
                st.session_state.dataset_status_msg = f"✅ Loaded custom dataset with {len(parsed_nodes)} stops! Matrices and classical baselines re-evaluated (Best Classical: {best_classical_dist:.2f} km)."
                st.toast("Custom dataset loaded successfully!", icon="📥")
                st.rerun()

# Tab 3: Preloaded Sample
with tab_sample:
    st.subheader("Load Standard Hyderabad Metropolitan Test Instance")
    st.caption("A calibrated 4-stop baseline network (1 central depot + 3 customer delivery locations).")
    if st.button("🔄 Reset to Default 4-Node Test Network", use_container_width=True):
        default_nodes = SyntheticDataGenerator.generate(num_customers=3, seed=42)
        update_dataset_and_recalculate(default_nodes)
        st.session_state.dataset_status_msg = "✅ Reset to default 4-node test network! All matrices and baselines restored."
        st.toast("Restored default 4-node dataset!", icon="🔄")
        st.rerun()

st.markdown("---")

nodes = st.session_state.nodes
total_dem = sum(n.demand for n in nodes if not n.is_depot)
m = len(nodes) - 1

# Quick Network Summary Cards
q1, q2, q3, q4 = st.columns(4)
q1.metric("Current Stops", len(nodes), f"{m} Customers + 1 Depot")
q2.metric("Total Customer Demand", f"{total_dem:.1f} units")
q3.metric("QUBO Qubits", m * m, f"m² = {m}²")
q4.metric("Active Distance Metric", st.session_state.distance_metric.capitalize())

st.markdown("<br>", unsafe_allow_html=True)

# Data Table & Map View
st.subheader("📋 Active Logistics Nodes Table")
df_nodes = DataLoader.to_dataframe(nodes)
st.dataframe(df_nodes, use_container_width=True)

# Distance & Cost Matrix Viewers
st.subheader("📏 Pairwise Distance & Augmented Cost Matrices")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown(f"**Distance Matrix ({st.session_state.distance_metric.capitalize()}, km)**")
    df_dist = DistanceMatrixCalculator.to_dataframe(st.session_state.distance_matrix, st.session_state.labels)
    st.dataframe(df_dist, use_container_width=True)

with m_col2:
    st.markdown(f"**Augmented Cost Matrix (w_dist={st.session_state.distance_weight}, w_prio={st.session_state.priority_weight})**")
    df_cost = DistanceMatrixCalculator.to_dataframe(st.session_state.cost_matrix, st.session_state.labels)
    st.dataframe(df_cost, use_container_width=True)
