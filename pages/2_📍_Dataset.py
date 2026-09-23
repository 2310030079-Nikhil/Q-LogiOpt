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
    st.subheader("Upload Any Logistics or Tabular CSV")
    st.markdown(
        "Upload a formatted logistics CSV or **any general tabular dataset** (e.g. customer lists, survey records, sales data). "
        "Q-LogiOpt will automatically detect columns or adapt records into a calibrated logistics network for quantum route optimization."
    )
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Choose a CSV file to ingest", type=["csv"], key="csv_uploader")
    with col_up2:
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            with open("data/sample_routes.csv", "r") as f:
                sample_csv_data = f.read()
        except Exception:
            sample_csv_data = "location_id,latitude,longitude,demand,priority\nDEPOT,17.3850,78.4867,0,0\nC1,17.4000,78.4800,5,3\nC2,17.4100,78.5000,8,2\n"
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_csv_data,
            file_name="sample_routes.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a pre-formatted logistics stop template"
        )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to parse CSV: {e}")
            raw_df = None

        if raw_df is not None:
            st.markdown("##### 🔍 Uploaded Data Preview & Schema Inspection")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Rows", f"{len(raw_df):,}")
            m_col2.metric("Columns", f"{len(raw_df.columns):,}")
            m_col3.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

            with st.expander("👀 View First 5 Rows of Uploaded File", expanded=False):
                st.dataframe(raw_df.head(5), use_container_width=True)

            # 1. Try strict standard validation
            parsed_nodes, strict_errors = DataLoader.validate_and_parse(raw_df)
            
            # 2. Try fuzzy auto-mapping if strict failed
            auto_nodes = None
            detected_map = {}
            if strict_errors:
                auto_nodes, detected_map, auto_errors = DataLoader.try_auto_map_and_parse(raw_df)

            if parsed_nodes:
                cust_count = len(parsed_nodes) - 1
                st.success(f"✅ **Standard Logistics Schema Detected!** Found {len(parsed_nodes)} stops ({cust_count} customers + 1 depot). Required Qubits: {cust_count**2}.")
                if cust_count > 6:
                    st.warning(f"⚠️ Note: {cust_count} customers requires {cust_count**2} qubits. Statevector simulation may experience latency on consumer hardware.")
                if st.button("📥 Apply Standard Logistics Dataset", type="primary", use_container_width=True):
                    update_dataset_and_recalculate(parsed_nodes)
                    best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
                    st.session_state.dataset_status_msg = f"✅ Loaded custom dataset with {len(parsed_nodes)} stops! Matrices and baselines re-evaluated (Best Classical: {best_classical_dist:.2f} km)."
                    st.toast("Custom dataset loaded successfully!", icon="📥")
                    st.rerun()

            elif auto_nodes:
                cust_count = len(auto_nodes) - 1
                mapped_info = ", ".join([f"`{k}` ➔ `{v}`" for k, v in detected_map.items() if v])
                st.success(f"🎯 **Auto-Mapped Columns Successfully!** ({mapped_info}). Found {len(auto_nodes)} stops ({cust_count} customers + 1 depot).")
                if st.button("📥 Apply Auto-Mapped Logistics Dataset", type="primary", use_container_width=True):
                    update_dataset_and_recalculate(auto_nodes)
                    best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
                    st.session_state.dataset_status_msg = f"✅ Loaded auto-mapped dataset with {len(auto_nodes)} stops! Matrices and baselines re-evaluated (Best Classical: {best_classical_dist:.2f} km)."
                    st.toast("Auto-mapped dataset loaded successfully!", icon="📥")
                    st.rerun()

            else:
                # 3. Flexible Adaptive Conversion Studio for general tabular datasets
                st.info(
                    f"💡 **Adaptive Dataset Conversion Active**: `{uploaded_file.name}` contains `{len(raw_df.columns)}` tabular columns. "
                    "Q-LogiOpt can automatically adapt and convert these records into a calibrated logistics network ready for quantum route optimization."
                )
                
                st.markdown("###### ⚙️ Adaptation & Column Mapping Calibration")
                c_map1, c_map2 = st.columns(2)
                
                with c_map1:
                    id_candidate = DataLoader.detect_column_mapping(raw_df).get("location_id")
                    text_cols = [c for c in raw_df.columns if raw_df[c].dtype == object or pd.api.types.is_string_dtype(raw_df[c])]
                    id_options = ["(Auto-Number Stops)"] + list(raw_df.columns)
                    def_id_idx = id_options.index(id_candidate) if id_candidate in id_options else (id_options.index(text_cols[0]) if text_cols else 0)
                    chosen_id = st.selectbox("Stop Label / Name Column", id_options, index=def_id_idx, help="Column used to name customer delivery locations")
                    
                    num_stops_slider = st.slider(
                        "Number of Stops to Optimize (including Depot)",
                        min_value=3,
                        max_value=min(8, max(3, len(raw_df))),
                        value=min(4, len(raw_df)),
                        step=1,
                        help="Quantum QAOA statevector simulation scales as (N-1)² qubits. 4 stops = 9 qubits, ideal for simulation."
                    )
                    
                with c_map2:
                    numeric_cols = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
                    demand_options = ["(Auto-Assign Realistic Demands: 2-8 units)"] + numeric_cols
                    chosen_dem = st.selectbox("Customer Demand Column (Optional)", demand_options, index=0)
                    
                    prio_options = ["(Auto-Assign Priorities: 1-5)"] + numeric_cols
                    chosen_prio = st.selectbox("Delivery Priority Column (Optional)", prio_options, index=0)

                st.caption("🗺️ Geographic coordinates will be automatically synthesized and clustered around the central Hyderabad depot.")

                if st.button("⚡ Adapt & Load into Quantum Network", type="primary", use_container_width=True):
                    with st.spinner("Adapting dataset and synthesizing logistics network..."):
                        adapted_nodes, notes = DataLoader.adapt_any_dataframe(
                            df=raw_df,
                            num_stops=num_stops_slider,
                            depot_lat=17.3850,
                            depot_lon=78.4867,
                            radius_km=8.0,
                            id_col=chosen_id if chosen_id != "(Auto-Number Stops)" else None,
                            demand_col=chosen_dem if chosen_dem != "(Auto-Assign Realistic Demands: 2-8 units)" else None,
                            priority_col=chosen_prio if chosen_prio != "(Auto-Assign Priorities: 1-5)" else None
                        )
                        update_dataset_and_recalculate(adapted_nodes)
                        best_classical_dist = min([s.total_distance for s in st.session_state.solutions.values()]) if st.session_state.solutions else 0.0
                        st.session_state.dataset_status_msg = (
                            f"✅ Successfully adapted '{uploaded_file.name}' into {len(adapted_nodes)} stops! "
                            f"Matrices, QUBO ({(len(adapted_nodes)-1)**2} qubits), and classical baselines re-evaluated (Best Classical: {best_classical_dist:.2f} km)."
                        )
                        st.toast(f"Adapted {len(adapted_nodes)} stops from {uploaded_file.name}!", icon="⚡")
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
