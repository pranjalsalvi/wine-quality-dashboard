"""
SPRINT 2 – Interactive Streamlit Data Dashboard
Wine Quality Analysis | Screaming Eagle Winery (Case Study)
Run: streamlit run sprint2_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine Quality Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {
        background-color: #F9F4EF;
    }

    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E6E6;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* KPI Label */
    /* KPI Label - Force Black */
    label[data-testid="stMetricLabel"],
    label[data-testid="stMetricLabel"] *,
    label[data-testid="stMetricLabel"] p {
        color: #000000 !important;
        fill: #000000 !important;
        opacity: 1 !important;
    }

    /* KPI Value */
    /* KPI Value - Force Black */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {
        color: #000000 !important;
        fill: #000000 !important;
        opacity: 1 !important;
    }

    h1 {
        color: #6B1A1A;
        font-family: 'Georgia', serif;
    }

    h2, h3 {
        color: #8B2C2C;
    }

    .block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Color Theme ────────────────────────────────────────────────────────────
WINE_RED = "#8B1A1A"
WINE_GOLD = "#C4A35A"
WINE_CREAM = "#F5ECD7"

PALETTE = [
    WINE_RED,
    WINE_GOLD,
    "#5B86A8",
    "#4A7A5B",
    "#8B6914"
]

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists("wine_quality.csv"):
        df = pd.read_csv("wine_quality.csv")
    else:
        red   = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",   sep=";")
        white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
        red["color"]   = "red"
        white["color"] = "white"
        df = pd.concat([red, white], ignore_index=True)
        df["good"] = (df["quality"] >= 7).astype(int)
        df.to_csv("wine_quality.csv", index=False)
    return df

df = load_data()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ["quality", "good"]]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Wine_grapes03.jpg/320px-Wine_grapes03.jpg",
    width="stretch"
)

wine_color = st.sidebar.multiselect(
    "Wine Color", options=df["color"].unique(), default=list(df["color"].unique())
)
quality_range = st.sidebar.slider(
    "Quality Score Range", int(df["quality"].min()), int(df["quality"].max()),
    (int(df["quality"].min()), int(df["quality"].max()))
)
selected_feature = st.sidebar.selectbox("Feature to Analyze", feature_cols, index=0)

df_filtered = df[
    (df["color"].isin(wine_color)) &
    (df["quality"] >= quality_range[0]) &
    (df["quality"] <= quality_range[1])
]

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("🍷 Wine Quality Intelligence Dashboard")
st.markdown("> *Empowering data-driven decisions at the world's premier wineries.*")
st.markdown("---")

# ─── KPI Row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Wines",    f"{len(df_filtered):,}")
col2.metric("Avg Quality",    f"{df_filtered['quality'].mean():.2f}")
col3.metric("Avg Alcohol %",  f"{df_filtered['alcohol'].mean():.1f}%")
col4.metric("Good Wines",     f"{df_filtered['good'].sum():,}  ({df_filtered['good'].mean()*100:.0f}%)")
col5.metric("Unique Scores",  f"{df_filtered['quality'].nunique()}")

st.markdown("---")

# ─── Tab Layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔬 Feature Analysis", "🔥 Correlations",
    "⚗️ Chemistry Deep Dive", "📋 Data Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Wine Quality Distribution")
    r1c1, r1c2 = st.columns(2)

    # Quality histogram
    fig_hist = px.histogram(
        df_filtered, x="quality", color="color",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        barmode="group", nbins=10,
        title="Quality Score Distribution by Wine Type",
        labels={"quality": "Quality Score", "count": "Count"},
        template="plotly_white",
    )
    fig_hist.update_traces(marker_line_width=1, marker_line_color="white")
    r1c1.plotly_chart(fig_hist, width="stretch")

    # Pie: Good vs Not-Good
    good_cnt     = df_filtered["good"].sum()
    not_good_cnt = len(df_filtered) - good_cnt
    fig_pie = go.Figure(go.Pie(
        labels=["Premium (≥7)", "Standard (<7)"],
        values=[good_cnt, not_good_cnt],
        hole=0.45,
        marker_colors=[WINE_RED, WINE_GOLD],
        textinfo="label+percent",
    ))
    fig_pie.update_layout(title="Premium vs Standard Wine Ratio", template="plotly_white")
    r1c2.plotly_chart(fig_pie, width="stretch")

    # Box: Quality by color
    fig_box = px.box(
        df_filtered, x="color", y="quality", color="color",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        title="Quality Spread by Wine Color",
        points="outliers", template="plotly_white",
    )
    r1c1.plotly_chart(fig_box, width="stretch")

    # Violin: Alcohol by quality
    fig_vio = px.violin(
        df_filtered, x="quality", y="alcohol", color="color",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        box=True, title="Alcohol Distribution by Quality Score",
        template="plotly_white",
    )
    r1c2.plotly_chart(fig_vio, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Analyzing: **{selected_feature}**")
    c1, c2 = st.columns(2)

    # Distribution
    fig_dist = px.histogram(
        df_filtered, x=selected_feature, color="color",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        marginal="box", barmode="overlay", opacity=0.7,
        title=f"Distribution of {selected_feature}",
        template="plotly_white",
    )
    c1.plotly_chart(fig_dist, width="stretch")

    # Feature vs Quality scatter
    fig_scat = px.scatter(
        df_filtered, x=selected_feature, y="quality",
        color="color", opacity=0.4, trendline="ols",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        title=f"{selected_feature} vs Quality",
        template="plotly_white",
    )
    c2.plotly_chart(fig_scat, width="stretch")

    # Mean quality by feature quartile
    df_q = df_filtered.copy()
    df_q["quartile"] = pd.qcut(df_q[selected_feature], q=4,
                                labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"],
                                duplicates="drop")
    mean_q = df_q.groupby("quartile", observed=False)["quality"].mean().reset_index()
    fig_bar = px.bar(
        mean_q, x="quartile", y="quality",
        color="quality", color_continuous_scale="RdYlGn",
        title=f"Average Quality by {selected_feature} Quartile",
        text_auto=".2f", template="plotly_white",
    )
    fig_bar.update_traces(marker_line_width=1, marker_line_color="white")
    c1.plotly_chart(fig_bar, width="stretch")

    # Box by quality score
    fig_boxq = px.box(
        df_filtered, x="quality", y=selected_feature, color="color",
        color_discrete_map={"red": WINE_RED, "white": WINE_GOLD},
        title=f"{selected_feature} by Quality Score",
        template="plotly_white",
    )
    c2.plotly_chart(fig_boxq, width="stretch")

    # Stats table
    st.subheader("Descriptive Statistics")
    stats_df = df_filtered.groupby("color")[selected_feature].describe().round(3)
    st.dataframe(stats_df, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Correlation Analysis")

    # Heatmap
    corr_matrix = df_filtered[feature_cols + ["quality"]].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values.round(2),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate="%{text}",
        colorbar_title="Correlation",
    ))
    fig_heat.update_layout(
        title="Correlation Heatmap – All Features",
        height=600, template="plotly_white",
    )
    st.plotly_chart(fig_heat, width="stretch")

    # Bar: Correlation with quality
    quality_corr = corr_matrix["quality"].drop("quality").sort_values()
    bar_colors = [WINE_RED if v < 0 else "#4A7A5B" for v in quality_corr.values]
    fig_corr_bar = go.Figure(go.Bar(
        x=quality_corr.values.round(3),
        y=quality_corr.index,
        orientation="h",
        marker_color=bar_colors,
        text=quality_corr.values.round(3),
        textposition="outside",
    ))
    fig_corr_bar.update_layout(
        title="Feature Correlations with Wine Quality",
        xaxis_title="Pearson r", template="plotly_white", height=420,
    )
    st.plotly_chart(fig_corr_bar, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CHEMISTRY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Chemical Composition Analysis")

    # Radar chart: avg chemistry by wine type
    radar_features = ["alcohol", "volatile acidity", "sulphates",
                      "density", "pH", "chlorides"]
    radar_data = df_filtered.groupby("color")[radar_features].mean()

    fig_radar = go.Figure()
    for color_type, row in radar_data.iterrows():
        norm = (row - df_filtered[radar_features].min()) / \
               (df_filtered[radar_features].max() - df_filtered[radar_features].min())
        clr = WINE_RED if color_type == "red" else WINE_GOLD
        fig_radar.add_trace(go.Scatterpolar(
            r=norm.values.tolist() + [norm.values[0]],
            theta=radar_features + [radar_features[0]],
            fill="toself", opacity=0.5, name=color_type.capitalize(),
            line_color=clr,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Chemical Profile: Red vs White Wine (Normalized)",
        template="plotly_white", showlegend=True,
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Parallel coordinates
    c1, c2 = st.columns(2)
    plot_cols = ["alcohol", "volatile acidity", "sulphates", "density", "quality"]
    sample = df_filtered[plot_cols + ["color"]].dropna().sample(min(1500, len(df_filtered)),
                                                                  random_state=42)
    sample["color_num"] = (sample["color"] == "red").astype(int)

    fig_par = px.parallel_coordinates(
        sample, dimensions=plot_cols,
        color="quality", color_continuous_scale=px.colors.sequential.RdBu,
        title="Parallel Coordinates: Key Features vs Quality",
    )
    fig_par.update_layout(template="plotly_white")
    c1.plotly_chart(fig_par, width="stretch")

    # 3D scatter
    fig_3d = px.scatter_3d(
        sample, x="alcohol", y="volatile acidity", z="sulphates",
        color="quality", color_continuous_scale="RdYlGn",
        opacity=0.6, size_max=6,
        title="3D: Alcohol × Volatile Acidity × Sulphates",
    )
    c2.plotly_chart(fig_3d, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Raw Data Explorer")
    st.info(f"Showing {len(df_filtered):,} rows after current filter selections.")

    col_select = st.multiselect("Columns to Display", df.columns.tolist(),
                                default=df.columns.tolist())
    sort_col   = st.selectbox("Sort by", numeric_cols, index=numeric_cols.index("quality"))
    sort_asc   = st.radio("Sort Direction", ["Descending", "Ascending"], horizontal=True)

    display_df = df_filtered[col_select].sort_values(
        sort_col, ascending=(sort_asc == "Ascending")
    )
    st.dataframe(display_df, width="stretch", height=420)

    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=display_df.to_csv(index=False).encode(),
        file_name="wine_quality_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Summary Statistics")
    st.dataframe(df_filtered[feature_cols + ["quality"]].describe().round(3),
                 width="stretch")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🍷 Wine Quality Intelligence | Built with Streamlit & Plotly "
    "| Data: UCI Wine Quality Dataset</small></center>",
    unsafe_allow_html=True,
)