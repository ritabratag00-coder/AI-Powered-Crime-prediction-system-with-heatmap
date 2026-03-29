import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  — dark, high-contrast theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0a0c12;
    --panel:    #10131d;
    --card:     #141824;
    --border:   rgba(255,255,255,0.08);
    --accent:   #e85d26;
    --accent2:  #ff8c42;
    --text:     #f0f0f0;
    --muted:    #8a8fa8;
    --green:    #22c55e;
    --yellow:   #f59e0b;
    --red:      #ef4444;
}

/* ── App shell ── */
.stApp { background: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: var(--panel) !important; border-bottom: 1px solid var(--border); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Headings — solid bright white, zero ambiguity ── */
h1, h2, h3, h4 {
    font-family: 'Rajdhani', sans-serif !important;
    color: #ffffff !important;
    letter-spacing: 1px;
    text-shadow: none !important;
}
h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; font-weight: 600 !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* ── Body text ── */
p, li, span, div { color: var(--text); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    border-top: 3px solid var(--accent) !important;
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Alert banners ── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    border-left: 4px solid !important;
    font-size: 13px !important;
}

/* ── Charts ── */
[data-testid="stPlotlyChart"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* ── Map iframe ── */
iframe {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: var(--text) !important; font-size: 13px !important; }

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Multiselect & Selectbox — high contrast inputs ── */
[data-baseweb="select"] > div {
    background: #1c2235 !important;
    border: 1.5px solid rgba(232,93,38,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 13px !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #e85d26 !important;
    box-shadow: 0 0 0 2px rgba(232,93,38,0.2) !important;
}
/* Dropdown menu */
[data-baseweb="popover"] [role="listbox"] {
    background: #1c2235 !important;
    border: 1px solid rgba(232,93,38,0.4) !important;
    border-radius: 8px !important;
}
[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: #f0f0f0 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: rgba(232,93,38,0.15) !important;
    color: #ffffff !important;
}
/* Selected tags inside multiselect */
[data-baseweb="tag"] {
    background: rgba(232,93,38,0.25) !important;
    border: 1px solid #e85d26 !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
}
[data-baseweb="tag"] span { color: #ff8c42 !important; font-size: 12px !important; font-weight: 500 !important; }
[data-baseweb="tag"] [role="presentation"] svg { fill: #e85d26 !important; }
/* Placeholder text */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p { color: #8a8fa8 !important; }
input[aria-autocomplete="list"] {
    background: transparent !important;
    color: #ffffff !important;
    font-size: 13px !important;
}
/* Section label above each filter */
.filter-section-label {
    background: rgba(232,93,38,0.1);
    border-left: 3px solid #e85d26;
    border-radius: 0 6px 6px 0;
    padding: 5px 10px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #e85d26 !important;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSlider"] > div > div > div { background: rgba(232,93,38,0.3) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CITY COORDINATES
# ─────────────────────────────────────────────
CITY_COORDS = {
    "Agra":         (27.176, 78.008),
    "Ahmedabad":    (23.022, 72.572),
    "Bangalore":    (12.972, 77.594),
    "Bhopal":       (23.259, 77.412),
    "Chennai":      (13.083, 80.270),
    "Delhi":        (28.704, 77.102),
    "Faridabad":    (28.408, 77.317),
    "Ghaziabad":    (28.669, 77.453),
    "Hyderabad":    (17.385, 78.486),
    "Indore":       (22.719, 75.857),
    "Jaipur":       (26.912, 75.787),
    "Kalyan":       (19.243, 73.136),
    "Kanpur":       (26.449, 80.331),
    "Kolkata":      (22.572, 88.363),
    "Lucknow":      (26.847, 80.946),
    "Ludhiana":     (30.901, 75.857),
    "Meerut":       (28.984, 77.706),
    "Mumbai":       (19.076, 72.877),
    "Nagpur":       (21.145, 79.088),
    "Nashik":       (19.998, 73.789),
    "Patna":        (25.612, 85.143),
    "Pune":         (18.520, 73.857),
    "Rajkot":       (22.303, 70.802),
    "Srinagar":     (34.083, 74.797),
    "Surat":        (21.170, 72.831),
    "Thane":        (19.218, 72.978),
    "Varanasi":     (25.317, 82.973),
    "Vasai":        (19.388, 72.835),
    "Visakhapatnam":(17.686, 83.218),
}


# ─────────────────────────────────────────────
#  DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("crime_data.csv")

    # Parse time of occurrence for hour
    df["datetime"] = pd.to_datetime(df["Time of Occurrence"], format="mixed", errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["hour"]    = df["datetime"].dt.hour
    df["month"]   = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday  # 0=Mon

    # Lat/lon from city
    df["lat"] = df["City"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
    df["lon"] = df["City"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])
    df = df.dropna(subset=["lat", "lon"])

    # Severity score (used for heatmap intensity)
    severity_map = {
        "HOMICIDE": 1.0, "SEXUAL ASSAULT": 0.95, "KIDNAPPING": 0.90,
        "ARSON": 0.85, "ROBBERY": 0.80, "ASSAULT": 0.78,
        "FIREARM OFFENSE": 0.75, "DOMESTIC VIOLENCE": 0.70,
        "DRUG OFFENSE": 0.60, "EXTORTION": 0.60, "BURGLARY": 0.55,
        "VEHICLE - STOLEN": 0.50, "FRAUD": 0.45, "CYBERCRIME": 0.42,
        "COUNTERFEITING": 0.38, "IDENTITY THEFT": 0.35,
        "VANDALISM": 0.30, "ILLEGAL POSSESSION": 0.30,
        "SHOPLIFTING": 0.20, "PUBLIC INTOXICATION": 0.15,
        "TRAFFIC VIOLATION": 0.10,
    }
    df["severity"] = df["Crime Description"].map(severity_map).fillna(0.4)
    return df


# ─────────────────────────────────────────────
#  ML MODEL — Risk Prediction
# ─────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    le_city   = LabelEncoder()
    le_crime  = LabelEncoder()
    le_weapon = LabelEncoder()
    le_gender = LabelEncoder()

    features = df[["hour", "month", "weekday", "Victim Age", "Police Deployed",
                   "City", "Crime Description", "Weapon Used", "Victim Gender"]].copy()
    features["City"]              = le_city.fit_transform(features["City"])
    features["Crime Description"] = le_crime.fit_transform(features["Crime Description"])
    features["Weapon Used"]       = le_weapon.fit_transform(features["Weapon Used"].fillna("Other"))
    features["Victim Gender"]     = le_gender.fit_transform(features["Victim Gender"].fillna("M"))

    # Target: High-severity crime (severity >= 0.6)
    severity_map = {
        "HOMICIDE": 1.0, "SEXUAL ASSAULT": 0.95, "KIDNAPPING": 0.90,
        "ARSON": 0.85, "ROBBERY": 0.80, "ASSAULT": 0.78,
        "FIREARM OFFENSE": 0.75, "DOMESTIC VIOLENCE": 0.70,
        "DRUG OFFENSE": 0.60, "EXTORTION": 0.60, "BURGLARY": 0.55,
        "VEHICLE - STOLEN": 0.50, "FRAUD": 0.45, "CYBERCRIME": 0.42,
        "COUNTERFEITING": 0.38, "IDENTITY THEFT": 0.35,
        "VANDALISM": 0.30, "ILLEGAL POSSESSION": 0.30,
        "SHOPLIFTING": 0.20, "PUBLIC INTOXICATION": 0.15,
        "TRAFFIC VIOLATION": 0.10,
    }
    df["sev"] = df["Crime Description"].map(severity_map).fillna(0.4)
    target = (df["sev"] >= 0.6).astype(int)

    model = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(features, target)
    return model, le_city, le_crime, le_weapon, le_gender


def predict_risk(model, encoders, city, crime_type, weapon, gender, age, hour, police, month, weekday):
    le_city, le_crime, le_weapon, le_gender = encoders
    try:
        city_enc   = le_city.transform([city])[0]
        crime_enc  = le_crime.transform([crime_type])[0]
        weapon_enc = le_weapon.transform([weapon])[0]
        gender_enc = le_gender.transform([gender])[0]
        X = [[hour, month, weekday, age, police, city_enc, crime_enc, weapon_enc, gender_enc]]
        prob = model.predict_proba(X)[0][1]
        return prob
    except Exception:
        return 0.5


# ─────────────────────────────────────────────
#  PLOTLY DARK THEME DEFAULTS
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(20,24,36,1)",
    plot_bgcolor="rgba(20,24,36,1)",
    font=dict(family="Inter, sans-serif", color="#f0f0f0", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    title_font=dict(family="Rajdhani, sans-serif", size=16, color="#ffffff"),
)
ACCENT_COLOR  = "#e85d26"
COLOR_PALETTE = ["#e85d26","#ef4444","#f59e0b","#22c55e","#60a5fa","#a78bfa","#fb7185","#34d399","#fbbf24","#38bdf8"]


# ─────────────────────────────────────────────
#  RISK LEVEL HELPER
# ─────────────────────────────────────────────
def get_risk(density):
    if density < 0.4:    return "SAFE",   "success", "#22c55e"
    elif density < 0.9:  return "LOW",    "success", "#86efac"
    elif density < 1.4:  return "MEDIUM", "warning", "#f59e0b"
    elif density < 2.0:  return "HIGH",   "warning", "#e85d26"
    else:                return "DANGER", "error",   "#ef4444"


# ─────────────────────────────────────────────
#  LOAD DATA + MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading data & training model…"):
    df = load_data()
    model, le_city, le_crime, le_weapon, le_gender = train_model(df)

all_cities  = sorted(df["City"].unique().tolist())
all_crimes  = sorted(df["Crime Description"].unique().tolist())
all_weapons = sorted(df["Weapon Used"].dropna().unique().tolist())
all_genders = sorted(df["Victim Gender"].dropna().unique().tolist())


# ─────────────────────────────────────────────
#  SIDEBAR — FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:8px 0 18px 0;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:16px;'>
        <div style='width:26px;height:26px;background:#e85d26;clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);flex-shrink:0;'></div>
        <span style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;letter-spacing:1px;color:#fff;'>SENTINEL <span style="color:#e85d26">AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-section-label">⏰ Time Filter</div>', unsafe_allow_html=True)
    hour = st.slider("Hour of Day", 0, 23, 12, format="%d:00")

    st.markdown("---")
    st.markdown('<div class="filter-section-label">🗺 Select Cities</div>', unsafe_allow_html=True)
    selected_cities = st.multiselect("Search & select cities", all_cities, default=all_cities[:8], placeholder="Type to search cities…")

    st.markdown("---")
    st.markdown('<div class="filter-section-label">🚔 Crime Filters</div>', unsafe_allow_html=True)
    selected_crimes = st.multiselect("Search & select crime types", all_crimes, default=all_crimes, placeholder="Type to search crime types…")
    selected_domains = st.multiselect("Crime Domain", sorted(df["Crime Domain"].unique().tolist()), default=df["Crime Domain"].unique().tolist(), placeholder="Select domains…")

    st.markdown("---")
    st.markdown('<div class="filter-section-label">👤 Victim Demographics</div>', unsafe_allow_html=True)
    selected_genders = st.multiselect("Gender", all_genders, default=all_genders, placeholder="Select gender…")
    age_range = st.slider("Age Range", int(df["Victim Age"].min()), int(df["Victim Age"].max()), (10, 79))

    st.markdown("---")
    st.markdown('<div class="filter-section-label">🔫 Weapon Filter</div>', unsafe_allow_html=True)
    selected_weapons = st.multiselect("Weapon Used", all_weapons, default=all_weapons, placeholder="Select weapons…")

    st.markdown("---")

    # City risk table in sidebar
    with st.expander("🏙 City Risk Index", expanded=False):
        city_risk_rows = []
        for c in (selected_cities if selected_cities else all_cities):
            cdf = df[(df["City"] == c) & (df["hour"] == hour)]
            density = len(cdf) / max((age_range[1] - age_range[0]), 1)
            lvl, _, col = get_risk(density)
            city_risk_rows.append({"City": c, "Density": round(density, 2), "Risk": lvl})
        risk_df = pd.DataFrame(city_risk_rows).sort_values("Density", ascending=False)
        st.dataframe(risk_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
#  FILTER DATA
# ─────────────────────────────────────────────
if not selected_cities:
    selected_cities = all_cities
if not selected_crimes:
    selected_crimes = all_crimes

filtered = df[
    (df["hour"] == hour) &
    (df["City"].isin(selected_cities)) &
    (df["Crime Description"].isin(selected_crimes)) &
    (df["Crime Domain"].isin(selected_domains if selected_domains else df["Crime Domain"].unique())) &
    (df["Victim Gender"].isin(selected_genders if selected_genders else all_genders)) &
    (df["Victim Age"].between(age_range[0], age_range[1])) &
    (df["Weapon Used"].isin(selected_weapons if selected_weapons else all_weapons))
]

graph_df = df[
    (df["City"].isin(selected_cities)) &
    (df["Crime Description"].isin(selected_crimes)) &
    (df["Crime Domain"].isin(selected_domains if selected_domains else df["Crime Domain"].unique())) &
    (df["Victim Gender"].isin(selected_genders if selected_genders else all_genders)) &
    (df["Victim Age"].between(age_range[0], age_range[1])) &
    (df["Weapon Used"].isin(selected_weapons if selected_weapons else all_weapons))
]


# ─────────────────────────────────────────────
#  RISK COMPUTATION
# ─────────────────────────────────────────────
age_span = max(age_range[1] - age_range[0], 1)
city_densities = [len(filtered[filtered["City"] == c]) / age_span for c in selected_cities]
real_density   = np.mean(city_densities) if city_densities else 0
risk_level, alert_type, risk_color = get_risk(real_density)

case_closure_rate = (filtered["Case Closed"] == "Yes").mean() * 100 if len(filtered) > 0 else 0
avg_police        = filtered["Police Deployed"].mean() if len(filtered) > 0 else 0


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#141824,#0a0c12);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:20px 28px;margin-bottom:20px;
            border-left:4px solid #e85d26;">
    <h1 style="margin:0;line-height:1;">SENTINEL AI — CRIME PREDICTION DASHBOARD</h1>
    <p style="margin:6px 0 0;font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8a8fa8;letter-spacing:1px;">
        // NATIONAL CRIME HEATMAP — INDIA &nbsp;|&nbsp; AI-POWERED RISK ANALYSIS &nbsp;|&nbsp; 40,160 RECORDS
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ALERT BANNER
# ─────────────────────────────────────────────
alert_msgs = {
    "DANGER": "🚨 DANGER — Extremely high crime density detected for the selected filters. Immediate law enforcement attention advised.",
    "HIGH":   "⚠ HIGH RISK — Elevated crime activity detected. Exercise caution in flagged zones.",
    "MEDIUM": "ℹ MEDIUM — Moderate crime activity observed across selected cities and time window.",
    "LOW":    "✅ LOW — Crime activity is relatively low for the selected filters.",
    "SAFE":   "🟢 SAFE — Area appears safe based on current filters and time of day.",
}
getattr(st, alert_type)(alert_msgs[risk_level])


# ─────────────────────────────────────────────
#  KPI METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🚨 Risk Level",       risk_level)
c2.metric("📊 Crime Density",    f"{real_density:.2f}")
c3.metric("📁 Total Incidents",  f"{len(filtered):,}")
c4.metric("🏙 Case Closure Rate",f"{case_closure_rate:.1f}%")
c5.metric("👮 Avg Police Deployed", f"{avg_police:.0f}")

st.markdown("---")


# ─────────────────────────────────────────────
#  HEATMAP
# ─────────────────────────────────────────────
st.markdown("### 🗺 Crime Heatmap — India")

m = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="CartoDB dark_matter",
)

heat_data = filtered[["lat", "lon", "severity"]].dropna().values.tolist()

if len(heat_data) > 0:
    HeatMap(
        heat_data,
        radius=35,
        blur=22,
        max_zoom=14,
        min_opacity=0.4,
        gradient={
            0.1: "#1e3a5f",
            0.3: "#2563eb",
            0.5: "#22c55e",
            0.65:"#f59e0b",
            0.8: "#e85d26",
            1.0: "#ef4444",
        },
    ).add_to(m)

    # City markers
    for city in selected_cities:
        c_df   = filtered[filtered["City"] == city]
        coords = CITY_COORDS.get(city)
        if coords and len(c_df) > 0:
            density = len(c_df) / age_span
            lvl, _, col = get_risk(density)
            top_crime = c_df["Crime Description"].value_counts().index[0] if len(c_df) > 0 else "N/A"
            folium.CircleMarker(
                location=coords,
                radius=max(5, min(18, density * 4)),
                color=col,
                fill=True,
                fill_color=col,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"""<div style='font-family:monospace;font-size:12px;min-width:180px;'>
                        <b style='font-size:14px;'>{city}</b><br>
                        Risk: <b style='color:{col};'>{lvl}</b><br>
                        Incidents: <b>{len(c_df)}</b><br>
                        Top Crime: <b>{top_crime}</b><br>
                        Density: <b>{density:.2f}</b>
                    </div>""",
                    max_width=220
                ),
                tooltip=f"{city} — {lvl}",
            ).add_to(m)
else:
    st.info("No incidents match the current filter combination. Try adjusting the hour or filters.")

st_folium(m, use_container_width=True, height=520)


# ─────────────────────────────────────────────
#  AI RISK PREDICTION PANEL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 AI Risk Predictor")
st.markdown(
    '<p style="color:#8a8fa8;font-size:13px;margin-top:-8px;">Enter a specific scenario to get a machine-learning risk probability.</p>',
    unsafe_allow_html=True
)

with st.form("ai_predict_form"):
    pa, pb, pc = st.columns(3)
    p_city    = pa.selectbox("City",        all_cities,  index=0)
    p_crime   = pb.selectbox("Crime Type",  all_crimes,  index=0)
    p_weapon  = pc.selectbox("Weapon",      all_weapons, index=0)

    pd_col, pe, pf, pg = st.columns(4)
    p_gender  = pd_col.selectbox("Victim Gender", all_genders, index=0)
    p_age     = pe.number_input("Victim Age",   min_value=10, max_value=79, value=28)
    p_hour    = pf.number_input("Hour (0–23)",  min_value=0,  max_value=23, value=hour)
    p_police  = pg.number_input("Police Deployed", min_value=0, max_value=50, value=10)

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    import datetime
    now    = datetime.datetime.now()
    prob   = predict_risk(
        model, (le_city, le_crime, le_weapon, le_gender),
        p_city, p_crime, p_weapon, p_gender,
        p_age, p_hour, p_police, now.month, now.weekday()
    )
    pct = prob * 100
    if pct >= 70:   p_risk, p_color, p_type = "DANGER",  "#ef4444", "error"
    elif pct >= 50: p_risk, p_color, p_type = "HIGH",    "#e85d26", "warning"
    elif pct >= 35: p_risk, p_color, p_type = "MEDIUM",  "#f59e0b", "warning"
    elif pct >= 20: p_risk, p_color, p_type = "LOW",     "#22c55e", "success"
    else:           p_risk, p_color, p_type = "SAFE",    "#22c55e", "success"

    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div style='background:#141824;border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px;padding:22px;text-align:center;
                    border-top:3px solid {p_color};'>
            <p style='font-family:IBM Plex Mono,monospace;font-size:11px;
                      color:#8a8fa8;letter-spacing:1px;text-transform:uppercase;margin:0 0 4px;'>
                AI Risk Score
            </p>
            <p style='font-family:Rajdhani,sans-serif;font-size:52px;
                      font-weight:700;color:{p_color};margin:0;line-height:1;'>
                {pct:.0f}%
            </p>
            <p style='font-family:Rajdhani,sans-serif;font-size:22px;
                      font-weight:600;color:{p_color};margin:6px 0 0;letter-spacing:1px;'>
                {p_risk}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        # Feature importance bar
        importances = model.feature_importances_
        feat_names  = ["Hour", "Month", "Weekday", "Age", "Police", "City", "Crime Type", "Weapon", "Gender"]
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance (Random Forest)",
            color="Importance", color_continuous_scale=["#1e3a5f","#e85d26","#ef4444"]
        )
        fig_fi.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=240)
        st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────
#  ANALYTICS CHARTS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Crime Analytics")

# Row 1: Hourly trend + Crime type breakdown
ch1, ch2 = st.columns([3, 2])

with ch1:
    hourly_counts = graph_df.groupby("hour").size().reset_index(name="count")
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly_counts = all_hours.merge(hourly_counts, on="hour", how="left").fillna(0)

    colors = ["#e85d26" if h == hour else "rgba(255,255,255,0.15)" for h in range(24)]
    fig_hour = go.Figure(go.Bar(
        x=hourly_counts["hour"],
        y=hourly_counts["count"],
        marker_color=colors,
        hovertemplate="Hour %{x}:00 — %{y} incidents<extra></extra>",
    ))
    fig_hour.update_layout(
        **PLOT_LAYOUT,
        title="Crime Activity by Hour of Day",
        xaxis_title="Hour",
        yaxis_title="Incidents",
        xaxis=dict(tickvals=list(range(24)), ticktext=[f"{h:02d}" for h in range(24)]),
        height=320,
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with ch2:
    crime_counts = graph_df["Crime Description"].value_counts().reset_index()
    crime_counts.columns = ["Crime", "Count"]
    fig_pie = px.pie(
        crime_counts.head(8), values="Count", names="Crime",
        title="Crime Type Distribution",
        color_discrete_sequence=COLOR_PALETTE,
        hole=0.45,
    )
    fig_pie.update_layout(**PLOT_LAYOUT, height=320, showlegend=True,
                          legend=dict(font=dict(size=10)))
    fig_pie.update_traces(textfont_size=11, hovertemplate="%{label}: %{value} incidents<extra></extra>")
    st.plotly_chart(fig_pie, use_container_width=True)


# Row 2: City bar + Weapon breakdown
ch3, ch4 = st.columns([3, 2])

with ch3:
    city_stats = graph_df["City"].value_counts().reset_index()
    city_stats.columns = ["City", "Count"]
    city_stats = city_stats.sort_values("Count", ascending=False)

    fig_city = px.bar(
        city_stats, x="City", y="Count",
        title="Crimes by City",
        color="Count",
        color_continuous_scale=["#1e3a5f", "#2563eb", "#e85d26", "#ef4444"],
    )
    fig_city.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320,
                           xaxis_tickangle=-35)
    st.plotly_chart(fig_city, use_container_width=True)

with ch4:
    weapon_counts = graph_df["Weapon Used"].value_counts().reset_index()
    weapon_counts.columns = ["Weapon", "Count"]
    fig_weapon = px.bar(
        weapon_counts, x="Count", y="Weapon", orientation="h",
        title="Weapon Usage",
        color="Count",
        color_continuous_scale=["#1e3a5f","#e85d26","#ef4444"],
    )
    fig_weapon.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False, height=320)
    st.plotly_chart(fig_weapon, use_container_width=True)


# Row 3: Age distribution + Weekday heatmap
ch5, ch6 = st.columns(2)

with ch5:
    fig_age = px.histogram(
        graph_df, x="Victim Age", nbins=25,
        title="Victim Age Distribution",
        color_discrete_sequence=[ACCENT_COLOR],
    )
    fig_age.update_layout(**PLOT_LAYOUT, height=300)
    fig_age.update_traces(hovertemplate="Age %{x} — %{y} victims<extra></extra>")
    st.plotly_chart(fig_age, use_container_width=True)

with ch6:
    pivot = graph_df.groupby(["weekday", "hour"]).size().reset_index(name="count")
    pivot_table = pivot.pivot(index="weekday", columns="hour", values="count").fillna(0)
    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    pivot_table.index = [day_names[i] for i in pivot_table.index]

    fig_hw = px.imshow(
        pivot_table,
        title="Crime Heatmap — Hour × Weekday",
        color_continuous_scale=["#0a0c12","#1e3a5f","#e85d26","#ef4444"],
        aspect="auto",
    )
    fig_hw.update_layout(**PLOT_LAYOUT, height=300)
    st.plotly_chart(fig_hw, use_container_width=True)


# ─────────────────────────────────────────────
#  RAW DATA TABLE
# ─────────────────────────────────────────────
with st.expander("📋 View Raw Filtered Data", expanded=False):
    display_cols = ["Report Number","City","Crime Description","Crime Domain",
                    "Victim Age","Victim Gender","Weapon Used","Police Deployed",
                    "Case Closed","hour"]
    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
    st.caption(f"Showing {len(filtered):,} filtered records out of {len(df):,} total.")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:12px 0;font-family:'IBM Plex Mono',monospace;
            font-size:11px;color:#8a8fa8;letter-spacing:0.8px;">
    SENTINEL AI &nbsp;|&nbsp; POWERED BY RANDOM FOREST ML &nbsp;|&nbsp;
    BUILT WITH STREAMLIT + FOLIUM + PLOTLY
</div>
""", unsafe_allow_html=True)
