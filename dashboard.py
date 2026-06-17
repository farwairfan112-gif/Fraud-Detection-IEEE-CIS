# dashboard.py
# Author: Farwa Irfan | MS Data Science, FAST-NUCES
# Fraud Detection Dashboard — IEEE-CIS Dataset (Flagged Version)

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
#from joblib import load
import pickle
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
st.set_page_config(page_title="Fraud Detection Dashboard | Farwa Irfan", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6C63FF;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #A8E6CF 0%, #D4F1F4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #6C63FF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .success-card {
        background: linear-gradient(135deg, #C7F9CC 0%, #E3FCE9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-card {
        background: linear-gradient(135deg, #FFD6BA 0%, #FFE8E1 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background: linear-gradient(135deg, #D4C1EC 0%, #F2E8FF 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #9370DB;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 👩‍💻 About")
    st.markdown("**Farwa Irfan**")
    st.markdown("MS Data Science — FAST-NUCES")
    st.markdown("[GitHub](https://github.com/farwairfan112-gif)")
    st.markdown("---")
    st.markdown("### 🛡️ Project")
    st.markdown("IEEE-CIS Fraud Detection")
    st.markdown("**Best Model:** LightGBM + Threshold Tuning")
    st.markdown("**ROC-AUC:** 0.9107")
    st.markdown("**F1-Score:** 0.5402")

# ---------- PATHS (relative — place data/ and models/ folders next to this file) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_BEFORE_PATH  = os.path.join(BASE_DIR, "data", "df_before.csv")
# LEAKAGE FIX: df_train should be training data only (not full merged df)
# Save your X_train (after preprocessing, before SMOTE) as df_train.csv
DF_TRAIN_PATH   = os.path.join(BASE_DIR, "data", "df_train.csv")
DF_CLEAN_PATH   = os.path.join(BASE_DIR, "data", "df_clean.csv")

MODELS_PATHS = {
    'lgb': os.path.join(BASE_DIR, "models", "lgb_best.pkl"),
    'rf':  os.path.join(BASE_DIR, "models", "rf_model_sm.pkl"),
    'xgb': os.path.join(BASE_DIR, "models", "xgb_model_sm.pkl"),
    'log': os.path.join(BASE_DIR, "models", "log_model_sm.pkl")
}

# ---------- HELPERS ----------
@st.cache_data
def load_data(before_path, train_path, clean_path):
    try:
        df_before = pd.read_csv(before_path)
    except Exception as e:
        st.warning(f"Could not load df_before: {e}")
        df_before = pd.DataFrame()
    try:
        # Use training data only for defaults — avoids leaking test distribution
        df_train = pd.read_csv(train_path)
    except Exception:
        df_train = pd.DataFrame()
    try:
        df_clean = pd.read_csv(clean_path)
    except Exception as e:
        st.warning(f"Could not load df_clean: {e}")
        df_clean = pd.DataFrame()
    return df_before, df_train, df_clean

@st.cache_resource
def load_models(paths):
    models = {}
    for key, path in paths.items():
        try:
            models[key] = joblib.load(path)
        except Exception as e:
            st.warning(f"Model '{key}' not loaded: {e}")
    return models
def get_feature_defaults(df_train):
    """Derive defaults from TRAINING data only to prevent leakage."""
    defaults = {}
    cat_maps = {}
    dtypes = {}
    if df_train.empty:
        return defaults, cat_maps, dtypes
    for col in df_train.columns:
        dtypes[col] = str(df_train[col].dtype)
        if pd.api.types.is_numeric_dtype(df_train[col]):
            defaults[col] = float(df_train[col].median(skipna=True)) if df_train[col].notna().any() else 0.0
        else:
            cats = pd.Series(df_train[col].fillna("<<NA>>")).astype('category')
            mapping = {cat: int(code) for code, cat in enumerate(cats.cat.categories)}
            cat_maps[col] = mapping
            mode_val = cats.mode().iloc[0] if not cats.mode().empty else cats.iloc[0]
            defaults[col] = mapping.get(mode_val, 0)
    return defaults, cat_maps, dtypes

def build_sample_from_user_inputs(feature_names, defaults, cat_maps, dtypes, user_inputs):
    row = {}
    for feat in feature_names:
        if feat in user_inputs:
            row[feat] = user_inputs[feat]
        elif feat in defaults:
            row[feat] = defaults[feat]
        else:
            row[feat] = 0.0
    sample = pd.DataFrame([row], columns=feature_names)
    sample = sample.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return sample

# ---------- LOAD ----------
df_before, df_train, df_clean = load_data(DF_BEFORE_PATH, DF_TRAIN_PATH, DF_CLEAN_PATH)
models = load_models(MODELS_PATHS)
defaults, cat_maps, dtypes = get_feature_defaults(df_train)

# ---------- PERFORMANCE DATA (Flagged Version Only) ----------
baseline_with_flags = pd.DataFrame({
    'Model': ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
    'Accuracy': [0.9700, 0.9736, 0.9715, 0.9732],
    'Precision': [0.6917, 0.8186, 0.6569, 0.7312],
    'Recall': [0.2308, 0.2975, 0.3590, 0.3514],
    'F1-Score': [0.3461, 0.4364, 0.4643, 0.4747],
    'ROC-AUC': [0.8466, 0.8844, 0.8976, 0.9046],
    'PR-AUC': [0.3833, 0.5003, 0.4800, 0.5026]
})

smote_with_flags = pd.DataFrame({
    'Model': ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
    'Accuracy': [0.8023, 0.9739, 0.9725, 0.9726],
    'Precision': [0.1184, 0.7610, 0.6791, 0.7011],
    'Recall': [0.7362, 0.3519, 0.3797, 0.3573],
    'F1-Score': [0.2040, 0.4812, 0.4871, 0.4733],
    'ROC-AUC': [0.8507, 0.8968, 0.8922, 0.8950],
    'PR-AUC': [0.3553, 0.5131, 0.4938, 0.4850]
})

lgbm_tuned = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"],
    "Value":  [0.9749, 0.9107, 0.5432, 0.7698, 0.3858, 0.5140]
})

lgbm_threshold_final = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"],
    "Value":  [0.9711, 0.9107, 0.5432, 0.5980, 0.4926, 0.5402]
})

# ---------- NAVIGATION ----------
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

nav_items = {
    "home":         " Home",
    "data-overview":" Data Overview",
    "before-preprocessing": " Before Preprocessing",
    "after-preprocessing":  " After Preprocessing",
    "with-flags":   " With Flags",
    "optimized":    " Optimized LightGBM",
    "prediction":   " Real-Time Prediction",
    "conclusions":  " Conclusions"
}

cols = st.columns(len(nav_items))
for i, (page_key, page_label) in enumerate(nav_items.items()):
    with cols[i]:
        if st.button(page_label, use_container_width=True,
                     type="primary" if st.session_state.current_page == page_key else "secondary"):
            st.session_state.current_page = page_key
            st.rerun()

current_page = st.session_state.current_page

# ---------- PAGES ----------

if current_page == "home":
    st.markdown('<div class="main-header"> Fraud Detection Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3> About This Project</h3>
            <p>End-to-end fraud detection pipeline on the IEEE-CIS dataset (590K+ transactions).
            Compares XGBoost, LightGBM, Random Forest, and Logistic Regression with missing-value
            flag engineering, SMOTE, hyperparameter tuning, threshold optimization, and temporal validation.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="success-card">
            <h3>🏆 Best Model</h3>
            <p><strong>LightGBM + Threshold 0.20</strong><br>
            F1-Score: 0.5402<br>
            Recall: 0.4926<br>
            Precision: 0.5980<br>
            ROC-AUC: 0.9107</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###  Model Performance at a Glance")

    perf_df = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Random Forest',
            'XGBoost',
            'LightGBM (SMOTE)',
            'LightGBM (Tuned + Threshold 0.20)'
        ],
        'F1-Score': [0.2040, 0.4812, 0.4871, 0.4733, 0.5402],
        'Recall':   [0.7362, 0.3519, 0.3797, 0.3573, 0.4926],
        'Precision':[0.1184, 0.7610, 0.6791, 0.7011, 0.5980]
    })
    fig = px.bar(perf_df, x='Model', y=['F1-Score', 'Recall', 'Precision'],
                 title="Model Performance Comparison (With Flags)",
                 barmode='group',
                 color_discrete_sequence=['#6C63FF', '#4CAF50', '#FF6B6B'])
    st.plotly_chart(fig, use_container_width=True)

elif current_page == "data-overview":
    st.markdown('<div class="main-header"> Data Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card"><h3> Raw Dataset</h3><p>Before preprocessing</p></div>', unsafe_allow_html=True)
        if not df_before.empty:
            st.write(f"**Shape:** {df_before.shape}")
            st.dataframe(df_before.head(), use_container_width=True)
        else:
            st.warning("df_before.csv not found in data/ folder.")
    with col2:
        st.markdown('<div class="success-card"><h3> Processed Dataset</h3><p>After preprocessing + flags</p></div>', unsafe_allow_html=True)
        if not df_clean.empty:
            st.write(f"**Shape:** {df_clean.shape}")
            st.dataframe(df_clean.head(), use_container_width=True)
        else:
            st.warning("df_clean.csv not found in data/ folder.")

    if not df_before.empty and not df_clean.empty:
        st.markdown("---")
        st.markdown("###  Missing Values Comparison")
        mv = pd.DataFrame({
            "Before": df_before.isnull().sum(),
            "After":  df_clean.isnull().sum()
        })
        st.dataframe(mv[mv["Before"] > 0].sort_values("Before", ascending=False).head(30),
                     use_container_width=True)

elif current_page == "before-preprocessing":
    st.markdown('<div class="main-header"> Before Preprocessing</div>', unsafe_allow_html=True)

    if not df_before.empty:
        num_cols = df_before.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_col = st.selectbox("Select Numeric Column", num_cols)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df_before, x=selected_col, nbins=40,
                                   title=f"Distribution of {selected_col}",
                                   color_discrete_sequence=['#FFB6C1'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df_before, y=selected_col,
                             title=f"Boxplot of {selected_col}",
                             color_discrete_sequence=['#87CEEB'])
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_before[selected_col].describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found.")
    else:
        st.error("df_before.csv not found in data/ folder.")

elif current_page == "after-preprocessing":
    st.markdown('<div class="main-header"> After Preprocessing</div>', unsafe_allow_html=True)

    if not df_clean.empty:
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            selected_col = st.selectbox("Select Numeric Column", num_cols)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df_clean, x=selected_col, nbins=40,
                                   title=f"Distribution of {selected_col}",
                                   color_discrete_sequence=['#98FB98'])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df_clean, y=selected_col,
                             title=f"Boxplot of {selected_col}",
                             color_discrete_sequence=['#DDA0DD'])
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_clean[selected_col].describe(), use_container_width=True)
        else:
            st.warning("No numeric columns found.")
    else:
        st.error("df_clean.csv not found in data/ folder.")

elif current_page == "with-flags":
    st.markdown('<div class="main-header"> Model Comparison — With Missing-Value Flags</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([" Baseline", " After SMOTE"])

    with tab1:
        st.markdown('<div class="info-card"><h4>Baseline Models — With Flags</h4></div>', unsafe_allow_html=True)
        st.dataframe(baseline_with_flags.style.highlight_max(axis=0, color='#C7F9CC'), use_container_width=True)
        fig = px.bar(baseline_with_flags, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     title="Baseline Performance (With Flags)", barmode='group',
                     color_discrete_sequence=['#6C63FF', '#4CAF50', '#FF6B6B', '#9370DB'])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="success-card"><h4>SMOTE Retrained — With Flags</h4></div>', unsafe_allow_html=True)
        st.dataframe(smote_with_flags.style.highlight_max(axis=0, color='#C7F9CC'), use_container_width=True)
        fig = px.bar(smote_with_flags, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     title="SMOTE Performance (With Flags)", barmode='group',
                     color_discrete_sequence=['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD'])
        st.plotly_chart(fig, use_container_width=True)

elif current_page == "optimized":
    st.markdown('<div class="main-header">🚀 Optimized LightGBM</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs([" Hyperparameter Tuned", "⚡ Threshold Tuned (0.20)"])

    with tab1:
        st.markdown('<div class="info-card"><h4>LightGBM — Hyperparameter Tuned</h4></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(lgbm_tuned, use_container_width=True)
        with col2:
            for _, row in lgbm_tuned.iterrows():
                st.metric(row['Metric'], row['Value'])

    with tab2:
        st.markdown('<div class="success-card"><h4>LightGBM — Threshold 0.20 (Final Model)</h4></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(lgbm_threshold_final, use_container_width=True)
        with col2:
            for _, row in lgbm_threshold_final.iterrows():
                st.metric(row['Metric'], row['Value'])

    st.markdown("---")
    st.markdown("###  LightGBM Progression Across Stages")

    progression = pd.DataFrame({
        'Stage':     ['Baseline', 'After SMOTE', 'Hyperparameter Tuned', 'Threshold 0.20'],
        'F1-Score':  [0.4733, 0.4733, 0.5140, 0.5402],
        'Recall':    [0.3573, 0.3573, 0.3858, 0.4926],
        'Precision': [0.7011, 0.7011, 0.7698, 0.5980]
    })
    fig = px.line(progression, x='Stage', y=['F1-Score', 'Recall', 'Precision'],
                  title="LightGBM Performance Across Pipeline Stages", markers=True,
                  color_discrete_sequence=['#6C63FF', '#4CAF50', '#FF6B6B'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="warning-card">
        <h4> Key Takeaway</h4>
        <ul>
            <li><strong>Hyperparameter Tuning</strong>: Improved precision (73.58%) but recall dropped to 40.85%</li>
            <li><strong>Threshold 0.20</strong>: Better balance — recall up to 52.24%, F1 0.5362</li>
            <li><strong>Missing-value flags</strong> consistently improved minority class detection across all stages</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif current_page == "prediction":
    st.markdown('<div class="main-header">🔮 Real-Time Fraud Prediction</div>', unsafe_allow_html=True)

    if 'lgb' not in models:
        st.error("⚠️ LightGBM model not loaded. Add lgbm_model.joblib to the models/ folder.")
        st.stop()

    st.markdown("""
    <div class="info-card">
        <h3>💡 How It Works</h3>
        <p>Enter transaction details below. All other features are filled using training-set medians
        (no test data leakage). Uses threshold τ = 0.20 as optimized during model training.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("Transaction Details")
        col1, col2 = st.columns(2)

        with col1:
            TransactionAmt = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
            hour = st.slider("Transaction Hour", 0, 23, 12)
            product_options = sorted(df_train['ProductCD'].astype(str).unique().tolist()) if 'ProductCD' in df_train.columns else ['W', 'H', 'C', 'S', 'R']
            ProductCD = st.selectbox("Product Code", options=product_options)

        with col2:
            card1_default = float(df_train['card1'].median()) if 'card1' in df_train.columns else 0.0
            card1 = st.number_input("Card 1", value=card1_default)
            card2_default = float(df_train['card2'].median()) if 'card2' in df_train.columns else 0.0
            card2 = st.number_input("Card 2", value=card2_default)
            card4_options = sorted(df_train['card4'].astype(str).unique().tolist()) if 'card4' in df_train.columns else ['visa', 'mastercard']
            card4 = st.selectbox("Card Type", options=card4_options)
            card6_options = sorted(df_train['card6'].astype(str).unique().tolist()) if 'card6' in df_train.columns else ['credit', 'debit']
            card6 = st.selectbox("Card Category", options=card6_options)

        submitted = st.form_submit_button(" Predict Fraud Risk", use_container_width=True)

    if submitted:
        try:
            model = models['lgb']
            try:
                feature_names = model.feature_name_
            except AttributeError:
                try:
                    feature_names = model.booster_.feature_name()
                except AttributeError:
                    try:
                        feature_names = model.feature_names_in_
                    except AttributeError:
                        feature_names = df_train.columns.tolist()
                        st.warning("Using training dataset columns as feature names.")

            user_inputs = {}
            mapping_dict = {
                'TransactionAmt': TransactionAmt,
                'hour': hour,
                'ProductCD': ProductCD,
                'card1': card1,
                'card2': card2,
                'card4': card4,
                'card6': card6
            }
            for feature, value in mapping_dict.items():
                if feature in feature_names:
                    if feature in cat_maps:
                        user_inputs[feature] = cat_maps[feature].get(str(value), defaults.get(feature, 0))
                    else:
                        user_inputs[feature] = float(value)

            sample = build_sample_from_user_inputs(feature_names, defaults, cat_maps, dtypes, user_inputs)

            proba = model.predict_proba(sample)[0][1]
            threshold = 0.20
            pred = 1 if proba > threshold else 0

            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                if pred == 1:
                    st.markdown('<div class="warning-card" style="text-align:center;"><h2> FRAUD DETECTED</h2><p>Transaction Flagged</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-card" style="text-align:center;"><h2> LEGITIMATE</h2><p>Transaction Approved</p></div>', unsafe_allow_html=True)

            with col2:
                st.markdown(f'<div class="metric-card" style="text-align:center;"><h3> Fraud Probability</h3><h1 style="color:#6C63FF;margin:0;">{proba:.2%}</h1></div>', unsafe_allow_html=True)

            with col3:
                risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
                risk_color = "#FF6B6B" if risk_level == "High" else "#FFA726" if risk_level == "Medium" else "#4CAF50"
                st.markdown(f'<div class="metric-card" style="text-align:center;"><h3>⚡ Risk Level</h3><h1 style="color:{risk_color};margin:0;">{risk_level}</h1></div>', unsafe_allow_html=True)

            st.markdown("###  Risk Analysis")
            if proba > 0.7:
                st.warning("**High Risk:** Strong fraud signals detected. Immediate manual review recommended.")
            elif proba > 0.3:
                st.info("**Medium Risk:** Some suspicious patterns. Additional verification suggested.")
            else:
                st.success("**Low Risk:** Transaction appears normal. Standard processing recommended.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Check that the model is loaded and features match the training pipeline.")

elif current_page == "conclusions":
    st.markdown('<div class="main-header">🎯 Conclusions & Business Impact</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="success-card">
            <h3> Key Achievements</h3>
            <ul style="color:#555;">
                <li><strong>Two-version framework</strong> — showed missingness flags improve minority-class detection</li>
                <li><strong>SMOTE on training only</strong> — avoids data leakage into test set</li>
                <li><strong>Temporal validation</strong> — chronological 80/20 split + quartile drift analysis</li>
                <li><strong>Threshold tuning</strong> — τ=0.20 maximizes F1 and improves recall to 52.24%</li>
                <li><strong>Production-ready output</strong> — submission.csv with fraud probabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3> Limitations</h3>
            <ul style="color:#555;">
                <li>Test labels unavailable — real deployment accuracy unknown</li>
                <li>Temporal drift in Q3 shows model may degrade over time</li>
                <li>SMOTE generates synthetic points that may not reflect real fraud patterns</li>
                <li>Only LightGBM was extensively tuned due to compute constraints</li>
                <li>Threshold lowering increases false positives in production</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warning-card">
        <h3> Future Work</h3>
        <ul style="color:#555;">
            <li><strong>Autoencoders</strong> for unsupervised anomaly detection</li>
            <li><strong>Graph Neural Networks</strong> to detect organized fraud rings</li>
            <li><strong>Online learning</strong> for real-time adaptation to new fraud patterns</li>
            <li><strong>SHAP explainability</strong> for per-transaction risk reasoning</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------- END ----------
