"""
HDR Roland Dehghan Kamaragi
Outil de prédiction de ventilation — Badgir (prototype)
Lancer : streamlit run app/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Badgir — Prédiction CFD par IA", page_icon="🌀", layout="wide")

st.title("🌀 Badgir — Outil de prédiction IA")
st.markdown("**HDR Roland Dehghan Kamaragi** · EVCAU · ENSA Paris-Val de Seine")
st.divider()

# ── Données d'entraînement (CFD thèse) ────────────────────────────────────────
V_data = np.array([1, 3, 6, 10, 15, 20])
Q_data = np.array([0.030, 0.085, 0.180, 0.298, 0.447, 0.596])

scaler = StandardScaler()
X_train = scaler.fit_transform(V_data.reshape(-1, 1))
model = GradientBoostingRegressor(n_estimators=200, random_state=42)
model.fit(X_train, Q_data)

# ── Interface ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Paramètres")
    V_vent = st.slider("Vitesse du vent (m/s)", 0.5, 20.0, 5.0, 0.5)
    angle  = st.selectbox("Angle d'incidence (°)", [0, 15, 30, 45])
    st.caption("Maquette 1/50 · Soufflerie Ville-d'Avray · 4 conduits")
    
    Q_pred = model.predict(scaler.transform([[V_vent]]))[0]
    st.metric("Débit prédit Q (m³/s)", f"{Q_pred:.3f}", delta=f"α={angle}°")
    
    T_ext = st.slider("T extérieure (°C)", 25, 50, 38)
    T_int_est = T_ext - (Q_pred * 15)  # estimation simplifiée
    st.metric("T intérieure estimée (°C)", f"{max(T_int_est, 22):.1f}", delta=f"ΔT ≈ {T_ext - max(T_int_est, 22):.1f}°C")

with col2:
    st.subheader("📊 Courbe débit vs vitesse du vent")
    V_range = np.linspace(0.5, 20, 200)
    Q_range = model.predict(scaler.transform(V_range.reshape(-1, 1)))
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(V_range, Q_range, '#e07840', linewidth=2.5, label='Modèle surrogate (ML)')
    ax.scatter(V_data, Q_data, color='black', s=60, zorder=5, label='Points CFD Fluent')
    ax.axvline(V_vent, color='red', linestyle='--', alpha=0.7, label=f'V={V_vent} m/s')
    ax.axhline(Q_pred, color='blue', linestyle='--', alpha=0.7, label=f'Q={Q_pred:.3f} m³/s')
    ax.scatter([V_vent], [Q_pred], color='red', s=120, zorder=6)
    ax.set_xlabel('Vitesse du vent (m/s)', fontsize=12)
    ax.set_ylabel('Débit Q (m³/s)', fontsize=12)
    ax.set_title(f'Prédiction — Badgir 4 conduits, α={angle}°', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    st.pyplot(fig)

st.divider()
st.caption("Prototype HDR · Données : thèse 2014 ENSA Paris-Malaquais, soufflerie Ville-d'Avray, CFD Fluent k-ε")
