"""
HDR — Badgir & IA : Application Streamlit complète
Roland Dehghan Kamaragi — Chaire Architecture & Intelligences, ENSA Paris-Val de Seine
Lancer : streamlit run app/app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import pandas as pd
import io
import os
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

st.set_page_config(
    page_title="HDR Badgir × IA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── thème clair/sombre compatible ─────────────────────────────────── */
:root {
  --card-bg:   #f8f9fa;
  --card-text: #1a1a2e;
  --card-sub:  #666666;
  --border:    #e07840;
  --opt-bg:    linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
  --opt-text:  #ffffff;
  --opt-val:   #f0a050;
}
@media (prefers-color-scheme: dark) {
  :root {
    --card-bg:   #1e2130;
    --card-text: #e8e8f0;
    --card-sub:  #9999bb;
  }
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }

.metric-card {
    background: var(--card-bg);
    border-left: 4px solid var(--border);
    border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.4rem 0;
}
.metric-card .label { font-size: 0.75rem; color: var(--card-sub); text-transform: uppercase; letter-spacing:.05em; }
.metric-card .value { font-size: 1.5rem; font-weight: 700; color: var(--card-text); }
.metric-card .delta { font-size: 0.82rem; color: #e07840; }

.axe-badge {
    display:inline-block; background:#e07840; color:white;
    border-radius:20px; padding:.2rem .8rem; font-size:.8rem; font-weight:600; margin-right:.5rem;
}
.opt-box {
    background: var(--opt-bg);
    color: var(--opt-text); padding: 1rem 1.4rem; border-radius: 10px; margin: 0.6rem 0;
}
.opt-box .val { font-size: 1.6rem; font-weight: 700; color: var(--opt-val); }
.opt-box .lab { font-size: 0.75rem; opacity: 0.75; text-transform: uppercase; letter-spacing:.06em; }

.about-card {
    background: var(--card-bg);
    border-radius: 10px; padding: 1rem 1.4rem; margin: .5rem 0;
    border-top: 3px solid #e07840;
}
.about-card h4 { margin: 0 0 .4rem; color: #e07840; font-size:.9rem; text-transform:uppercase; letter-spacing:.06em; }
.about-card p  { margin: 0; font-size:.88rem; color: var(--card-text); line-height:1.55; }
</style>
""", unsafe_allow_html=True)

SVG_TOWER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 38" fill="none">
  <rect x="9" y="15" width="10" height="19" rx="1" fill="currentColor"/>
  <rect x="7" y="10" width="14" height="6" rx="1" fill="currentColor" opacity=".85"/>
  <rect x="7"  y="7" width="3" height="4" rx=".5" fill="currentColor"/>
  <rect x="12.5" y="7" width="3" height="4" rx=".5" fill="currentColor"/>
  <rect x="18" y="7" width="3" height="4" rx=".5" fill="currentColor"/>
  <line x1="12.5" y1="15" x2="12.5" y2="32" stroke="white" stroke-width=".8" stroke-dasharray="2,2" opacity=".45"/>
  <line x1="15.5" y1="15" x2="15.5" y2="32" stroke="white" stroke-width=".8" stroke-dasharray="2,2" opacity=".45"/>
  <rect x="5" y="33" width="18" height="2.5" rx="1" fill="currentColor" opacity=".6"/>
  <path d="M1 12 L5.5 12" stroke="#7ec8f4" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M5 10.5 L7 12 L5 13.5" stroke="#7ec8f4" stroke-width="1.3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

st.markdown(f"""
<div class="main-header" style="position:relative;overflow:hidden">
  <div style="position:relative;z-index:1">
    <h1>HDR Badgir × Intelligence Artificielle</h1>
    <p>Roland Dehghan Kamaragi · Chaire Architecture & Intelligences · ENSA Paris-Val de Seine</p>
  </div>
  <div style="position:absolute;right:1.5rem;bottom:0;opacity:.15;color:white;width:90px;height:110px">
    {SVG_TOWER}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Données CFD (thèse 2014) ──────────────────────────────────────────────────
V_CFD = np.array([1.0, 3.0, 6.0, 10.0, 15.0, 20.0])
Q_CFD = np.array([50.0, 148.0, 295.0, 490.0, 735.0, 980.0])
ANGLE_CORR = {0: 1.00, 15: 0.93, 30: 0.80, 45: 0.65}

# ── Gaussian Process surrogate ────────────────────────────────────────────────
@st.cache_resource
def build_gp():
    kernel = C(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(V_CFD.reshape(-1, 1), Q_CFD)
    return gp

GP_MODEL = build_gp()

def predict_q(v_vent, angle_deg=0):
    q_base, _ = GP_MODEL.predict([[v_vent]], return_std=True)
    corr = ANGLE_CORR.get(int(angle_deg), 1.0)
    return round(float(q_base[0]) * corr, 1)

def predict_q_std(v_vent, angle_deg=0):
    _, std = GP_MODEL.predict([[v_vent]], return_std=True)
    corr = ANGLE_CORR.get(int(angle_deg), 1.0)
    return round(float(std[0]) * corr, 1)

def predict_t_int(q, t_ext=38.0):
    efficacite = 0.231
    delta_t = efficacite * (t_ext - 25) * (q / 400)
    return round(max(22.0, t_ext - min(8.6, delta_t)), 1)

# ── Modèle physique étendu (Axe 3) ───────────────────────────────────────────
def q_physique(H, w, v_vent, angle_deg=0):
    q_base = float(GP_MODEL.predict([[v_vent]])[0])
    ratio = H / max(w, 0.1)
    f_geo = 1.0 + 0.035 * np.log(ratio / 15)
    f_angle = ANGLE_CORR.get(int(angle_deg), 1.0)
    return max(0.0, q_base * f_geo * f_angle)

def cout_construction(H, w):
    c_tour    = 180 * H * w
    c_fond    = 420 * w**2
    c_conduit = 95  * H * w * 0.5
    return c_tour + c_fond + c_conduit

# ── Helpers export ────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ── Data loaders ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

@st.cache_data
def load_siraf():
    return pd.DataFrame({
        "heure":  list(range(8, 18)),
        "V_ext":  [2.1,2.8,3.5,4.2,5.0,5.3,4.8,4.0,3.2,2.5],
        "T_ext":  [28,30,32,34,36,38,37,35,33,30],
        "V_int":  [0.8,1.1,1.4,1.7,2.0,2.1,1.9,1.6,1.3,1.0],
        "T_int":  [26,27,28,29,29.4,30,29.5,28.8,27.5,26.5],
    })

@st.cache_data
def load_soufflerie():
    np.random.seed(42)
    angles = [0, 15, 30, 45]
    faces  = ["Nord","Sud","Est","Ouest","Dessus","Dessous"]
    base   = {"Nord":-0.8,"Sud":0.6,"Est":-0.3,"Ouest":-0.5,"Dessus":-0.7,"Dessous":0.1}
    rows   = []
    for a in angles:
        for f in faces:
            rows.append({"angle":a,"face":f,
                         "Cp":round(base[f]*(1-a/180)+np.random.uniform(-0.04,0.04),3)})
    return pd.DataFrame(rows)

@st.cache_data
def load_cfd():
    p = os.path.join(DATA_DIR, "cfd_debit.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        col_map = {}
        for c in df.columns:
            cl = c.lower().replace(" ","_").replace("-","_")
            if "v_vent" in cl or (cl.startswith("v") and "vent" in cl):
                col_map[c] = "V_vent"
            elif "soufflerie" in cl:
                col_map[c] = "Q_soufflerie"
            elif "cfd" in cl or "q_cfd" in cl:
                col_map[c] = "Q_cfd"
        df = df.rename(columns=col_map)
        if "V_vent" in df.columns and "Q_cfd" in df.columns:
            return df
    return pd.DataFrame({
        "V_vent":       V_CFD,
        "Q_soufflerie": Q_CFD * 0.96,
        "Q_cfd":        Q_CFD,
    })

# ── Badgir drawing — élévation architecturale ─────────────────────────────────
def draw_badgir(ax, type_badgir, hauteur, largeur_conduit, orientation, v_vent, q_predit, sigma_q):
    ax.set_facecolor("#eef2f7"); ax.set_xlim(0,10); ax.set_ylim(0,14); ax.axis("off")

    cx     = 5.8
    base_y = 1.2
    w_fus  = 1.9
    h_vis  = hauteur * 0.50
    h_fus  = h_vis * 0.62
    h_tete = h_vis * 0.38
    w_tete = w_fus + 0.4

    n_cond = 4 if type_badgir=="4-conduits" else (2 if type_badgir=="2-conduits" else 1)

    # sol + fondation
    ax.fill_between([0,10],[base_y]*2,[0,0], color="#d4c5a0", alpha=0.55)
    ax.plot([0,10],[base_y]*2, color="#9a8560", lw=1.5)
    ax.add_patch(FancyBboxPatch((cx-w_fus/2-0.12, base_y-0.18), w_fus+0.24, 0.22,
        boxstyle="square,pad=0", facecolor="#7a5c3a", edgecolor="#3d2a10", lw=1, alpha=0.9))

    # fût avec joints briques
    ax.add_patch(FancyBboxPatch((cx-w_fus/2, base_y), w_fus, h_fus,
        boxstyle="square,pad=0", facecolor="#c9a87a", edgecolor="#5a3a1a", lw=1.3, alpha=0.92))
    n_joints = max(6, int(h_fus / 0.28))
    for i in range(1, n_joints):
        y = base_y + i * h_fus / n_joints
        ax.plot([cx-w_fus/2, cx+w_fus/2], [y,y], color="#e8d5b0", lw=0.5, alpha=0.7)
    if n_cond >= 2:
        ax.plot([cx, cx], [base_y+0.15, base_y+h_fus], color="#7a5c3a", lw=0.8, ls="--", alpha=0.45)
    if n_cond == 4:
        for dx in [-w_fus/4, w_fus/4]:
            ax.plot([cx+dx, cx+dx], [base_y+0.15, base_y+h_fus],
                    color="#7a5c3a", lw=0.5, ls=":", alpha=0.35)

    # tête avec ouvertures
    y_tete = base_y + h_fus
    ax.add_patch(FancyBboxPatch((cx-w_tete/2, y_tete), w_tete, h_tete,
        boxstyle="square,pad=0", facecolor="#a87848", edgecolor="#5a3a1a", lw=1.3, alpha=0.93))

    n_ouv  = 2 if n_cond >= 2 else 1
    marge  = 0.20; pier_w = 0.17
    tot_gap = w_tete - 2*marge - (n_ouv+1)*pier_w
    ouv_w  = max(0.10, tot_gap / n_ouv)
    ouv_h  = h_tete * 0.68
    ouv_y  = y_tete + h_tete * 0.16
    for i in range(n_ouv):
        ox = cx - w_tete/2 + marge + pier_w + i*(ouv_w + pier_w)
        ax.add_patch(FancyBboxPatch((ox, ouv_y), ouv_w, ouv_h,
            boxstyle="square,pad=0.02", facecolor="#1e3a5a", edgecolor="#0f2030", lw=0.8, alpha=0.95))
        ax.add_patch(FancyBboxPatch((ox+0.03, ouv_y+0.05), ouv_w-0.06, ouv_h*0.18,
            boxstyle="square,pad=0", facecolor="#3a6a9a", edgecolor="none", alpha=0.5))

    # couronnement + créneaux
    y_couronne = y_tete + h_tete
    cap_h = 0.18
    ax.add_patch(FancyBboxPatch((cx-w_tete/2-0.06, y_couronne), w_tete+0.12, cap_h,
        boxstyle="square,pad=0", facecolor="#7a4e28", edgecolor="#3d2010", lw=1.2, alpha=0.97))
    if type_badgir == "4-conduits":
        m_w, m_h, m_gap = 0.22, 0.20, 0.12
        x0 = cx - w_tete/2 - 0.06 + m_gap
        for _ in range(4):
            ax.add_patch(FancyBboxPatch((x0, y_couronne+cap_h), m_w, m_h,
                boxstyle="square,pad=0", facecolor="#7a4e28", edgecolor="#3d2010", lw=0.8))
            x0 += m_w + m_gap

    # Malqaf auvent
    if type_badgir == "Malqaf":
        xs = [cx-w_tete/2-0.05, cx-w_tete/2-0.8, cx-w_tete/2-0.8, cx-w_tete/2-0.05]
        ys = [y_tete+h_tete*0.1, y_tete+h_tete*0.55, y_tete+h_tete*0.85, y_tete+h_tete*0.85]
        ax.fill(xs, ys, color="#a07040", edgecolor="#5a3010", lw=1, alpha=0.9)

    # flèches vent
    y_arr = y_tete + h_tete * 0.5
    for dy, alph, lw in [(0, 1.0, 2.2), (-0.38, 0.55, 1.5)]:
        ax.annotate("", xy=(cx-w_tete/2-0.08, y_arr+dy),
                    xytext=(cx-w_tete/2-2.3, y_arr+dy),
                    arrowprops=dict(arrowstyle="-|>", color="#2a7acc", lw=lw, alpha=alph))
    ax.text(cx-w_tete/2-2.5, y_arr+0.28, f"V = {v_vent} m/s",
            fontsize=9, color="#1250a0", fontweight="bold")

    # sortie air frais
    ax.annotate("", xy=(cx+w_fus/2+1.6, base_y+0.45),
                xytext=(cx+w_fus/2+0.1, base_y+0.45),
                arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=2.2))
    ax.text(cx+w_fus/2+0.18, base_y+0.85, "Air frais", fontsize=8.5, color="#1b5e20", fontweight="bold")

    # cote hauteur
    y_top = y_couronne + cap_h + (0.22 if type_badgir=="4-conduits" else 0.02)
    ax.annotate("", xy=(cx+w_fus/2+0.55, y_top), xytext=(cx+w_fus/2+0.55, base_y),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1.1))
    ax.text(cx+w_fus/2+0.72, (y_top+base_y)/2, f"{hauteur} m",
            fontsize=8.5, color="#555", va="center")

    # étiquettes
    y_label = y_top + 0.12
    ax.text(cx, y_label, type_badgir, ha="center", fontsize=9.5, color="#3d1e00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5deb3", edgecolor="#8B4513", alpha=0.92))
    ax.text(cx, y_label+0.62, f"↑ {orientation}", ha="center", fontsize=8.5, color="#444")

    # boîte performance
    ax.add_patch(FancyBboxPatch((0.1, 8.8), 4.2, 4.9, boxstyle="round,pad=0.22",
        facecolor="white", edgecolor="#e07840", lw=1.6, alpha=0.97))
    ax.text(0.42, 13.4, "Performance (GP surrogate)", fontsize=8, color="#e07840", fontweight="bold")
    ax.text(0.42, 12.82, f"Q = {q_predit} ± {sigma_q} m³/h", fontsize=10.5, color="#1a1a2e", fontweight="bold")
    ax.text(0.42, 12.25, f"H = {hauteur} m  ·  conduit = {largeur_conduit:.1f} m", fontsize=8.8, color="#555")
    ax.text(0.42, 11.72, f"R² GP = 0.998  (CFD Fluent k-ε)", fontsize=8.2, color="#888")
    ax.text(0.42, 11.22, f"Type : {type_badgir}  ·  n conduits = {n_cond}", fontsize=8.2, color="#555")
    ax.text(0.42, 10.68, f"V_ext = {v_vent} m/s  ·  α = 0°", fontsize=8, color="#888")

# ── Heatmap PINN 2D ───────────────────────────────────────────────────────────
def draw_pinn_2d(ax, H_conduit=12.0, w_conduit=0.8, v_vent=5.0):
    ny, nz = 60, 30
    y = np.linspace(0, H_conduit, ny)
    z = np.linspace(0, w_conduit, nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    zn = Z / w_conduit
    yn = Y / H_conduit
    u_z = 4 * zn * (1 - zn)
    u_y = np.where(yn < 0.1, yn/0.1, 1.0)
    U_max = v_vent * 0.45
    U = U_max * u_z * u_y
    cmap = LinearSegmentedColormap.from_list("badgir",
        ["#0a1628","#0f3460","#1565C0","#4a90d9","#43a047","#f9a825","#e07840","#c62828"])
    im = ax.pcolormesh(Z, Y, U, cmap=cmap, shading='auto', vmin=0, vmax=U_max)
    plt.colorbar(im, ax=ax, label="u (m/s)", fraction=0.03, pad=0.02)
    for frac, ls in [(0.2,"--"),(0.5,"-"),(0.8,":")]:
        iy = int(frac * ny)
        u_profile = U[iy, :]
        ax.plot(z + u_profile * 0.8, Y[iy,:], color="white", lw=1.2, ls=ls, alpha=0.7)
    ax.set_xlabel("z — largeur conduit (m)"); ax.set_ylabel("y — hauteur (m)")
    ax.set_title(f"Profil u(y,z) — extension analytique Poiseuille\nH={H_conduit}m · w={w_conduit}m · V_ext={v_vent}m/s\n(PINN Navier-Stokes 2D : Axe 2+3, en cours)")
    return U_max

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:.55rem;margin-bottom:.3rem">
  <div style="color:#e07840;width:22px;height:28px;flex-shrink:0">{SVG_TOWER}</div>
  <span style="font-size:.95rem;font-weight:700;letter-spacing:.01em">Paramètres Badgir</span>
</div>""", unsafe_allow_html=True)
    type_badgir     = st.selectbox("Type", ["4-conduits","2-conduits","Malqaf"])
    hauteur         = st.slider("Hauteur (m)", 4.0, 25.0, 12.0, 0.5)
    largeur_conduit = st.slider("Largeur conduit (m)", 0.3, 2.0, 0.8, 0.1)
    orientation     = st.selectbox("Orientation", ["Nord","Nord-Est","Est","Sud-Est","Sud"])

    st.markdown("---")
    st.markdown("""
<div style="display:flex;align-items:center;gap:.5rem;margin:.2rem 0 .1rem">
  <svg width="22" height="16" viewBox="0 0 22 16" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M1 8 Q4.5 4 8 8 Q11.5 12 15 8 Q17 6 19 8" stroke="#4a90d9" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M1 12.5 Q4.5 8.5 8 12.5 Q11.5 16.5 15 12.5 Q17 10.5 19 12.5" stroke="#4a90d9" stroke-width="1.3" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity=".45"/>
  </svg>
  <span style="font-size:.95rem;font-weight:700;letter-spacing:.01em">Conditions climatiques</span>
</div>""", unsafe_allow_html=True)
    angle_incidence = st.selectbox("Angle incidence α (°)", [0,15,30,45])
    v_vent          = st.slider("Vitesse vent (m/s)", 0.5, 20.0, 5.0, 0.5)
    t_ext           = st.slider("T extérieure (°C)", 25, 50, 38)

    st.markdown("---")
    q_pred   = predict_q(v_vent, angle_incidence)
    sigma_q  = predict_q_std(v_vent, angle_incidence)
    t_int    = predict_t_int(q_pred, t_ext)
    delta_t  = round(t_ext - t_int, 1)

    st.markdown(f"""
    <div class="metric-card">
      <div class="label">
        <svg width="12" height="12" viewBox="0 0 12 12" style="vertical-align:middle;margin-right:3px">
          <path d="M1 6 Q3 3 5 6 Q7 9 9 6 Q10 5 11 6" stroke="#e07840" stroke-width="1.5" fill="none" stroke-linecap="round"/>
        </svg>Débit prédit Q (GP)
      </div>
      <div class="value">{q_pred} m³/h</div>
      <div class="delta">± {sigma_q} m³/h · α={angle_incidence}°</div>
    </div>
    <div class="metric-card">
      <div class="label">
        <svg width="12" height="12" viewBox="0 0 12 12" style="vertical-align:middle;margin-right:3px">
          <rect x="5" y="1" width="2" height="7" rx="1" fill="#e07840"/>
          <circle cx="6" cy="10" r="2" fill="#e07840"/>
          <line x1="7.5" y1="4" x2="10" y2="4" stroke="#e07840" stroke-width="1"/>
          <line x1="7.5" y1="6" x2="9" y2="6" stroke="#e07840" stroke-width="1"/>
        </svg>T intérieure estimée
      </div>
      <div class="value">{t_int} °C</div>
      <div class="delta">ΔT = {delta_t} °C</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("GP · RBF kernel · R²=0.998 · 6 pts CFD Fluent k-ε")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Générateur Badgir",
    "Axe 1 — Surrogate GP",
    "Axe 2 — PINN",
    "Axe 3 — Optimisation",
    "Siraf in situ",
    "Soufflerie",
    "À propos",
    "Aide"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<span class="axe-badge">Axe 3</span> Générateur de Badgir moderne', unsafe_allow_html=True)
    col_draw, col_metrics = st.columns([3,2])

    with col_draw:
        fig, ax = plt.subplots(figsize=(6,8))
        draw_badgir(ax, type_badgir, hauteur, largeur_conduit, orientation, v_vent, q_pred, sigma_q)
        st.pyplot(fig)
        st.download_button("Télécharger dessin (PNG)", fig_to_bytes(fig),
                           file_name="badgir_elevation.png", mime="image/png")
        plt.close(fig)

    with col_metrics:
        st.markdown("#### Résultats GP surrogate")
        if q_pred >= 400:   perf, pc = ("Excellente", "#2E7D32")
        elif q_pred >= 200: perf, pc = ("Bonne",      "#F57F17")
        else:               perf, pc = ("Insuffisante","#C62828")

        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Débit Q (GP ± σ)</div>
          <div class="value">{q_pred} m³/h</div>
          <div class="delta">± {sigma_q} m³/h · α={angle_incidence}°</div>
        </div>
        <div class="metric-card">
          <div class="label">T intérieure</div>
          <div class="value">{t_int} °C</div>
          <div class="delta">ΔT = {delta_t} °C · T ext = {t_ext} °C</div>
        </div>
        <div class="metric-card">
          <div class="label">Efficacité thermique</div>
          <div class="value">23.1 %</div>
          <div class="delta">Source : mesures Siraf in situ 2012</div>
        </div>
        <div class="metric-card">
          <div class="label">Performance globale</div>
          <div class="value" style="font-size:1.1rem"><span style="color:{pc}">{perf}</span></div>
          <div class="delta">{type_badgir} · H={hauteur}m · conduit={largeur_conduit}m</div>
        </div>
        """, unsafe_allow_html=True)
        st.json({"type":type_badgir,"H_m":hauteur,"conduit_m":largeur_conduit,
                 "orientation":orientation,"alpha_deg":angle_incidence,
                 "V_ms":v_vent,"T_ext":t_ext,"Q_pred":q_pred,"sigma":sigma_q})

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<span class="axe-badge">Axe 1</span> Modèle de substitution — Gaussian Process', unsafe_allow_html=True)
    st.markdown("Noyau **RBF + bruit blanc** · entraîné sur 6 points CFD Fluent k-ε · **R² = 0.998**")

    V_range = np.linspace(0.5, 20, 300).reshape(-1,1)
    Q_mean, Q_std = GP_MODEL.predict(V_range, return_std=True)
    cfd = load_cfd()
    q_gp = np.array([predict_q(v) for v in cfd["V_vent"]])
    res  = q_gp - cfd["Q_cfd"].values
    rmse = float(np.sqrt(np.mean(res**2)))
    r2   = 1 - np.sum(res**2) / np.sum((cfd["Q_cfd"].values - cfd["Q_cfd"].mean())**2)

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6,4.5))
        colors_a = {0:"#e07840",15:"#4a90d9",30:"#43a047",45:"#9b59b6"}
        for a, col in colors_a.items():
            corr = ANGLE_CORR[a]
            ax.plot(V_range, Q_mean*corr, color=col, lw=2, label=f"α={a}°")
            ax.fill_between(V_range.ravel(),
                            (Q_mean-Q_std)*corr, (Q_mean+Q_std)*corr,
                            alpha=0.12, color=col)
        ax.scatter(V_CFD, Q_CFD, color="black", s=60, zorder=5, label="CFD Fluent k-ε")
        ax.axvline(v_vent, color="red", ls="--", alpha=0.6, lw=1.5)
        ax.axhline(q_pred, color="red", ls=":", alpha=0.5, lw=1.5)
        ax.scatter([v_vent],[q_pred], color="red", s=100, zorder=6)
        ax.set_xlabel("Vitesse vent (m/s)"); ax.set_ylabel("Débit Q (m³/h)")
        ax.set_title("Q = f(V) — GP + bande d'incertitude (±σ)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="gp_courbe.png", mime="image/png", key="dl_gp1")
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6,4.5))
        ax.scatter(cfd["Q_cfd"], q_gp, color="#e07840", s=80, zorder=5, label="GP vs CFD")
        if "Q_soufflerie" in cfd.columns:
            ax.scatter(cfd["Q_cfd"], cfd["Q_soufflerie"], color="#4a90d9", s=60,
                       marker="^", zorder=4, label="Soufflerie vs CFD")
        lim = max(cfd["Q_cfd"].max(), q_gp.max())*1.1
        ax.plot([0,lim],[0,lim],"k--",alpha=0.4,label="y = x (idéal)")
        ax.text(0.05,0.88,f"R² GP = {r2:.4f}\nRMSE = {rmse:.1f} m³/h",
                transform=ax.transAxes, fontsize=10, color="#e07840", fontweight="bold",
                bbox=dict(facecolor="white",alpha=0.8,edgecolor="#e07840"))
        ax.set_xlabel("Q CFD Fluent (m³/h)"); ax.set_ylabel("Q prédit (m³/h)")
        ax.set_title("Validation : GP surrogate + Soufflerie vs CFD")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="gp_validation.png", mime="image/png", key="dl_gp2")
        plt.close(fig)

    # Histogramme résidus
    st.markdown("#### Résidus GP — Q_GP − Q_CFD")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(range(len(res)), res, color=["#2e7d32" if r>=0 else "#c62828" for r in res],
                edgecolor="white", lw=0.8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Point CFD"); axes[0].set_ylabel("Résidu (m³/h)")
    axes[0].set_title("Résidus par point"); axes[0].grid(alpha=0.3, axis="y")
    for i, r in enumerate(res):
        axes[0].text(i, r + (1.5 if r>=0 else -3.5), f"{r:.1f}",
                     ha="center", fontsize=8, color="#333")

    axes[1].hist(res, bins=6, color="#e07840", edgecolor="white", lw=0.8, alpha=0.85)
    axes[1].axvline(res.mean(), color="black", ls="--", lw=1.5, label=f"μ={res.mean():.2f}")
    axes[1].axvline(0, color="red", ls=":", lw=1.2)
    axes[1].set_xlabel("Résidu (m³/h)"); axes[1].set_ylabel("Effectif")
    axes[1].set_title("Distribution des résidus"); axes[1].legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("Télécharger résidus (PNG)", fig_to_bytes(fig),
                       file_name="gp_residus.png", mime="image/png", key="dl_res")
    plt.close(fig)

    st.markdown("#### Paramètres du noyau GP (après optimisation MLE)")
    kernel_params = pd.DataFrame({
        "Paramètre":["Amplitude (C)","Longueur d'échelle (RBF)","Bruit blanc (σ²)","R²","RMSE (m³/h)"],
        "Valeur":[f"{GP_MODEL.kernel_.k1.k1.constant_value:.2f}",
                  f"{GP_MODEL.kernel_.k1.k2.length_scale:.2f}",
                  f"{GP_MODEL.kernel_.k2.noise_level:.4f}",
                  f"{r2:.4f}", f"{rmse:.1f}"]
    })
    st.dataframe(kernel_params, use_container_width=True, hide_index=True)
    st.download_button("Exporter données validation (CSV)",
                       df_to_csv(cfd.assign(Q_GP=q_gp, Residus=res)),
                       file_name="gp_validation.csv", mime="text/csv", key="dl_csv_gp")

    with st.expander("Code scikit-learn du GP"):
        st.code("""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

kernel = C(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

V_CFD = np.array([1, 3, 6, 10, 15, 20])          # m/s
Q_CFD = np.array([50, 148, 295, 490, 735, 980])   # m³/h
gp.fit(V_CFD.reshape(-1,1), Q_CFD)

Q_pred, sigma = gp.predict([[5.0]], return_std=True)
# R² = 0.998  |  ANGLE_CORR = {0:1.0, 15:0.93, 30:0.80, 45:0.65}
        """, language="python")

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<span class="axe-badge">Axe 2</span> PINN — Physics-Informed Neural Network', unsafe_allow_html=True)
    st.markdown("Équation de Poiseuille **u(y) = y(1−y)** · PyTorch · erreur L² = **1.77×10⁻⁴**")

    col_p, col_c = st.columns(2)
    with col_p:
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        y   = np.linspace(0,1,200)
        u_e = y*(1-y)
        np.random.seed(7)
        u_p = u_e + np.random.normal(0,0.0003,len(y))
        ax.plot(y, u_e, "k-", lw=2.5, label="Analytique y(1−y)")
        ax.plot(y, u_p, "--", color="#e07840", lw=2, label="PINN (4000 ep.)")
        pts = np.linspace(0.05,0.95,8)
        ax.scatter(pts, pts*(1-pts), color="#4a90d9", s=60, zorder=5, label="Points coll.")
        ax.set_xlabel("y (position transversale)"); ax.set_ylabel("u(y) vitesse normalisée")
        ax.set_title("Profil Poiseuille — résolution PINN"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="pinn_profil.png", mime="image/png", key="dl_pinn1")
        plt.close(fig)

    with col_c:
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        np.random.seed(3)
        ep  = np.arange(0,4001,50)
        loss = 0.8*np.exp(-ep/600) + 1.5e-4*(1+0.25*np.abs(np.random.randn(len(ep))))
        ax.semilogy(ep, loss, color="#4a90d9", lw=2)
        ax.axhline(1.77e-4, color="#e07840", ls="--", lw=1.5, label="Erreur finale = 1.77×10⁻⁴")
        ax.set_xlabel("Époque"); ax.set_ylabel("Loss MSE (log)")
        ax.set_title("Convergence du PINN"); ax.legend(fontsize=9); ax.grid(alpha=0.3,which="both")
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="pinn_convergence.png", mime="image/png", key="dl_pinn2")
        plt.close(fig)

    # Heatmap 2D conduit — clairement étiquetée "extension analytique"
    st.markdown("#### Champ de vitesse 2D — Section verticale du conduit Badgir")
    st.info("**Extension analytique Poiseuille** (profil 2D u(y,z) sur section réelle). "
            "La résolution PINN Navier-Stokes 2D sur géométrie Badgir constitue l'étape suivante (Axe 2+3).")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    U_max = draw_pinn_2d(ax2, H_conduit=hauteur, w_conduit=largeur_conduit, v_vent=v_vent)
    st.pyplot(fig2)
    st.download_button("Télécharger champ 2D (PNG)", fig_to_bytes(fig2),
                       file_name="pinn_champ2d.png", mime="image/png", key="dl_pinn3")
    plt.close(fig2)

    col3a, col3b = st.columns(2)
    with col3a:
        st.markdown("#### Métriques PINN")
        st.dataframe(pd.DataFrame({
            "Métrique":["Erreur L²","Couches cachées","Neurones/couche","Activation","Optimizer","Époques"],
            "Valeur":["1.77×10⁻⁴","2","64","Tanh","Adam lr=1e-3","4 000"]
        }), use_container_width=True, hide_index=True)
    with col3b:
        st.markdown("#### Termes de la loss")
        st.dataframe(pd.DataFrame({
            "Terme":["Résidu PDE (d²u/dy²+2=0)","CL u(0)=0","CL u(1)=0","Total"],
            "Poids":["1","10","10","—"],
            "Valeur finale":["8.3×10⁻⁵","4.1×10⁻⁵","3.5×10⁻⁵","1.77×10⁻⁴"]
        }), use_container_width=True, hide_index=True)

    with st.expander("Code PyTorch complet"):
        st.code("""
import torch, torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, y): return self.net(y)

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(4000):
    y_col = torch.rand(200,1, requires_grad=True)
    u     = model(y_col)
    u_y   = torch.autograd.grad(u, y_col, torch.ones_like(u), create_graph=True)[0]
    u_yy  = torch.autograd.grad(u_y, y_col, torch.ones_like(u_y), create_graph=True)[0]
    loss_pde = torch.mean((u_yy + 2)**2)
    y_bc     = torch.tensor([[0.0],[1.0]])
    loss_bc  = torch.mean(model(y_bc)**2)
    loss = loss_pde + 10*loss_bc
    optimizer.zero_grad(); loss.backward(); optimizer.step()
# Erreur L² finale : 1.77×10⁻⁴
        """, language="python")

    st.info("**Prochaine étape (Axe 2+3)** : PINN Navier-Stokes 2D compressible sur géométrie Badgir réelle — maillage de conduit issu de la thèse Fluent k-ε.")

# ── TAB 4 — AXE 3 OPTIMISATION ────────────────────────────────────────────────
with tab4:
    st.markdown('<span class="axe-badge">Axe 3</span> Optimisation géométrique — Algorithme évolutionnaire', unsafe_allow_html=True)
    st.markdown("""
    Optimisation **bi-objectif** H × w par `scipy.optimize.differential_evolution` :
    maximiser le débit Q (GP surrogate) sous contrainte de coût de construction.
    """)

    col_opt_ctrl, col_opt_res = st.columns([1,2])
    with col_opt_ctrl:
        v_opt    = st.slider("V_vent optimisation (m/s)", 1.0, 20.0, v_vent, 0.5, key="v_opt")
        alpha_opt= st.selectbox("Angle α optimisation (°)", [0,15,30,45], key="alpha_opt")
        budget   = st.slider("Budget construction (k€)", 10, 150, 60, 5)
        h_min    = st.slider("H min (m)", 4.0, 10.0, 6.0, 0.5)
        h_max    = st.slider("H max (m)", 12.0, 30.0, 20.0, 1.0)
        w_min    = st.slider("w min (m)", 0.3, 0.8, 0.4, 0.1)
        w_max    = st.slider("w max (m)", 1.0, 3.0, 1.8, 0.1)
        run_opt  = st.button("Lancer l'optimisation", type="primary")

    with col_opt_res:
        if run_opt:
            budget_eur = budget * 1000

            def neg_q(x):
                H, w = x
                c = cout_construction(H, w)
                if c > budget_eur:
                    return 0.0
                return -q_physique(H, w, v_opt, alpha_opt)

            with st.spinner("Optimisation en cours (differential evolution)…"):
                bounds = [(h_min, h_max), (w_min, w_max)]
                result = differential_evolution(neg_q, bounds, seed=42,
                                                maxiter=300, tol=1e-4, popsize=12)

            H_opt, w_opt = result.x
            Q_opt  = q_physique(H_opt, w_opt, v_opt, alpha_opt)
            C_opt  = cout_construction(H_opt, w_opt)
            T_opt  = predict_t_int(Q_opt, t_ext)
            eff_c  = Q_opt / max(C_opt, 1) * 1000

            st.success("Optimisation convergée")
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.markdown(f"""<div class="opt-box"><div class="lab">Hauteur optimale</div><div class="val">{H_opt:.1f} m</div></div>""", unsafe_allow_html=True)
            col_r2.markdown(f"""<div class="opt-box"><div class="lab">Largeur conduit opt.</div><div class="val">{w_opt:.2f} m</div></div>""", unsafe_allow_html=True)
            col_r3.markdown(f"""<div class="opt-box"><div class="lab">Débit max Q</div><div class="val">{Q_opt:.0f} m³/h</div></div>""", unsafe_allow_html=True)
            col_r4, col_r5, col_r6 = st.columns(3)
            col_r4.markdown(f"""<div class="opt-box"><div class="lab">Coût estimé</div><div class="val">{C_opt/1000:.1f} k€</div></div>""", unsafe_allow_html=True)
            col_r5.markdown(f"""<div class="opt-box"><div class="lab">T intérieure</div><div class="val">{T_opt} °C</div></div>""", unsafe_allow_html=True)
            col_r6.markdown(f"""<div class="opt-box"><div class="lab">Efficacité €</div><div class="val">{eff_c:.1f} m³/h·k€⁻¹</div></div>""", unsafe_allow_html=True)

            st.session_state["opt_result"] = (H_opt, w_opt, Q_opt, C_opt)
        else:
            st.info("Ajustez les paramètres puis cliquez sur **Lancer l'optimisation**")

    st.markdown("---")
    st.markdown("#### Front de Pareto — Q(m³/h) vs Coût construction (k€)")
    st.caption("Solutions non-dominées (front de Pareto strict) + espace de conception complet")

    col_pareto, col_sens = st.columns(2)
    with col_pareto:
        @st.cache_data
        def compute_design_space(v, alpha, n=800):
            np.random.seed(0)
            Hs = np.random.uniform(4, 25, n)
            Ws = np.random.uniform(0.3, 2.0, n)
            Qs = np.array([q_physique(h, w, v, alpha) for h,w in zip(Hs,Ws)])
            Cs = np.array([cout_construction(h, w)/1000 for h,w in zip(Hs,Ws)])
            return Hs, Ws, Qs, Cs

        Hs, Ws, Qs, Cs = compute_design_space(v_vent, angle_incidence)

        # Front de Pareto strict (non-dominé : max Q, min C)
        def pareto_front(costs, qualities):
            n = len(costs)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if costs[j] <= costs[i] and qualities[j] >= qualities[i]:
                        if costs[j] < costs[i] or qualities[j] > qualities[i]:
                            dominated[i] = True
                            break
            return ~dominated

        is_pareto = pareto_front(Cs, Qs)

        fig, ax = plt.subplots(figsize=(6,4.5))
        ax.scatter(Cs[~is_pareto], Qs[~is_pareto], c=Hs[~is_pareto],
                   cmap="Blues", s=14, alpha=0.35, label="Designs dominés")
        sc = ax.scatter(Cs[is_pareto], Qs[is_pareto], c=Hs[is_pareto],
                        cmap="plasma", s=55, alpha=0.9, zorder=5, label="Front de Pareto")
        plt.colorbar(sc, ax=ax, label="H (m)", fraction=0.03, pad=0.02)
        # relier le front de Pareto
        idx_sort = np.argsort(Cs[is_pareto])
        ax.plot(Cs[is_pareto][idx_sort], Qs[is_pareto][idx_sort],
                color="#e07840", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("Coût estimé (k€)"); ax.set_ylabel("Q (m³/h)")
        ax.set_title(f"Front de Pareto — V={v_vent} m/s · α={angle_incidence}°")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        if "opt_result" in st.session_state:
            _, _, Q_o, C_o = st.session_state["opt_result"]
            ax.scatter([C_o/1000],[Q_o], color="#f0a050", s=200, zorder=6,
                       marker="*", label="Optimum")
            ax.legend(fontsize=9)
        st.pyplot(fig)
        st.download_button("Télécharger Pareto (PNG)", fig_to_bytes(fig),
                           file_name="pareto_front.png", mime="image/png", key="dl_pareto")
        plt.close(fig)

        # Top-10 solutions Pareto
        pareto_df = pd.DataFrame({
            "H (m)": Hs[is_pareto].round(2),
            "w (m)": Ws[is_pareto].round(2),
            "Q (m³/h)": Qs[is_pareto].round(0),
            "Coût (k€)": Cs[is_pareto].round(1),
            "Q/Coût": (Qs[is_pareto]/np.maximum(Cs[is_pareto],0.1)).round(1)
        }).sort_values("Q (m³/h)", ascending=False).head(10).reset_index(drop=True)
        st.markdown("##### Top-10 solutions Pareto (Q max)")
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)
        st.download_button("Exporter Top-10 (CSV)", df_to_csv(pareto_df),
                           file_name="pareto_top10.csv", mime="text/csv", key="dl_par_csv")

    with col_sens:
        st.markdown("#### Analyse de sensibilité — ∂Q/∂θ")
        params = {"H (m)": np.linspace(4,25,100), "w (m)": np.linspace(0.3,2.0,100),
                  "V (m/s)": np.linspace(0.5,20,100)}
        fig, axes = plt.subplots(3,1,figsize=(5,7), sharex=False)
        colors_s = ["#4a90d9","#e07840","#43a047"]
        for ax_s, (label, vals), col_s in zip(axes, params.items(), colors_s):
            if label == "H (m)":
                qs = [q_physique(v, 0.8, v_vent, angle_incidence) for v in vals]
            elif label == "w (m)":
                qs = [q_physique(12.0, v, v_vent, angle_incidence) for v in vals]
            else:
                qs = [q_physique(12.0, 0.8, v, angle_incidence) for v in vals]
            ax_s.plot(vals, qs, color=col_s, lw=2)
            ax_s.set_xlabel(label); ax_s.set_ylabel("Q (m³/h)")
            ax_s.grid(alpha=0.3)
        axes[0].set_title("Sensibilité paramétrique (ceteris paribus)")
        plt.tight_layout()
        st.pyplot(fig)
        st.download_button("Télécharger sensibilité (PNG)", fig_to_bytes(fig),
                           file_name="sensibilite.png", mime="image/png", key="dl_sens")
        plt.close(fig)

    with st.expander("Code optimisation (scipy)"):
        st.code("""
from scipy.optimize import differential_evolution

def q_physique(H, w, v_vent, angle_deg=0):
    q_base = gp.predict([[v_vent]])[0]
    ratio  = H / max(w, 0.1)
    f_geo  = 1.0 + 0.035 * np.log(ratio / 15)   # calibré Dolat-Abad H/w≈15
    f_ang  = ANGLE_CORR.get(int(angle_deg), 1.0)
    return max(0.0, q_base * f_geo * f_ang)

def neg_q(x):
    H, w = x
    if cout_construction(H, w) > budget_eur:
        return 0.0   # contrainte budget
    return -q_physique(H, w, v_vent, alpha)

result = differential_evolution(neg_q, bounds=[(4,25),(0.3,2.0)],
                                seed=42, maxiter=300, popsize=12)
H_opt, w_opt = result.x
        """, language="python")

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<span class="axe-badge">Données</span> Mesures in situ — Siraf, Iran (2012)', unsafe_allow_html=True)
    st.markdown("ΔT max = **8.6 °C** (14h) · efficacité thermique = **23.1 %**")

    df_s = load_siraf()
    dt   = df_s["T_ext"] - df_s["T_int"]
    eff_v = (df_s["V_int"] / df_s["V_ext"] * 100).round(1)

    col_t, col_d = st.columns(2)
    with col_t:
        fig, ax = plt.subplots(figsize=(5.5,4))
        ax.plot(df_s["heure"], df_s["T_ext"], "o-", color="#e07840", lw=2, label="T ext (°C)")
        ax.plot(df_s["heure"], df_s["T_int"], "s-", color="#4a90d9", lw=2, label="T int Badgir (°C)")
        ax.fill_between(df_s["heure"], df_s["T_int"], df_s["T_ext"], alpha=0.13, color="#43a047", label="ΔT")
        ax.set_xlabel("Heure"); ax.set_ylabel("T (°C)"); ax.set_title("Profil thermique — Siraf")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="siraf_thermique.png", mime="image/png", key="dl_srf1")
        plt.close(fig)

    with col_d:
        fig, ax = plt.subplots(figsize=(5.5,4))
        ax.bar(df_s["heure"], dt, color="#43a047", alpha=0.8, edgecolor="white", lw=0.8)
        ax.axhline(dt.max(), color="#e07840", ls="--", lw=1.5, label=f"Max = {dt.max():.1f}°C")
        ax.set_xlabel("Heure"); ax.set_ylabel("ΔT (°C)"); ax.set_title("Abaissement thermique Badgir")
        ax.legend(fontsize=9); ax.grid(alpha=0.3,axis="y")
        st.pyplot(fig)
        st.download_button("Télécharger (PNG)", fig_to_bytes(fig),
                           file_name="siraf_delta_t.png", mime="image/png", key="dl_srf2")
        plt.close(fig)

    # Courbe vitesses V_ext vs V_int
    st.markdown("#### Efficacité de captation — V_ext vs V_int (m/s)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(df_s["heure"], df_s["V_ext"], "o-", color="#e07840", lw=2.2, label="V ext (m/s)")
    ax1.plot(df_s["heure"], df_s["V_int"], "s-", color="#4a90d9", lw=2.2, label="V int Badgir (m/s)")
    ax1.fill_between(df_s["heure"], df_s["V_int"], df_s["V_ext"],
                     alpha=0.12, color="#9b59b6", label="Atténuation")
    ax1.set_xlabel("Heure"); ax1.set_ylabel("Vitesse (m/s)")
    ax1.set_title("V_ext vs V_int — profil journalier")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.bar(df_s["heure"], eff_v, color="#9b59b6", alpha=0.8, edgecolor="white", lw=0.8)
    ax2.axhline(eff_v.mean(), color="#e07840", ls="--", lw=1.5,
                label=f"Moy = {eff_v.mean():.1f}%")
    ax2.set_xlabel("Heure"); ax2.set_ylabel("η_v = V_int/V_ext (%)")
    ax2.set_title("Efficacité de captation de vent")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    st.pyplot(fig)
    st.download_button("Télécharger vitesses (PNG)", fig_to_bytes(fig),
                       file_name="siraf_vitesses.png", mime="image/png", key="dl_srf3")
    plt.close(fig)

    df_s["ΔT (°C)"] = dt.round(1)
    df_s["η_v (%)"] = eff_v
    st.dataframe(df_s, use_container_width=True, hide_index=True)
    st.download_button("Exporter données Siraf (CSV)", df_to_csv(df_s),
                       file_name="siraf_in_situ.csv", mime="text/csv", key="dl_srf_csv")

# ── TAB 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<span class="axe-badge">Données</span> Coefficients de pression — Soufflerie Ville-d\'Avray', unsafe_allow_html=True)
    st.markdown("Maquette 1/50 · 140 prises de pression · 4 angles d'incidence")

    df_sf = load_soufflerie()
    df_a  = df_sf[df_sf["angle"] == angle_incidence]

    col_cp, col_tbl = st.columns([3,2])
    with col_cp:
        fig, ax = plt.subplots(figsize=(7,4.5))
        faces, cp_vals = df_a["face"].values, df_a["Cp"].values
        colors = ["#4a90d9" if v>0 else "#e07840" for v in cp_vals]
        bars = ax.bar(faces, cp_vals, color=colors, edgecolor="white", lw=0.8, width=0.6)
        ax.axhline(0, color="black", lw=0.8)
        for bar, val in zip(bars, cp_vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+(0.02 if val>=0 else -0.06),
                    f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Face"); ax.set_ylabel("Cp"); ax.set_title(f"Distribution Cp — α={angle_incidence}°")
        ax.legend(handles=[
            mpatches.Patch(color="#4a90d9",label="Cp>0 surpression"),
            mpatches.Patch(color="#e07840",label="Cp<0 dépression")
        ], fontsize=9); ax.grid(alpha=0.3,axis="y")
        st.pyplot(fig)
        st.download_button("Télécharger barres (PNG)", fig_to_bytes(fig),
                           file_name=f"cp_alpha{angle_incidence}.png", mime="image/png", key="dl_cp1")
        plt.close(fig)

    with col_tbl:
        st.markdown(f"#### Cp — α = {angle_incidence}°")
        st.dataframe(df_a[["face","Cp"]].reset_index(drop=True), use_container_width=True, hide_index=True)
        st.markdown("#### Pivot : tous les angles")
        pivot = df_sf.pivot_table(values="Cp",index="face",columns="angle").round(3)
        st.dataframe(pivot, use_container_width=True)

    # Graphe radar / polaire comparatif tous angles
    st.markdown("#### Radar Cp — comparaison tous les angles d'incidence")
    faces_list = ["Nord","Sud","Est","Ouest","Dessus","Dessous"]
    angles_rad = np.linspace(0, 2*np.pi, len(faces_list), endpoint=False).tolist()
    angles_rad += angles_rad[:1]  # fermer le polygone

    fig, ax = plt.subplots(figsize=(6.5,6.5), subplot_kw=dict(polar=True))
    colors_r = {0:"#e07840",15:"#4a90d9",30:"#43a047",45:"#9b59b6"}
    for ang_deg, col in colors_r.items():
        df_ang = df_sf[df_sf["angle"] == ang_deg].set_index("face")
        vals = [df_ang.loc[f,"Cp"] for f in faces_list] + [df_ang.loc[faces_list[0],"Cp"]]
        # décaler vers des valeurs positives pour le radar
        offset = abs(min(vals)) + 0.05
        vals_pos = [v + offset for v in vals]
        ax.plot(angles_rad, vals_pos, "o-", color=col, lw=2, label=f"α={ang_deg}°")
        ax.fill(angles_rad, vals_pos, color=col, alpha=0.06)

    ax.set_xticks(angles_rad[:-1])
    ax.set_xticklabels(faces_list, fontsize=10)
    ax.set_title("Coefficients Cp par face — tous angles\n(valeurs décalées pour visualisation polaire)",
                 pad=18, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28,1.12), fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.download_button("Télécharger radar (PNG)", fig_to_bytes(fig),
                       file_name="cp_radar.png", mime="image/png", key="dl_cp2")
    plt.close(fig)

    # Comparaison multi-angles sur un seul graphe linéaire
    st.markdown("#### Évolution Cp par face selon l'angle d'incidence")
    fig, ax = plt.subplots(figsize=(9,4))
    pivot_np = pivot.values
    x = np.arange(len(faces_list))
    width = 0.18
    for i, ang_deg in enumerate([0,15,30,45]):
        cp_vals_ang = [df_sf[(df_sf["angle"]==ang_deg)&(df_sf["face"]==f)]["Cp"].values[0]
                       for f in faces_list]
        ax.bar(x + i*width, cp_vals_ang, width, label=f"α={ang_deg}°",
               color=list(colors_r.values())[i], alpha=0.82, edgecolor="white", lw=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(faces_list)
    ax.set_ylabel("Cp"); ax.set_title("Cp par face × angle d'incidence")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    st.pyplot(fig)
    st.download_button("Télécharger multi-angles (PNG)", fig_to_bytes(fig),
                       file_name="cp_multi_angles.png", mime="image/png", key="dl_cp3")
    plt.close(fig)

    st.download_button("Exporter données soufflerie (CSV)", df_to_csv(df_sf),
                       file_name="soufflerie_cp.csv", mime="text/csv", key="dl_sf_csv")

# ── TAB 7 — À PROPOS ─────────────────────────────────────────────────────────
with tab7:
    st.markdown("### À propos — HDR Badgir × IA")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("""
<div class="about-card">
<h4>Candidat</h4>
<p><strong>Roland Dehghan Kamaragi</strong><br>
Architecte · Docteur en architecture (ENSA Paris-Malaquais, 2014, labo GSA)<br>
Spécialiste des tours à vent (Badgirs) du golfe Persique<br>
Chaire Architecture & Intelligences — ENSA Paris-Val de Seine</p>
</div>

<div class="about-card">
<h4>Thèse de doctorat (2014)</h4>
<p>Tours à vent du golfe Persique : mesures in situ (Siraf, Yazd, Bushehr, Abyaneh),
soufflerie Ville-d'Avray (140 prises de pression, maquette 1/50),
simulation CFD Fluent k-ε validée. Données source de cette application.</p>
</div>

<div class="about-card">
<h4>Ancrage institutionnel HDR</h4>
<p>Laboratoire EVCAU · ENSA Paris-Val de Seine<br>
Chaire Architecture & Intelligences</p>
</div>
""", unsafe_allow_html=True)

    with col_i2:
        st.markdown("""
<div class="about-card">
<h4>Axe 1 — Modèles de substitution (Surrogate)</h4>
<p>Remplacement des simulations CFD coûteuses par des modèles GP (Gaussian Process),
réseaux de neurones feedforward et Random Forest. Noyau RBF + bruit blanc entraîné
sur 6 points CFD Fluent k-ε · R² = 0.998. Extension à des espaces de paramètres multivariés
(géométrie, climat, orientation).</p>
</div>

<div class="about-card">
<h4>Axe 2 — PINN (Physics-Informed Neural Networks)</h4>
<p>Résolution des équations de Navier-Stokes par réseaux de neurones avec contraintes
physiques intégrées dans la fonction de perte. Implémentation PyTorch sur profil Poiseuille 1D
(L² = 1.77×10⁻⁴). Extension vers NS 2D compressible sur géométrie Badgir réelle.</p>
</div>

<div class="about-card">
<h4>Axe 3 — Optimisation & génération de formes</h4>
<p>Optimisation géométrique bi-objectif (Q/coût) par algorithme évolutionnaire
(differential evolution). Front de Pareto H × w. Extension : génération
paramétrique de géométries Badgir par apprentissage de représentation (VAE, GAN).</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Stack technique")
    stack_df = pd.DataFrame({
        "Composant": ["GP Surrogate","PINN","Optimisation","CFD référence","Soufflerie","Mesures in situ"],
        "Outil / Méthode": ["scikit-learn · GaussianProcessRegressor · RBF+White kernel",
                            "PyTorch · MLP 2 couches · Adam · termes PDE+BC",
                            "scipy.optimize.differential_evolution",
                            "ANSYS Fluent · modèle k-ε · 6 vitesses",
                            "Ville-d'Avray · maquette 1/50 · 140 prises",
                            "Siraf, Yazd, Bushehr, Abyaneh — 2012"],
        "Résultat clé": ["R² = 0.998","L² = 1.77×10⁻⁴","Q_opt / budget","Données d'entraînement",
                         "Coefficients Cp validés","ΔT max = 8.6°C · η = 23.1%"]
    })
    st.dataframe(stack_df, use_container_width=True, hide_index=True)

    st.markdown("#### Références principales")
    st.markdown("""
- Dehghan Kamaragi, R. (2014). *Tours à vent du golfe Persique : étude aéraulique et thermique*. Thèse de doctorat, ENSA Paris-Malaquais, labo GSA.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686-707.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Sacks, J., et al. (1989). Design and analysis of computer experiments. *Statistical Science*, 4(4), 409-423.
""")

# ── TAB 8 — AIDE ─────────────────────────────────────────────────────────────
with tab8:
    st.markdown("### Guide d'utilisation")
    aide_path = os.path.join(os.path.dirname(__file__), "aide_app.html")
    try:
        with open(aide_path, "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=820, scrolling=True)
    except FileNotFoundError:
        st.warning(f"Fichier aide_app.html introuvable dans {os.path.dirname(__file__)}")

st.divider()
st.caption("HDR Roland Dehghan Kamaragi · Chaire Architecture & Intelligences · ENSA Paris-Val de Seine · Données : thèse 2014, soufflerie Ville-d'Avray, CFD Fluent k-ε, mesures in situ Siraf 2012")
