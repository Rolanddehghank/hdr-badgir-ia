# HDR — Ventilation passive & IA
**Roland Dehghan Kamaragi** · EVCAU · ENSA Paris-Val de Seine · Chaire Architecture & Intelligences

## Structure du projet
```
hdr_projet/
├── data/                   ← Données expérimentales
│   ├── siraf_mesures.csv      Mesures in situ Siraf (avril 2013)
│   ├── soufflerie_prises.csv  140 prises de pression, 4 angles
│   ├── cfd_debit.csv          Comparatif soufflerie / CFD Fluent
│   └── yazd_geometrie.json    Géométrie Dolat-Abad
├── notebooks/              ← Progression étape par étape
│   ├── 01_python_bases_siraf.ipynb     Étape 2 : Python + données
│   ├── 02_soufflerie_pression.ipynb    Étape 2 : Visualisation Cp
│   ├── 03_surrogate_ml.ipynb           Étape 3 : Axe 1 — Surrogate
│   └── 04_pinn_intro.ipynb             Étape 5 : Axe 2 — PINN
├── models/                 ← Modèles entraînés (.pkl, .pth)
├── figures/                ← Graphes générés par les notebooks
├── app/
│   └── app.py              Application Streamlit (prototype)
└── requirements.txt
```

## Installation
```bash
conda create -n hdr python=3.11 -y
conda activate hdr
pip install -r requirements.txt
```

## Lancer les notebooks
```bash
jupyter notebook notebooks/
```

## Lancer l'application
```bash
streamlit run app/app.py
```

## 3 axes de recherche HDR
- **Axe 1** : Modèles surrogate ML (notebook 03)
- **Axe 2** : PINN / Physics-Informed Neural Networks (notebook 04)
- **Axe 3** : Génération de formes (à venir)
