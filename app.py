from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="House Price Estimator")

BASE = Path(__file__).parent
NEIGHBORHOODS = ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards',
                 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge',
                 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "house_model.pkl", "rb"))


@st.cache_resource
def load_meta():
    return pickle.load(open(BASE / "house_meta.pkl", "rb"))


model = load_model()
meta = load_meta()
feature_cols, defaults = meta["feature_cols"], meta["defaults"]

st.title("House Price Estimator")
st.write(
    "A Random Forest model (trained on the Kaggle 'House Prices - Advanced Regression Techniques' dataset, "
    "1,460 homes in Ames, Iowa) estimates a sale price from the most important features. Every other feature "
    "is filled in with its typical (median or most common) value from the training data."
)

col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall quality (1=worst, 10=best)", 1, 10, 6)
    overall_cond = st.slider("Overall condition (1=worst, 9=best)", 1, 9, 5)
    gr_liv_area = st.slider("Above-ground living area (sq ft)", 300, 5000, 1500, 10)
    total_bsmt_sf = st.slider("Total basement area (sq ft)", 0, 3000, 800, 10)
    first_flr_sf = st.slider("First floor area (sq ft)", 300, 3000, 1100, 10)
    second_flr_sf = st.slider("Second floor area (sq ft)", 0, 2500, 400, 10)
with col2:
    year_built = st.slider("Year built", 1870, 2025, 1990)
    year_remod = st.slider("Year remodeled (= year built if never remodeled)", 1870, 2025, 2000)
    garage_cars = st.slider("Garage capacity (cars)", 0, 4, 2)
    garage_area = st.slider("Garage area (sq ft)", 0, 1200, 480, 10)
    lot_area = st.slider("Lot area (sq ft)", 1000, 50000, 9500, 100)
    full_bath = st.slider("Full bathrooms", 0, 4, 2)

neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=NEIGHBORHOODS.index("NAmes"))
central_air = st.selectbox("Central air conditioning", ["Yes", "No"])

if st.button("Estimate price"):
    row = dict(defaults)
    row.update({
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "LotArea": lot_area,
        "FullBath": full_bath,
        "Neighborhood": neighborhood,
        "CentralAir": "Y" if central_air == "Yes" else "N",
    })

    X = pd.get_dummies(pd.DataFrame([row]))
    X = X.reindex(columns=feature_cols, fill_value=0)

    pred_log = model.predict(X)[0]
    pred = float(np.expm1(pred_log))
    st.success(f"Estimated sale price: **${pred:,.0f}**")

st.caption(
    "Model: Random Forest on the log-transformed sale price (validation RMSE ≈ 0.146 on the log scale). "
    "'Overall quality' has by far the largest effect on the prediction (about 56% of the model's decisions)."
)
