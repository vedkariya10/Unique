import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# ── Ordinal mappings ─────────────────────────────────────────────────────────
AGE_ORDER       = ["Under 18","18-22","23-28","29-35","36-45","45+"]
INCOME_ORDER    = ["Below 3L","3-6L","6-12L","12-20L","20-35L","35L+"]
PAIRS_ORDER     = ["1-2","3-5","6-10","11-20","20+"]
FREQ_ORDER      = ["Yearly","Biannually","Quarterly","Monthly","Weekly"]
SPEND_ORDER     = ["Below 2000","2000-5000","5000-10000","10000-20000","20000-50000","50000+"]
DELIVERY_FEE_ORDER = ["Free only","Up to 49","Up to 99","Up to 199","Any fee"]
BUDGET_ORDER    = ["Below 3000","3000-6000","6000-10000","10000-18000","18000-35000","No ceiling"]
SUBS_ORDER      = ["No subscription","Maybe value","Yes 499","Yes 999"]

PERSONA_COLORS = {
    "Hype_Collector":       "#7F77DD",
    "Aspirational_Adopter": "#1D9E75",
    "Comfort_Pragmatist":   "#EF9F27",
    "Gift_Occasion_Buyer":  "#D85A30",
    "Research_Driven":      "#378ADD",
}

CLUSTER_PALETTE = ["#7F77DD","#1D9E75","#EF9F27","#D85A30","#378ADD","#D4537E"]

# ── Column groups ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "q06_ps1_identity","q06_ps2_social","q06_ps3_comfort",
    "q06_ps4_rarity","q06_ps5_research",
    "q24_tr1_spot_fake","q24_tr2_trust_platform","q24_tr3_auth_cert",
    "q12_vw1_too_cheap","q12_vw2_bargain","q12_vw3_expensive","q12_vw4_too_exp",
    "q18_delivery_imp","q26_satisfaction","q30_nps",
    "psychographic_identity_score","trust_index","wtp_range","wtp_midpoint",
    "brand_diversity_count","accessory_basket_size","occasion_breadth",
]

ORDINAL_COLS = [
    ("q01_age",            AGE_ORDER),
    ("q05_income",         INCOME_ORDER),
    ("q07_pairs_owned",    PAIRS_ORDER),
    ("q08_freq",           FREQ_ORDER),
    ("q09_max_spend",      SPEND_ORDER),
    ("q29_delivery_fee",   DELIVERY_FEE_ORDER),
    ("q31_everyday_budget",BUDGET_ORDER),
    ("q27_subscription",   SUBS_ORDER),
]

NOMINAL_COLS = [
    "q02_gender","q03_city","q04_occupation",
    "q16_priority","q17_switching","q19_tryon",
    "q20_digital","q21_hype","q23_social_role","q25_frustration",
    "q28_cleaning",
]

BINARY_DERIVED = ["hype_active","app_native","high_trust","gift_buyer"]

# ── Feature engineering on raw df ────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def safe_mean(row, cols):
        vals = [row[c] for c in cols if c in row.index and pd.notna(row[c])]
        return round(float(np.mean(vals)), 2) if vals else np.nan

    ps_cols = ["q06_ps1_identity","q06_ps2_social","q06_ps3_comfort","q06_ps4_rarity","q06_ps5_research"]
    tr_cols = ["q24_tr1_spot_fake","q24_tr2_trust_platform","q24_tr3_auth_cert"]

    df["psychographic_identity_score"] = df[ps_cols].mean(axis=1).round(2)
    df["trust_index"] = df[tr_cols].mean(axis=1).round(2)

    for c in ["q12_vw1_too_cheap","q12_vw2_bargain","q12_vw3_expensive","q12_vw4_too_exp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["wtp_range"]    = df["q12_vw4_too_exp"] - df["q12_vw1_too_cheap"]
    df["wtp_midpoint"] = ((df["q12_vw2_bargain"] + df["q12_vw3_expensive"]) / 2).round(0)

    if "q10_brands" in df.columns:
        df["brand_diversity_count"] = df["q10_brands"].fillna("").apply(
            lambda x: len([i for i in str(x).split("|") if i.strip()]))
    if "q14_accessories" in df.columns:
        df["accessory_basket_size"] = df["q14_accessories"].fillna("").apply(
            lambda x: len([i for i in str(x).split("|") if i.strip()]))
    if "q15_occasions" in df.columns:
        df["occasion_breadth"] = df["q15_occasions"].fillna("").apply(
            lambda x: len([i for i in str(x).split("|") if i.strip()]))

    if "q21_hype" in df.columns:
        df["hype_active"] = (df["q21_hype"] == "Yes actively").astype(int)
    if "q20_digital" in df.columns:
        df["app_native"] = (df["q20_digital"] == "Daily app user").astype(int)
    if "trust_index" in df.columns:
        df["high_trust"] = (df["trust_index"] >= 4.0).astype(int)
    if "q15_occasions" in df.columns:
        df["gift_buyer"] = df["q15_occasions"].fillna("").str.contains(
            "Birthday gift|Festival Diwali").astype(int)

    if "q30_nps" in df.columns:
        df["q30_nps"] = pd.to_numeric(df["q30_nps"], errors="coerce")
        df["nps_segment"] = pd.cut(
            df["q30_nps"], bins=[-1,5,7,10],
            labels=["Detractor","Passive","Promoter"])

    return df


def load_and_prepare(path: str = "swiftsole_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["noise_type"] == "clean"].copy()
    df = engineer_features(df)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return X (feature matrix) ready for sklearn."""
    num_feats    = [c for c in NUMERIC_COLS if c in df.columns]
    ord_feats    = [c for c, _ in ORDINAL_COLS if c in df.columns]
    ord_cats     = [cats for c, cats in ORDINAL_COLS if c in df.columns]
    nom_feats    = [c for c in NOMINAL_COLS if c in df.columns]
    bin_feats    = [c for c in BINARY_DERIVED if c in df.columns]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    ord_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("enc",    OrdinalEncoder(categories=ord_cats,
                                  handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ])
    nom_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("enc",    OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ])
    bin_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = []
    if num_feats: transformers.append(("num", num_pipe, num_feats))
    if ord_feats: transformers.append(("ord", ord_pipe, ord_feats))
    if nom_feats: transformers.append(("nom", nom_pipe, nom_feats))
    if bin_feats: transformers.append(("bin", bin_pipe, bin_feats))

    preprocessor = ColumnTransformer(transformers=transformers,
                                     remainder="drop")
    all_feats = num_feats + ord_feats + nom_feats + bin_feats
    return preprocessor, all_feats


def get_feature_names(preprocessor, num_feats, ord_feats, nom_feats, bin_feats):
    return num_feats + ord_feats + nom_feats + bin_feats
