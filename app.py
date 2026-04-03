# app_improved.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import traceback


import os

clf_path = os.path.join(os.path.dirname(__file__), "xgb_clf.pkl")
reg_path = os.path.join(os.path.dirname(__file__), "xgb_reg.pkl")

xgb_clf = joblib.load(clf_path)
xgb_reg = joblib.load(reg_path)

st.set_page_config(layout="wide", page_title="EMI Predictor", page_icon="💳")

# --------------------------
# Basic mappings (use exactly your encodings)
label_mappings = {
    'gender': {'female': 0, 'male': 1},
    'marital_status': {'Married': 0, 'Single': 1},
    'education': {'High School': 1, 'Graduate': 2, 'Post Graduate': 3, 'Professional': 4},
    'company_type': {'Startup': 1, 'Small': 2, 'Mid-size': 3, 'Large Indian': 4, 'MNC': 5},
    'house_type': {'Rented': 1, 'Family': 2, 'Own': 3},
    'existing_loans': {'No': 0, 'Yes': 1}
}
employment_types = ['Private', 'Government', 'Self-employed']
emi_scenarios = [
    'Personal Loan EMI','E-commerce Shopping EMI','Education EMI',
    'Vehicle EMI','Home Appliances EMI'
]
emi_to_num = {"Not_Eligible":0, "Eligible":2, "High_Risk":1}  # train mapping recall

# --------------------------
# Load models
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

clf = load_model("xgb_clf.pkl")
reg = load_model("xgb_reg.pkl")

# helper: get feature importances safely
def get_feat_importances(model, feature_list):
    try:
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            fi = booster.get_score(importance_type="gain")
            fi = pd.Series(fi).reindex(feature_list).fillna(0).values
        if fi is None:
            return None
        return pd.DataFrame({"feature": feature_list, "importance": fi}).sort_values("importance", ascending=False)
    except Exception:
        return None

# ---------- UI header ----------
st.markdown("## 💳 EMI Eligibility & Max EMI Predictor — Improved UI")
st.markdown("Two flows: (A) Predict eligibility — needs *max_monthly_emi* from user, (B) Predict max EMI — needs *emi_eligibility_num* from user. Use presets for quick testing.")

tabs = st.tabs(["A — Predict Eligibility (Classifier)", "B — Predict Max EMI (Regressor)"])

# Quick presets (use correct numeric types)
PRESETS = {
    "Default": {},
    "Conservative": {"monthly_salary":20000.0, "credit_score":650, "bank_balance":20000.0},
    "High Salary": {"monthly_salary":120000.0, "credit_score":760, "bank_balance":200000.0},
}

# mapping of preset keys -> widget key prefix used in base_inputs
_PRESET_TO_WIDGET = {
    "monthly_salary": "sal_",
    "credit_score": "cs_",
    "bank_balance": "bb_",
    "age": "age_",
    "family_size": "fam_",
    "years_of_employment": "yoe_",
    # add more if you include them in presets
}

# shared helper for base inputs (makes code tidy) — now reads/writes session_state values
def base_inputs(prefix, defaults=None):
    defaults = defaults or {}

    # safe-casting helpers
    def f(key, fallback):
        v = defaults.get(key, fallback)
        try:
            return float(v)
        except Exception:
            return float(fallback)

    def i(key, fallback):
        v = defaults.get(key, fallback)
        try:
            return int(v)
        except Exception:
            return int(fallback)

    # widget keys
    k_age = f"age_{prefix}"
    k_sal = f"sal_{prefix}"
    k_yoe = f"yoe_{prefix}"
    k_rent = f"rent_{prefix}"
    k_fam = f"fam_{prefix}"
    k_sf = f"sf_{prefix}"
    k_cf = f"cf_{prefix}"
    k_tr = f"tr_{prefix}"
    k_gr = f"gr_{prefix}"
    k_ot = f"ot_{prefix}"
    k_cemi = f"cemi_{prefix}"
    k_cs = f"cs_{prefix}"
    k_bb = f"bb_{prefix}"

    # determine initial values: session_state overrides defaults if present
    init_age = st.session_state.get(k_age, i('age', 30))
    init_sal = st.session_state.get(k_sal, f('monthly_salary', 30000.0))
    init_yoe = st.session_state.get(k_yoe, f('years_of_employment', 3.0))
    init_rent = st.session_state.get(k_rent, f('monthly_rent', 8000.0))
    init_fam = st.session_state.get(k_fam, i('family_size', 3))
    init_sf = st.session_state.get(k_sf, f('school_fees', 0.0))
    init_cf = st.session_state.get(k_cf, f('college_fees', 0.0))
    init_tr = st.session_state.get(k_tr, f('travel_expenses', 2000.0))
    init_gr = st.session_state.get(k_gr, f('groceries_utilities', 6000.0))
    init_ot = st.session_state.get(k_ot, f('other_monthly_expenses', 2000.0))
    init_cemi = st.session_state.get(k_cemi, f('current_emi_amount', 0.0))
    init_cs = st.session_state.get(k_cs, i('credit_score', 700))
    init_bb = st.session_state.get(k_bb, f('bank_balance', 50000.0))

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, value=init_age, key=k_age)
        monthly_salary = st.number_input("Monthly salary (₹)", 0.0, 5_000_000.0, value=init_sal, step=1000.0, key=k_sal)
        years_of_employment = st.number_input("Years of employment", 0.0, 60.0, value=init_yoe, step=0.5, key=k_yoe)
        monthly_rent = st.number_input("Monthly rent (₹)", 0.0, 500000.0, value=init_rent, key=k_rent)
        family_size = st.number_input("Family size", 1, 20, value=init_fam, key=k_fam)
        school_fees = st.number_input("School fees (monthly) (₹)", 0.0, 200000.0, value=init_sf, key=k_sf)
        college_fees = st.number_input("College fees (monthly) (₹)", 0.0, 500000.0, value=init_cf, key=k_cf)
    with c2:
        travel_expenses = st.number_input("Travel expenses (₹)", 0.0, 200000.0, value=init_tr, key=k_tr)
        groceries_utilities = st.number_input("Groceries & utilities (₹)", 0.0, 500000.0, value=init_gr, key=k_gr)
        other_monthly_expenses = st.number_input("Other monthly expenses (₹)", 0.0, 500000.0, value=init_ot, key=k_ot)
        current_emi_amount = st.number_input("Current EMI amount (₹)", 0.0, 1_000_000.0, value=init_cemi, key=k_cemi)
        credit_score = st.number_input("Credit score", 300, 900, value=init_cs, key=k_cs)
        bank_balance = st.number_input("Bank balance (₹)", 0.0, 10_000_000.0, value=init_bb, key=k_bb)

    return dict(age=age, monthly_salary=monthly_salary, years_of_employment=years_of_employment,
                monthly_rent=monthly_rent, family_size=family_size, school_fees=school_fees,
                college_fees=college_fees, travel_expenses=travel_expenses,
                groceries_utilities=groceries_utilities, other_monthly_expenses=other_monthly_expenses,
                current_emi_amount=current_emi_amount, credit_score=credit_score, bank_balance=bank_balance)

# ---------------- Tab A: Classifier (requires max_monthly_emi)
with tabs[0]:
    st.header("A — Predict EMI Eligibility")

    col_left, col_right = st.columns([1,1])
    with col_left:
        preset = st.selectbox("Load preset for inputs (A)", list(PRESETS.keys()), index=0, help="Choose a quick preset for inputs")
        if st.button("Apply preset (A)"):
            # set session state keys for preset values (only keys present in PRESETS)
            for k, v in PRESETS.get(preset, {}).items():
                widget_pref = _PRESET_TO_WIDGET.get(k)
                if widget_pref:
                    st.session_state[f"{widget_pref}A"] = v
            # rerun to reflect changes; fallback to instruct user to refresh
            try:
                st.experimental_rerun()
            except Exception:
                st.success("Preset applied — please refresh page to see values")

    # build base inputs
    baseA = base_inputs("A", PRESETS.get(preset))
    with st.expander("Additional inputs (required for classifier)"):
        requested_amount_A = st.number_input("Requested loan amount (₹)", 0.0, 5_000_000.0, value=200000.0, key="req_amt_A")
        requested_tenure_A = st.number_input("Requested tenure (months)", 1, 120, value=12, key="req_ten_A")
        max_monthly_emi_input = st.number_input("Max monthly EMI (₹) — required for classifier", 0.0, 500000.0, value=5000.0, key="maxemi_A")
        gender_A = st.selectbox("Gender", list(label_mappings['gender'].keys()), key="gA")
        marital_status_A = st.selectbox("Marital Status", list(label_mappings['marital_status'].keys()), key="mA")
        education_A = st.selectbox("Education", list(label_mappings['education'].keys()), key="eA")
        company_type_A = st.selectbox("Company Type", list(label_mappings['company_type'].keys()), key="compA")
        house_type_A = st.selectbox("House Type", list(label_mappings['house_type'].keys()), key="houseA")
        existing_loans_A = st.selectbox("Existing Loans", list(label_mappings['existing_loans'].keys()), key="exA")
        employment_type_A = st.selectbox("Employment Type", employment_types, key="empA")
        emi_scenario_A = st.selectbox("EMI Scenario", emi_scenarios, key="emiA")

    # prepare features
    clf_features = [
     'age','monthly_salary','years_of_employment','monthly_rent','family_size',
     'school_fees','college_fees','travel_expenses','groceries_utilities',
     'other_monthly_expenses','current_emi_amount','credit_score','bank_balance',
     'requested_amount','requested_tenure','max_monthly_emi',
     'gender_num','marital_status_num','education_num',
     'employment_type_Government','employment_type_Private','employment_type_Self-employed',
     'company_type_num','house_type_num','existing_loans_num',
     'emi_scenario_E-commerce Shopping EMI','emi_scenario_Education EMI',
     'emi_scenario_Home Appliances EMI','emi_scenario_Personal Loan EMI','emi_scenario_Vehicle EMI'
    ]
    clf_input = {
        **baseA,
        'requested_amount': requested_amount_A,
        'requested_tenure': requested_tenure_A,
        'max_monthly_emi': max_monthly_emi_input,
        'gender_num': label_mappings['gender'][gender_A],
        'marital_status_num': label_mappings['marital_status'][marital_status_A],
        'education_num': label_mappings['education'][education_A],
        'company_type_num': label_mappings['company_type'][company_type_A],
        'house_type_num': label_mappings['house_type'][house_type_A],
        'existing_loans_num': label_mappings['existing_loans'][existing_loans_A],
        'employment_type_Government': 1 if employment_type_A=='Government' else 0,
        'employment_type_Private': 1 if employment_type_A=='Private' else 0,
        'employment_type_Self-employed': 1 if employment_type_A=='Self-employed' else 0,
        'emi_scenario_Personal Loan EMI': 1 if emi_scenario_A=='Personal Loan EMI' else 0,
        'emi_scenario_E-commerce Shopping EMI': 1 if emi_scenario_A=='E-commerce Shopping EMI' else 0,
        'emi_scenario_Education EMI': 1 if emi_scenario_A=='Education EMI' else 0,
        'emi_scenario_Vehicle EMI': 1 if emi_scenario_A=='Vehicle EMI' else 0,
        'emi_scenario_Home Appliances EMI': 1 if emi_scenario_A=='Home Appliances EMI' else 0
    }
    clf_input_df = pd.DataFrame([clf_input]).reindex(columns=clf_features, fill_value=0)

    # show prepared input
    if st.checkbox("Show classifier-prepared input", key="show_clf_input"):
        st.dataframe(clf_input_df.T.rename(columns={0:"value"}), height=400)

    # predict
    colp1, colp2 = st.columns([1,2])
    with colp1:
        if st.button("Predict Eligibility (Classifier)"):
            if clf is None:
                st.error("Classifier model not loaded (xgb_clf.pkl).")
            else:
                try:
                    clf_input_df = clf_input_df.astype(float)
                    pred = clf.predict(clf_input_df)[0]
                    probs = clf.predict_proba(clf_input_df)[0] if hasattr(clf, "predict_proba") else None
                    label_map = {0:"Not_Eligible", 1:"High_Risk", 2:"Eligible"}
                    st.success(f"Predicted class: {label_map.get(pred,pred)}")
                    if probs is not None:
                        st.subheader("Prediction probabilities")
                        prob_df = pd.DataFrame({"class":[0,1,2],"prob":probs})
                        prob_df['label'] = prob_df['class'].map(label_map)
                        st.bar_chart(prob_df.set_index('label')['prob'])
                        st.write(prob_df[['label','prob']].set_index('label').T)
                    # show feature importance (top 8)
                    fi = get_feat_importances(clf, clf_features)
                    if fi is not None:
                        st.subheader("Top features (classifier)")
                        st.bar_chart(fi.head(10).set_index('feature')['importance'])
                except Exception:
                    st.error("Classifier failed:")
                    st.text(traceback.format_exc())

# ---------------- Tab B: Regressor (requires emi_eligibility_num)
with tabs[1]:
    st.header("B — Predict Max Monthly EMI")
    col_lp, col_rp = st.columns([1,1])
    with col_lp:
        presetB = st.selectbox("Load preset for inputs (B)", list(PRESETS.keys()), index=0, key="presetB")
        if st.button("Apply preset (B)"):
            for k, v in PRESETS.get(presetB, {}).items():
                widget_pref = _PRESET_TO_WIDGET.get(k)
                if widget_pref:
                    st.session_state[f"{widget_pref}B"] = v
            try:
                st.experimental_rerun()
            except Exception:
                st.success("Preset applied — please refresh page to see values")

        st.caption("Pick a preset to pre-fill values — then edit fields as needed.")
    baseB = base_inputs("B", PRESETS.get(presetB))
    with st.expander("Other inputs for EMI prediction"):
        requested_amount_B = st.number_input("Requested loan amount (₹)", 0.0, 5_000_000.0, value=200000.0, key="req_amt_B")
        requested_tenure_B = st.number_input("Requested tenure (months)", 1, 120, value=12, key="req_ten_B")
        emi_label_choice = st.selectbox("Provide EMI eligibility class (or run classifier above and copy)", ["Not_Eligible","High_Risk","Eligible"], key="emi_label_B")
        emi_eligibility_num_input = emi_to_num[emi_label_choice]
        gender_B = st.selectbox("Gender", list(label_mappings['gender'].keys()), key="gB")
        marital_status_B = st.selectbox("Marital Status", list(label_mappings['marital_status'].keys()), key="mB")
        education_B = st.selectbox("Education", list(label_mappings['education'].keys()), key="eB")
        company_type_B = st.selectbox("Company Type", list(label_mappings['company_type'].keys()), key="compB")
        house_type_B = st.selectbox("House Type", list(label_mappings['house_type'].keys()), key="houseB")
        existing_loans_B = st.selectbox("Existing Loans", list(label_mappings['existing_loans'].keys()), key="exB")
        employment_type_B = st.selectbox("Employment Type", employment_types, key="empB")
        emi_scenario_B = st.selectbox("EMI Scenario", emi_scenarios, key="emiB")

    reg_features = [
     'age','monthly_salary','years_of_employment','monthly_rent','family_size',
     'school_fees','college_fees','travel_expenses','groceries_utilities',
     'other_monthly_expenses','current_emi_amount','credit_score','bank_balance',
     'requested_amount','requested_tenure',
     'gender_num','marital_status_num','education_num',
     'employment_type_Government','employment_type_Private','employment_type_Self-employed',
     'company_type_num','house_type_num','existing_loans_num',
     'emi_scenario_E-commerce Shopping EMI','emi_scenario_Education EMI',
     'emi_scenario_Home Appliances EMI','emi_scenario_Personal Loan EMI','emi_scenario_Vehicle EMI',
     'emi_eligibility_num'
    ]

    reg_input = {
        **baseB,
        'requested_amount': requested_amount_B,
        'requested_tenure': requested_tenure_B,
        'gender_num': label_mappings['gender'][gender_B],
        'marital_status_num': label_mappings['marital_status'][marital_status_B],
        'education_num': label_mappings['education'][education_B],
        'company_type_num': label_mappings['company_type'][company_type_B],
        'house_type_num': label_mappings['house_type'][house_type_B],
        'existing_loans_num': label_mappings['existing_loans'][existing_loans_B],
        'employment_type_Government': 1 if employment_type_B=='Government' else 0,
        'employment_type_Private': 1 if employment_type_B=='Private' else 0,
        'employment_type_Self-employed': 1 if employment_type_B=='Self-employed' else 0,
        'emi_scenario_Personal Loan EMI': 1 if emi_scenario_B=='Personal Loan EMI' else 0,
        'emi_scenario_E-commerce Shopping EMI': 1 if emi_scenario_B=='E-commerce Shopping EMI' else 0,
        'emi_scenario_Education EMI': 1 if emi_scenario_B=='Education EMI' else 0,
        'emi_scenario_Vehicle EMI': 1 if emi_scenario_B=='Vehicle EMI' else 0,
        'emi_scenario_Home Appliances EMI': 1 if emi_scenario_B=='Home Appliances EMI' else 0,
        'emi_eligibility_num': emi_eligibility_num_input
    }
    reg_input_df = pd.DataFrame([reg_input]).reindex(columns=reg_features, fill_value=0)

    if st.checkbox("Show regressor-prepared input", key="show_reg_input"):
        st.dataframe(reg_input_df.T.rename(columns={0:"value"}), height=400)

    if st.button("Predict Max Monthly EMI (Regressor)"):
        if reg is None:
            st.error("Regressor not loaded (xgb_reg.pkl).")
        else:
            try:
                reg_input_df = reg_input_df.astype(float)
                pred_emi = reg.predict(reg_input_df)[0]
                st.success(f"Predicted Max monthly EMI: ₹{pred_emi:,.2f}")
                fi2 = get_feat_importances(reg, reg_features)
                if fi2 is not None:
                    st.subheader("Top features (regressor)")
                    st.bar_chart(fi2.head(10).set_index('feature')['importance'])
            except Exception:
                st.error("Regressor prediction failed:")
                st.text(traceback.format_exc())

# ---- footer notes ----
st.markdown("---")
st.caption("Tip: run the classifier (Tab A) first to get an eligibility class, then copy that into Tab B 'Provide EMI eligibility class' to get a regressed EMI. This preserves the models' training setup.")
