from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import json
import shap

# ── Startup — modelləri yüklə ─────────────────────────────────
app = FastAPI(
    title="Credit Risk Assessment API",
    description="Bank kredit riski qiymətləndirmə sistemi",
    version="1.0"
)

te        = joblib.load('models/target_encoder.pkl')
pipeline  = joblib.load('models/credit_pipeline.pkl')
explainer = joblib.load('models/shap_explainer.pkl')

with open('models/model_config.json') as f:
    config = json.load(f)

APPROVE_THRESHOLD = config['thresholds']['approve']
DECLINE_THRESHOLD = config['thresholds']['decline']
LGD               = config['lgd']

# ── Input schema ──────────────────────────────────────────────
class CustomerInput(BaseModel):
    # Manual input — müştəri doldurur
    name_contract_type  : str     = Field(default='Cash loans')
    code_gender         : str     = Field(default='M')
    flag_own_car        : str     = Field(default='N')
    flag_own_realty     : str     = Field(default='Y')
    amt_income_total    : float   = Field(default=150000.0)
    amt_credit          : float   = Field(default=300000.0)
    amt_annuity         : float   = Field(default=15000.0)
    name_education_type : str     = Field(default='Secondary / secondary special')
    name_income_type    : str     = Field(default='Working')
    name_family_status  : str     = Field(default='Married')
    name_housing_type   : str     = Field(default='House / apartment')
    name_type_suite     : str     = Field(default='Unaccompanied')
    occupation_type     : str     = Field(default='Laborers')
    organization_type   : str     = Field(default='Business Entity Type 3')
    cnt_fam_members     : float   = Field(default=2.0)
    age                 : float   = Field(default=35.0)
    years_employed      : float   = Field(default=5.0)

    # Büro məlumatları — sistem çəkir
    ext_source_1                 : Optional[float] = 0.5
    ext_source_2                 : Optional[float] = 0.5
    ext_source_3                 : Optional[float] = 0.5
    max_payment_delay            : Optional[float] = 0.0
    avg_payment_delay            : Optional[float] = 0.0
    sk_id_bureau_count_x         : Optional[float] = 0.0
    is_credit_active_sum_x       : Optional[float] = 0.0
    amt_credit_sum_debt_sum_x    : Optional[float] = 0.0
    amt_credit_sum_overdue_sum_x : Optional[float] = 0.0
    amt_req_credit_bureau_qrt    : Optional[float] = 0.0
    refused_ratio_x              : Optional[float] = 0.0
    approval_rate_x              : Optional[float] = 1.0
    inst_count                   : Optional[float] = 0.0
    instalment_utilization_x     : Optional[float] = 0.0
    debt_ratio__x                : Optional[float] = 0.0
    region_rating_client         : Optional[float] = 2.0


def _build_dataframe(customer: CustomerInput) -> pd.DataFrame:
    """Input-dan model üçün DataFrame yarat"""
    d = customer.dict()

    # Hesablanmış featurelər
    d['ext_source_mean']   = np.mean([d['ext_source_1'],
                                       d['ext_source_2'],
                                       d['ext_source_3']])
    d['EXT_SOURCES_MEAN']  = d['ext_source_mean']
    d['income_to_credit']  = d['amt_income_total'] / max(d['amt_credit'], 1)
    d['credit_to_goods']   = 1.0
    d['days_birth']        = -d['age'] * 365
    d['days_id_publish']   = -2000.0
    d['days_last_phone_change'] = -500.0
    d['days_registration'] = -5000.0
    d['DAYS_EMPLOYED_ANOM'] = False
    d['age_group']         = (
        'Young_High_Risk'      if d['age'] < 30 else
        'Middle_Age_Medium_Risk' if d['age'] < 50 else
        'Senior_Low_Risk'
    )
    d['credit_segment']    = 'Medium'

    # Default dəyərlər — bürodan gəlməyənlər
    defaults = {
        'avg_payment_gap'         : 0.0,
        'max_payment_gap'         : 0.0,
        'amt_credit_sum_sum_x'    : d.get('amt_credit_sum_debt_sum_x', 0),
        'avg_instalment_y'        : d['amt_annuity'],
        'total_down_payment_x'    : 0.0,
        'amt_annuity_mean_x'      : d['amt_annuity'],
        'avg_payment_delay'       : d.get('avg_payment_delay', 0),
        'last_instalment_day'     : -30.0,
        'asked_credit_ratio_mean_x': 1.0,
        'days_decision_max_x'     : -100.0,
        'credit_gap_x'            : 0.0,
        'avg_cnt_payment_x'       : 12.0,
        'activity_span_x'         : 12.0,
        'avg_credit_x'            : d['amt_credit'],
        'pos_contract_count_x'    : 0.0,
        'days_decision_min_x'     : -500.0,
        'days_decision_mean_x'    : -300.0,
        'total_credit_x'          : d['amt_credit'],
        'months_balance_max_x'    : -1.0,
        'months_balance_min_x'    : -12.0,
        'own_car_age'             : 0.0,
        'amt_annuity_max_x'       : d['amt_annuity'],
        'type_consumer credit_sum_x': 0.0,
        'floorsmax_avg'           : 0.0,
        'avg_dpd_def_x'           : 0.0,
        'avg_instalment_x'        : d['amt_annuity'],
        'approval_minus_refusal_x': 1.0,
        'hour_appr_process_start' : 10.0,
        'down_payment_ratio_x'    : 0.0,
        'max_instalment_x'        : d['amt_annuity'],
        'type_mortgage_sum_x'     : 0.0,
        'elevators_avg'           : 0.0,
        'weekday_appr_process_start': 'MONDAY',
    }

    for col, val in defaults.items():
        if col not in d:
            d[col] = val

    return pd.DataFrame([d])


def _get_shap_factors(df_te, n=5):
    """SHAP ilə top risk faktorlarını al"""
    try:
        preprocessed = pipeline[:-1].transform(df_te)
        
        # Feature sayını yoxla
        print(f"Preprocessed shape: {preprocessed.shape}")
        print(f"Explainer expected: {explainer.model.num_feature()}")
        
        sv           = explainer.shap_values(preprocessed)
        sv_default   = sv[1] if isinstance(sv, list) else sv

        factors = pd.DataFrame({
            'feature' : config['feature_names'],
            'shap'    : sv_default[0]
        }).assign(abs=lambda x: x['shap'].abs())\
          .sort_values('abs', ascending=False)\
          .head(n)

        return [
            {
                'feature'  : row['feature'],
                'impact'   : round(float(row['shap']), 4),
                'direction': 'riski artırır' if row['shap'] > 0
                             else 'riski azaldır'
            }
            for _, row in factors.iterrows()
        ]
    except Exception as e:
        print(f"SHAP xətası: {e}")
        return []   # ← xəta olsa boş qaytar, predict işləsin

# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model"  : config['model'],
        "version": config['version'],
        "auc"    : config['auc_roc'],
        "gini"   : config['gini']
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        # DataFrame yarat
        df = _build_dataframe(customer)

        # TargetEncoder tətbiq et
        df_te = te.transform(df)

        # Predict
        proba = pipeline.predict_proba(df_te)[0][1]

        # 3-Tier qərar
        if proba < APPROVE_THRESHOLD:
            decision = 'APPROVE'
            decision_az = '✅ Təsdiq'
        elif proba < DECLINE_THRESHOLD:
            decision = 'REVIEW'
            decision_az = '⚠️ Nəzərdən keçir'
        else:
            decision = 'DECLINE'
            decision_az = '❌ İmtina'

        # SHAP
        top_factors = _get_shap_factors(df_te)

        # Expected Loss
        EL = proba * LGD * customer.amt_credit

        return {
            'default_probability' : round(float(proba), 4),
            'risk_score'          : int((1 - proba) * 1000),
            'decision'            : decision,
            'decision_az'         : decision_az,
            'gini'                : round(2 * float(proba) - 1, 4),
            'expected_loss_azn'   : round(EL, 2),
            'top_risk_factors'    : top_factors,
            'model_version'       : config['version'],
            'thresholds'          : config['thresholds']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        'model'             : config['model'],
        'version'           : config['version'],
        'auc_roc'           : config['auc_roc'],
        'gini'              : config['gini'],
        'feature_count'     : len(config['feature_names']),
        'imbalance_strategy': config['imbalance_strategy'],
        'thresholds'        : config['thresholds'],
        'cost_fn'           : config['cost_fn'],
        'cost_fp'           : config['cost_fp']
    }