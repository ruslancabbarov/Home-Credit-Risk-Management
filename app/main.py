from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib
import json
import os
import traceback
from sqlalchemy import create_engine, text

load_dotenv()

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Bank kredit riski qiymətləndirmə sistemi",
    version="1.0"
)

# ── Modelləri yüklə ───────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

te       = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.pkl'))
pipeline = joblib.load(os.path.join(MODELS_DIR, 'credit_pipeline.pkl'))

with open(os.path.join(MODELS_DIR, 'model_config.json')) as f:
    config = json.load(f)

APPROVE_THRESHOLD = config['thresholds']['approve']
DECLINE_THRESHOLD = config['thresholds']['decline']
LGD               = config['lgd']

# ── DB — lazy initialization ──────────────────────────────────
DB_URI = os.getenv('DB_URI')

if DB_URI and DB_URI.startswith('postgresql://'):
    DB_URI = DB_URI.replace('postgresql://', 'postgresql+psycopg2://', 1)

engine = None 

def get_engine():
    global engine
    if engine is None and DB_URI:
        engine = create_engine(DB_URI)
    return engine

# ── Sütun siyahısı ────────────────────────────────────────────
EXPECTED_COLS = [
    'flag_own_car', 'flag_own_realty',
    'name_education_type',
    'name_contract_type', 'code_gender', 'name_type_suite',
    'name_income_type', 'name_family_status', 'name_housing_type',
    'age_group', 'credit_segment',
    'occupation_type', 'organization_type',
    'ext_source_mean', 'credit_to_goods', 'ext_source_1',
    'amt_annuity', 'instalment_utilization_x', 'amt_credit',
    'ext_source_3', 'days_id_publish', 'ext_source_2',
    'max_payment_delay', 'debt_ratio__x', 'days_last_phone_change',
    'amt_credit_sum_sum_x', 'avg_instalment_y', 'income_to_credit',
    'total_down_payment_x', 'amt_annuity_mean_x', 'days_registration',
    'avg_payment_delay', 'last_instalment_day', 'asked_credit_ratio_mean_x',
    'days_decision_max_x', 'credit_gap_x', 'inst_count',
    'avg_cnt_payment_x', 'activity_span_x', 'refused_ratio_x',
    'avg_credit_x', 'pos_contract_count_x', 'days_decision_min_x',
    'days_decision_mean_x', 'amt_income_total', 'total_credit_x',
    'amt_credit_sum_debt_sum_x', 'months_balance_max_x',
    'is_credit_active_sum_x', 'own_car_age', 'amt_annuity_max_x',
    'type_consumer credit_sum_x', 'floorsmax_avg',
    'avg_instalment_x', 'approval_minus_refusal_x',
    'months_balance_min_x', 'max_payment_gap', 'hour_appr_process_start',
    'region_rating_client', 'down_payment_ratio_x', 'max_instalment_x',
    'approval_rate_x', 'type_mortgage_sum_x', 'cnt_fam_members',
    'elevators_avg', 'sk_id_bureau_count_x', 'amt_req_credit_bureau_qrt',
    'amt_credit_sum_overdue_sum_x', 'years_employed',
    'age', 'DAYS_EMPLOYED_ANOM',
]

def _predict_from_df(df: pd.DataFrame) -> dict:
    df    = df[[c for c in EXPECTED_COLS if c in df.columns]]
    df_te = te.transform(df)
    proba = float(pipeline.predict_proba(df_te)[0][1])

    if proba < APPROVE_THRESHOLD:
        decision, decision_az = 'APPROVE', '✅ Təsdiq'
    elif proba < DECLINE_THRESHOLD:
        decision, decision_az = 'REVIEW', '⚠️ Nəzərdən keçir'
    else:
        decision, decision_az = 'DECLINE', '❌ İmtina'

    return {
        'default_probability': round(proba, 4),
        'risk_score'         : int((1 - proba) * 1000),
        'decision'           : decision,
        'decision_az'        : decision_az,
        'gini'               : round(2 * proba - 1, 4),
    }

# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "model"  : config['model'],
        "version": config['version'],
        "auc"    : config['auc_roc'],
        "gini"   : config['gini']
    }

@app.get("/predict/{sk_id}")
def predict_by_id(sk_id: int):
    eng = get_engine()
    if not eng:
        raise HTTPException(status_code=500, detail="DB bağlantısı yoxdur")
    try:
        query = f'SELECT * FROM customers WHERE "SK_ID_CURR" = {sk_id}'
        df    = pd.read_sql(query, eng)

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"SK_ID={sk_id} tapılmadı"
            )

        result = _predict_from_df(df)
        result['sk_id_curr']    = sk_id
        result['amt_credit']    = float(df['amt_credit'].values[0])
        result['amt_income']    = float(df['amt_income_total'].values[0])
        result['age']           = int(df['age'].values[0])
        result['expected_loss'] = round(
            result['default_probability'] * LGD *
            float(df['amt_credit'].values[0]), 2
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class CustomerInput(BaseModel):
    name_contract_type  : str   = Field(default='Cash loans')
    code_gender         : str   = Field(default='M')
    flag_own_car        : str   = Field(default='N')
    flag_own_realty     : str   = Field(default='Y')
    amt_income_total    : float = Field(default=150000.0)
    amt_credit          : float = Field(default=300000.0)
    amt_annuity         : float = Field(default=15000.0)
    name_education_type : str   = Field(default='Secondary / secondary special')
    name_income_type    : str   = Field(default='Working')
    name_family_status  : str   = Field(default='Married')
    name_housing_type   : str   = Field(default='House / apartment')
    name_type_suite     : str   = Field(default='Unaccompanied')
    occupation_type     : str   = Field(default='Laborers')
    organization_type   : str   = Field(default='Business Entity Type 3')
    cnt_fam_members     : float = Field(default=2.0)
    age                 : float = Field(default=35.0)
    years_employed      : float = Field(default=5.0)
    ext_source_1                : Optional[float] = 0.5
    ext_source_2                : Optional[float] = 0.5
    ext_source_3                : Optional[float] = 0.5
    max_payment_delay           : Optional[float] = 0.0
    avg_payment_delay           : Optional[float] = 0.0
    sk_id_bureau_count_x        : Optional[float] = 0.0
    is_credit_active_sum_x      : Optional[float] = 0.0
    amt_credit_sum_debt_sum_x   : Optional[float] = 0.0
    amt_credit_sum_overdue_sum_x: Optional[float] = 0.0
    amt_req_credit_bureau_qrt   : Optional[float] = 0.0
    refused_ratio_x             : Optional[float] = 0.0
    approval_rate_x             : Optional[float] = 1.0
    inst_count                  : Optional[float] = 0.0
    instalment_utilization_x    : Optional[float] = 0.0
    debt_ratio__x               : Optional[float] = 0.0
    region_rating_client        : Optional[float] = 2.0

@app.post("/predict")
def predict_new(customer: CustomerInput):
    try:
        d = customer.dict()

        d['ext_source_mean']        = np.mean([d['ext_source_1'],
                                                d['ext_source_2'],
                                                d['ext_source_3']])
        d['income_to_credit']       = d['amt_income_total'] / max(d['amt_credit'], 1)
        d['credit_to_goods']        = 1.0
        d['days_id_publish']        = -2000.0
        d['days_last_phone_change'] = -500.0
        d['days_registration']      = -5000.0
        d['DAYS_EMPLOYED_ANOM']     = 0
        d['age_group']              = (
            'Young_High_Risk'        if d['age'] < 30 else
            'Middle_Age_Medium_Risk' if d['age'] < 50 else
            'Senior_Low_Risk'
        )
        d['credit_segment'] = 'Medium'

        defaults = {
            'avg_payment_gap'           : 0.0,
            'max_payment_gap'           : 0.0,
            'amt_credit_sum_sum_x'      : d.get('amt_credit_sum_debt_sum_x', 0),
            'avg_instalment_y'          : d['amt_annuity'],
            'total_down_payment_x'      : 0.0,
            'amt_annuity_mean_x'        : d['amt_annuity'],
            'last_instalment_day'       : -30.0,
            'asked_credit_ratio_mean_x' : 1.0,
            'days_decision_max_x'       : -100.0,
            'credit_gap_x'              : 0.0,
            'avg_cnt_payment_x'         : 12.0,
            'activity_span_x'           : 12.0,
            'avg_credit_x'              : d['amt_credit'],
            'pos_contract_count_x'      : 0.0,
            'days_decision_min_x'       : -500.0,
            'days_decision_mean_x'      : -300.0,
            'total_credit_x'            : d['amt_credit'],
            'months_balance_max_x'      : -1.0,
            'months_balance_min_x'      : -12.0,
            'own_car_age'               : 0.0,
            'amt_annuity_max_x'         : d['amt_annuity'],
            'type_consumer credit_sum_x': 0.0,
            'floorsmax_avg'             : 0.0,
            'avg_instalment_x'          : d['amt_annuity'],
            'approval_minus_refusal_x'  : 1.0,
            'hour_appr_process_start'   : 10.0,
            'down_payment_ratio_x'      : 0.0,
            'max_instalment_x'          : d['amt_annuity'],
            'type_mortgage_sum_x'       : 0.0,
            'elevators_avg'             : 0.0,
        }
        for col, val in defaults.items():
            if col not in d:
                d[col] = val

        df     = pd.DataFrame([d])
        result = _predict_from_df(df)
        result['expected_loss'] = round(
            result['default_probability'] * LGD * customer.amt_credit, 2
        )
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        'model'             : config['model'],
        'auc_roc'           : config['auc_roc'],
        'gini'              : config['gini'],
        'version'           : config['version'],
        'feature_count'     : len(config['feature_names']),
        'imbalance_strategy': config['imbalance_strategy'],
        'thresholds'        : config['thresholds'],
    }