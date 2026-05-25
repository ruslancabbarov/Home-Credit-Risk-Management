import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

# API URL — environment variable-dan al, yoxsa default
API_URL = os.getenv('API_URL', 'https://credit-risk-api-9f27.onrender.com')

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    color: white; text-align: center;
}
.approve-card {
    background: linear-gradient(135deg, #d5f5e3, #a9dfbf);
    border-left: 6px solid #27ae60; padding: 1.5rem;
    border-radius: 12px; margin: 1rem 0;
}
.review-card {
    background: linear-gradient(135deg, #fef9e7, #fdebd0);
    border-left: 6px solid #f39c12; padding: 1.5rem;
    border-radius: 12px; margin: 1rem 0;
}
.decline-card {
    background: linear-gradient(135deg, #fadbd8, #f5b7b1);
    border-left: 6px solid #e74c3c; padding: 1.5rem;
    border-radius: 12px; margin: 1rem 0;
}
.metric-card {
    background: white; border-radius: 10px;
    padding: 1rem; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Risk Assessment System</h1>
    <p>Bank Kredit Riski Qiymətləndirmə Sistemi</p>
</div>
""", unsafe_allow_html=True)

# ── API sağlamlıq yoxlaması ───────────────────────────────────
@st.cache_data(ttl=60)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json()
    except:
        return None

health = check_api()
if health:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status",  "✅ Aktiv")
    col2.metric("Model",   health.get('model', 'N/A'))
    col3.metric("AUC-ROC", health.get('auc', 'N/A'))
    col4.metric("Gini",    health.get('gini', 'N/A'))
else:
    st.error("❌ API əlaqəsi yoxdur!")
    st.stop()

st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank.png", width=80)
    st.title("Naviqasiya")
    st.markdown("---")
    st.markdown("**Model Məlumatları:**")
    st.markdown(f"- AUC-ROC: `{health.get('auc')}`")
    st.markdown(f"- Gini: `{health.get('gini')}`")
    st.markdown(f"- Versiya: `{health.get('version')}`")
    st.markdown("---")
    st.markdown("**Threshold Sistemi:**")
    st.markdown("- 🟢 `PD < 0.20` → Təsdiq")
    st.markdown("- 🟡 `0.20 ≤ PD < 0.50` → Review")
    st.markdown("- 🔴 `PD ≥ 0.50` → İmtina")
    st.markdown("---")
    st.caption("© 2026 Credit Risk System v1.0")

# ── Tab-lar ───────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔍 Mövcud Müştəri (SK_ID)",
    "✍️ Yeni Müştəri"
])

def show_result(result: dict, credit_amount: float):
    decision = result['decision']
    proba    = result['default_probability']
    score    = result['risk_score']
    EL       = result.get('expected_loss', proba * 0.45 * credit_amount)

    if decision == 'APPROVE':
        st.markdown("""
        <div class="approve-card">
            <h2>✅ KREDİT TƏSDİQ EDİLDİ</h2>
            <p>Müştəri aşağı risk qrupundadır.</p>
        </div>""", unsafe_allow_html=True)
    elif decision == 'REVIEW':
        st.markdown("""
        <div class="review-card">
            <h2>⚠️ ƏLAVƏ YOXLAMA LAZIMDIR</h2>
            <p>Kredit mütəxəssisi nəzərdən keçirməlidir.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="decline-card">
            <h2>❌ KREDİT İMTİNA EDİLDİ</h2>
            <p>Müştəri yüksək risk qrupundadır.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Risk Skoru",       f"{score}/1000")
    c2.metric("📊 Default Ehtimalı", f"{proba*100:.1f}%")
    c3.metric("📈 Gini",             f"{result.get('gini', 0):.4f}")
    c4.metric("💸 Gözlənilən İtki",  f"{EL:,.0f} AZN")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Ehtimalı (%)"},
        delta={'reference': 20},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#e74c3c" if proba > 0.5 else
                             "#f39c12" if proba > 0.2 else "#2ecc71"},
            'steps': [
                {'range': [0, 20],   'color': '#d5f5e3'},
                {'range': [20, 50],  'color': '#fef9e7'},
                {'range': [50, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💰 Gözlənilən İtki Hesablaması"):
        st.latex(r"EL = PD \times LGD \times EAD")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PD", f"{proba:.4f}")
        col2.metric("LGD", "45%")
        col3.metric("EAD", f"{credit_amount:,.0f} AZN")
        col4.metric("EL",  f"{EL:,.0f} AZN")

# ── TAB 1 ─────────────────────────────────────────────────────
with tab1:
    st.subheader("🔍 Mövcud Müştəri Qiymətləndirmə")
    st.info("SK_ID_CURR daxil edin — sistem büro məlumatlarını avtomatik çəkəcək.")

    col1, col2 = st.columns([3, 1])
    with col1:
        sk_id = st.number_input(
            "SK_ID_CURR daxil edin",
            min_value=100000, max_value=999999999,
            value=100002, step=1,
            help="Nümunə: 100002, 100003, 100004"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Qiymətləndir",
                               type="primary",
                               use_container_width=True)

    if search_btn:
        with st.spinner("Məlumatlar yüklənir..."):
            try:
                r = requests.get(
                    f"{API_URL}/predict/{sk_id}",
                    timeout=60
                )
                if r.status_code == 404:
                    st.error(f"❌ SK_ID={sk_id} tapılmadı!")
                    st.stop()
                elif r.status_code != 200:
                    st.error(f"Xəta: {r.text}")
                    st.stop()
                result = r.json()
            except Exception as e:
                st.error(f"API xətası: {e}")
                st.stop()

        with st.expander("👤 Müştəri Profili", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**SK_ID:** {result.get('sk_id_curr')}")
                st.write(f"**Yaş:** {result.get('age', 'N/A')}")
            with c2:
                st.write(f"**Gəlir:** {result.get('amt_income', 0):,.0f} AZN")
                st.write(f"**Kredit:** {result.get('amt_credit', 0):,.0f} AZN")
            with c3:
                st.write(f"**Risk Skoru:** {result.get('risk_score')}/1000")
                st.write(f"**Qərar:** {result.get('decision_az')}")

        show_result(result, result.get('amt_credit', 300000))

# ── TAB 2 ─────────────────────────────────────────────────────
with tab2:
    st.subheader("✍️ Yeni Müştəri Qiymətləndirmə")

    with st.form("new_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Şəxsi**")
            gender    = st.selectbox("Cins", ["F", "M"])
            age       = st.number_input("Yaş", 18, 80, 35)
            education = st.selectbox("Təhsil", [
                "Lower secondary",
                "Secondary / secondary special",
                "Incomplete higher",
                "Higher education",
                "Academic degree"
            ], index=1)
            family  = st.selectbox("Ailə vəziyyəti", [
                "Single / not married", "Married",
                "Civil marriage", "Separated", "Widow"
            ])
            cnt_fam = st.number_input("Ailə üzvü", 1, 20, 2)

        with col2:
            st.markdown("**💰 Maliyyə**")
            income   = st.number_input("Gəlir (AZN)", 10000, 5000000, 150000, 5000)
            credit   = st.number_input("Kredit (AZN)", 10000, 5000000, 300000, 10000)
            annuity  = st.number_input("Aylıq ödəniş (AZN)", 1000, 200000, 15000, 1000)
            contract = st.selectbox("Kredit növü", ["Cash loans", "Revolving loans"])
            own_car  = st.radio("Maşın?", ["N", "Y"], horizontal=True)
            own_house= st.radio("Əmlak?", ["Y", "N"], horizontal=True)

        with col3:
            st.markdown("**🏢 İş**")
            years_emp  = st.number_input("İş təcrübəsi (il)", 0.0, 50.0, 5.0)
            income_t   = st.selectbox("Gəlir növü", [
                "Working", "Commercial associate",
                "Pensioner", "State servant"
            ])
            occupation = st.selectbox("Peşə", [
                "Laborers", "Core staff", "Sales staff",
                "Managers", "Drivers", "Accountants",
                "High skill tech staff", "Medicine staff"
            ])
            org_type = st.selectbox("Təşkilat", [
                "Business Entity Type 3", "School",
                "Government", "Self-employed", "Medicine"
            ])

        st.markdown("**🔢 EXT_SOURCE**")
        c1, c2, c3 = st.columns(3)
        with c1: ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
        with c2: ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
        with c3: ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)

        submit = st.form_submit_button(
            "🔍 Kredit Riskini Qiymətləndir",
            type="primary", use_container_width=True
        )

    if submit:
        payload = {
            "name_contract_type" : contract,
            "code_gender"        : gender,
            "flag_own_car"       : own_car,
            "flag_own_realty"    : own_house,
            "amt_income_total"   : float(income),
            "amt_credit"         : float(credit),
            "amt_annuity"        : float(annuity),
            "name_education_type": education,
            "name_income_type"   : income_t,
            "name_family_status" : family,
            "name_housing_type"  : "House / apartment",
            "name_type_suite"    : "Unaccompanied",
            "occupation_type"    : occupation,
            "organization_type"  : org_type,
            "cnt_fam_members"    : float(cnt_fam),
            "age"                : float(age),
            "years_employed"     : float(years_emp),
            "ext_source_1"       : float(ext1),
            "ext_source_2"       : float(ext2),
            "ext_source_3"       : float(ext3),
        }

        with st.spinner("Qiymətləndirilir..."):
            try:
                r = requests.post(
                    f"{API_URL}/predict",
                    json=payload, timeout=60
                )
                if r.status_code != 200:
                    st.error(f"Xəta: {r.text}")
                    st.stop()
                result = r.json()
            except Exception as e:
                st.error(f"API xətası: {e}")
                st.stop()

        show_result(result, float(credit))