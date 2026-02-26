import re
from io import BytesIO
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Bank Spend Analytics", page_icon="🏦", layout="wide")

COL_DATE = "Data contabile"
COL_DESC = "Descrizione"
COL_AMOUNT = "Importo"

NET_ORANGE = "#F28C28"

# -------------------------
# Helpers
# -------------------------
def safe_secrets():
    try:
        return dict(st.secrets)
    except Exception:
        return {}

def add_months(d: date, n: int) -> date:
    return (datetime(d.year, d.month, 1) + relativedelta(months=n)).date()

def parse_amount(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace("€", "", regex=False).str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)

    has_dot = s.str.contains(r"\.", regex=True)
    has_comma = s.str.contains(r",", regex=True)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    both = has_dot & has_comma
    if both.any():
        sub = s[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        out.loc[sub.index] = pd.to_numeric(sub, errors="coerce")
    only_comma = has_comma & ~has_dot
    if only_comma.any():
        sub = s[only_comma].str.replace(",", ".", regex=False)
        out.loc[sub.index] = pd.to_numeric(sub, errors="coerce")
    only_dot = has_dot & ~has_comma
    if only_dot.any():
        out.loc[s[only_dot].index] = pd.to_numeric(s[only_dot], errors="coerce")
    neither = ~has_dot & ~has_comma
    if neither.any():
        out.loc[s[neither].index] = pd.to_numeric(s[neither], errors="coerce")
    return out

def parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "/", regex=False).str.replace("-", "/", regex=False)
    return pd.to_datetime(s, dayfirst=True, errors="coerce", infer_datetime_format=True)

@st.cache_data(show_spinner=False)
def load_xlsx(url: str, sheet_name: str | None = None) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(BytesIO(r.content), sheet_name=(sheet_name if sheet_name else 0), engine="openpyxl")

def normalize_col_lookup(cols):
    return {str(c).strip().lower(): c for c in cols}

# -------------------------
# “AI-like” categorizer without sklearn
# -------------------------
CATEGORY_KEYWORDS = {
    "Cibo / Spesa / Ristoranti": [
        "coop", "conad", "esselunga", "carrefour", "lidl", "aldi", "super", "market",
        "ristor", "pizzer", "bar", "sushi", "koya", "glovo", "deliveroo", "just eat",
        "mcdonald", "burger", "pescheria", "maceller", "panett", "forno"
    ],
    "Viaggi": [
        "booking", "airbnb", "trenitalia", "italo", "ryanair", "easyjet", "marino",
        "flixbus", "uber", "taxi", "hotel", "volo", "aero", "tren"
    ],
    "Bonifici / Trasferimenti": ["bonifico", "giroconto", "sepa", "transfer", "paypal"],
    "Prelievi bancomat": ["preliev", "atm", "bancomat", "contanti", "cash withdrawal"],
    "Abbigliamento / Sport": ["zalando", "zara", "h&m", "hm", "bershka", "pull&bear", "piazza italia", "decathlon", "nike", "adidas"],
    "Tecnologia": ["mediaworld", "unieuro", "amazon", "apple", "samsung", "huawei", "sony", "console", "pc", "computer", "tablet", "iphone"],
    "Casa / Bollette": ["enel", "a2a", "hera", "iren", "bollett", "gas", "luce", "acqua", "condominio", "affitto", "rent"],
    "Salute": ["farmac", "medic", "dent", "osped", "clinica", "ticket"],
}

def categorize_desc(desc: str) -> str:
    if desc is None:
        return "Altro"
    d = str(desc).strip().lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in d for k in keys):
            return cat
    return "Altro"

def forecast_balance_by_month(current_balance: float, avg_monthly_net: float, start_month: str, months_ahead: int):
    y, m = map(int, start_month.split("-"))
    d0 = date(y, m, 1)
    out = []
    bal = float(current_balance)
    for i in range(1, months_ahead + 1):
        di = d0 + relativedelta(months=i)
        bal += float(avg_monthly_net)
        out.append((f"{di.year:04d}-{di.month:02d}", bal))
    return out

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Impostazioni")
sec = safe_secrets()
xlsx_url = sec.get("XLSX_URL")
sheet_name = (sec.get("SHEET_NAME") or "").strip()

if not xlsx_url:
    st.error('Manca XLSX_URL nei Secrets. Metti in `.streamlit/secrets.toml`:\n\nXLSX_URL = "https://.../pub?output=xlsx"')
    st.stop()

st.sidebar.subheader("Saldo contabile attuale (manuale)")
current_balance = st.sidebar.number_input("Saldo (€)", value=21039.34, step=100.0)
include_pending = st.sidebar.checkbox("Includi 'Non contabilizzato'", value=False)

st.sidebar.divider()
st.sidebar.subheader("Simulazioni")
target_balance = st.sidebar.number_input("Target saldo (€)", value=25000.0, step=100.0)
target_month = st.sidebar.date_input("Oppure: mese target (stima saldo)", value=add_months(date.today(), 6))
max_monthly_expense_cap = st.sidebar.number_input("Cap spesa media mensile (€)", value=1500.0, step=50.0)

# =========================
# Load data
# =========================
st.title("🏦 Bank Spend Analytics")

try:
    raw = load_xlsx(xlsx_url, sheet_name if sheet_name else None)
except Exception as e:
    st.error(f"Errore caricamento Excel: {e}")
    st.stop()

if raw is None or raw.empty:
    st.warning("Il file Excel è vuoto.")
    st.stop()

cols_norm = normalize_col_lookup(raw.columns)
need = [COL_DATE, COL_DESC, COL_AMOUNT]
missing = [c for c in need if c.strip().lower() not in cols_norm]
if missing:
    st.error(f"Mancano colonne nel file: {missing}\nColonne trovate: {list(raw.columns)}")
    st.stop()

col_date = cols_norm[COL_DATE.strip().lower()]
col_desc = cols_norm[COL_DESC.strip().lower()]
col_amount = cols_norm[COL_AMOUNT.strip().lower()]

df = raw.copy()
df["date_raw"] = df[col_date].astype(str).str.strip()
df["is_pending"] = df["date_raw"].str.lower().eq("non contabilizzato")
df["date"] = parse_date(df[col_date])
df["amount"] = parse_amount(df[col_amount])
df["description"] = df[col_desc].astype(str).str.strip()

df = df.dropna(subset=["amount"]).copy()
if not include_pending:
    df = df[~df["is_pending"]].copy()
df = df.dropna(subset=["date"]).copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

if df.empty:
    st.warning("Nessun dato valido (date/importi non parsati).")
    st.write("DEBUG prime righe:", raw.head(10))
    st.stop()

# Periodo selezionabile
min_d = df["date"].min().date()
max_d = df["date"].max().date()
period = st.sidebar.date_input("Periodo analisi (da–a)", value=(min_d, max_d), min_value=min_d, max_value=max_d)
d_from, d_to = period if isinstance(period, tuple) else (min_d, max_d)

dfp = df[(df["date"].dt.date >= d_from) & (df["date"].dt.date <= d_to)].copy()
if dfp.empty:
    st.warning("Nel periodo selezionato non ci sono movimenti.")
    st.stop()

dfp["month"] = dfp["date"].dt.to_period("M").astype(str)

# KPI
income = dfp.loc[dfp["amount"] > 0, "amount"].sum()
expense = -dfp.loc[dfp["amount"] < 0, "amount"].sum()
net = income - expense

m = dfp.groupby("month", as_index=False).agg(
    income=("amount", lambda s: s[s > 0].sum()),
    expense=("amount", lambda s: -s[s < 0].sum()),
    net=("amount", "sum"),
    n_tx=("amount", "size"),
).sort_values("month").reset_index(drop=True)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Saldo attuale", f"€ {float(current_balance):,.2f}", "manuale")
k2.metric("Entrate (periodo)", f"€ {float(income):,.2f}")
k3.metric("Uscite (periodo)", f"€ {float(expense):,.2f}")
k4.metric("Netto (periodo)", f"€ {float(net):,.2f}")

st.caption(f"Periodo: {d_from.strftime('%d/%m/%Y')} → {d_to.strftime('%d/%m/%Y')}")
st.markdown("---")

# Grafico unico Entrate/Uscite + Netto arancione
st.subheader("Entrate / Uscite mensili + Trend Netto")
fig = go.Figure()
fig.add_trace(go.Bar(x=m["month"], y=m["income"], name="Entrate"))
fig.add_trace(go.Bar(x=m["month"], y=m["expense"], name="Uscite"))
fig.add_trace(go.Scatter(
    x=m["month"], y=m["net"], name="Netto",
    mode="lines+markers",
    line=dict(color=NET_ORANGE, width=3)
))
fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Tipologie spesa (regole)
st.subheader("Tipologie di spesa (clusterizzato)")
sp = dfp[dfp["amount"] < 0].copy()
if sp.empty:
    st.info("Nessuna spesa nel periodo.")
else:
    sp["expense_abs"] = -sp["amount"]
    sp["category"] = sp["description"].apply(categorize_desc)

    cat = sp.groupby("category", as_index=False)["expense_abs"].sum().sort_values("expense_abs", ascending=False)
    figc = go.Figure()
    figc.add_trace(go.Bar(x=cat["expense_abs"], y=cat["category"], orientation="h", name="Spesa"))
    figc.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figc, use_container_width=True)

    with st.expander("Dettaglio: top descrizioni per categoria"):
        for c in cat["category"].tolist():
            sub = sp[sp["category"] == c].groupby("description", as_index=False)["expense_abs"].sum().sort_values("expense_abs", ascending=False).head(8)
            st.write(f"**{c}**")
            st.write(", ".join([f"{r['description']} (€ {r['expense_abs']:.2f})" for _, r in sub.iterrows()]))

st.markdown("---")

# Simulazioni
st.subheader("Simulazioni: target saldo")

avg_monthly_net = float(m["net"].mean()) if len(m) else 0.0
avg_monthly_income = float(m["income"].mean()) if len(m) else 0.0

cA, cB = st.columns(2)
cA.metric("Netto medio mensile", f"€ {avg_monthly_net:,.2f}")
cB.metric("Entrate medie mensili", f"€ {avg_monthly_income:,.2f}")

last_month = m["month"].iloc[-1]

# target saldo -> mese raggiungimento
if avg_monthly_net <= 0:
    st.warning("Netto medio ≤ 0: target non raggiungibile senza cambiare spesa/entrate.")
else:
    delta = float(target_balance) - float(current_balance)
    if delta <= 0:
        st.success("Target già raggiunto.")
    else:
        months_needed = int(np.ceil(delta / avg_monthly_net))
        reach_date = (datetime.strptime(last_month + "-01", "%Y-%m-%d") + relativedelta(months=months_needed)).date()
        st.success(f"Target € {float(target_balance):,.2f} → stima: **{reach_date.strftime('%B %Y')}** (~ {months_needed} mesi)")

# mese target -> saldo atteso
tm = date(target_month.year, target_month.month, 1)
lm = datetime.strptime(last_month + "-01", "%Y-%m-%d").date()
months_ahead = (tm.year - lm.year) * 12 + (tm.month - lm.month)
if months_ahead > 0:
    saldo_atteso = float(current_balance) + months_ahead * avg_monthly_net
    st.info(f"Saldo atteso a **{tm.strftime('%B %Y')}**: **€ {saldo_atteso:,.2f}**")

# cap spesa -> entro quando target
assumed_net_with_cap = avg_monthly_income - float(max_monthly_expense_cap)
if assumed_net_with_cap <= 0:
    st.warning("Con il cap spesa impostato, netto stimato ≤ 0.")
else:
    delta2 = float(target_balance) - float(current_balance)
    if delta2 > 0:
        months_needed2 = int(np.ceil(delta2 / assumed_net_with_cap))
        reach_date2 = (datetime.strptime(last_month + "-01", "%Y-%m-%d") + relativedelta(months=months_needed2)).date()
        st.success(
            f"Con spesa media ≤ € {float(max_monthly_expense_cap):,.2f} "
            f"(netto stimato € {assumed_net_with_cap:,.2f}/mese) → target in **{reach_date2.strftime('%B %Y')}**"
        )

# proiezione saldo (grafico)
proj_months = st.slider("Proiezione per N mesi", 3, 36, 12, 3)
proj = forecast_balance_by_month(float(current_balance), float(avg_monthly_net), last_month, int(proj_months))
proj_df = pd.DataFrame(proj, columns=["month", "balance_fc"])

figb = go.Figure()
figb.add_trace(go.Scatter(
    x=proj_df["month"], y=proj_df["balance_fc"],
    mode="lines+markers",
    name="Saldo stimato",
    line=dict(color=NET_ORANGE, width=3)
))
figb.add_hline(y=float(target_balance), line_dash="dash", annotation_text="Target", annotation_position="top left")
figb.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(figb, use_container_width=True)