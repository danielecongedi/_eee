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

# =========================
# Config fissa colonne (la tua banca)
# =========================
COL_DATE = "Data contabile"
COL_DESC = "Descrizione"
COL_AMOUNT = "Importo"

# =========================
# Helpers
# =========================
def safe_secrets() -> dict:
    try:
        s = st.secrets
        return dict(s) if isinstance(s, dict) else {}
    except Exception:
        return {}

def add_months(d: date, n: int) -> date:
    return (datetime(d.year, d.month, 1) + relativedelta(months=n)).date()

def safe_date_parse(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def parse_amount_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("€", "", regex=False).str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
    ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # formato IT: 1.234,56
    has_dot = ss.str.contains(r"\.", regex=True)
    has_comma = ss.str.contains(r",", regex=True)

    out = pd.Series(np.nan, index=ss.index, dtype="float64")

    both = has_dot & has_comma
    if both.any():
        sub = ss[both]
        last_comma = sub.str.rfind(",")
        last_dot = sub.str.rfind(".")
        it_mask = last_comma > last_dot

        it_vals = sub[it_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        en_vals = sub[~it_mask].str.replace(",", "", regex=False)

        out.loc[it_vals.index] = pd.to_numeric(it_vals, errors="coerce")
        out.loc[en_vals.index] = pd.to_numeric(en_vals, errors="coerce")

    only_comma = has_comma & ~has_dot
    if only_comma.any():
        sub = ss[only_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        out.loc[sub.index] = pd.to_numeric(sub, errors="coerce")

    only_dot = has_dot & ~has_comma
    if only_dot.any():
        sub = ss[only_dot].str.replace(",", "", regex=False)
        out.loc[sub.index] = pd.to_numeric(sub, errors="coerce")

    neither = ~has_dot & ~has_comma
    if neither.any():
        out.loc[ss[neither].index] = pd.to_numeric(ss[neither], errors="coerce")

    return out

def extract_balance_from_text(text: str):
    """
    Esempio:
    "saldo contabile al 26/02/2026 di euro +21039,34 EUR"
    """
    if not text:
        return None, None
    t = str(text).strip()
    m_val = re.search(r"([+-]?\d[\d\.]*,\d+)", t)
    m_date = re.search(r"(\d{2}/\d{2}/\d{4})", t)
    bal = None
    bal_date = None
    if m_val:
        bal = float(m_val.group(1).replace(".", "").replace(",", "."))
    if m_date:
        bal_date = datetime.strptime(m_date.group(1), "%d/%m/%Y").date()
    return bal, bal_date

def find_header_row(df_raw_noheader: pd.DataFrame) -> int:
    """
    Cerca la riga in cui compaiono le 3 intestazioni.
    """
    for i in range(min(50, len(df_raw_noheader))):
        row = df_raw_noheader.iloc[i].astype(str).str.strip().str.lower().tolist()
        joined = " | ".join(row)
        if ("data contabile" in joined) and ("descrizione" in joined) and ("importo" in joined):
            return i
    return -1

@st.cache_data(show_spinner=False)
def load_xlsx_table_and_balance(xlsx_url: str, sheet_name: str | None):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(xlsx_url, headers=headers, timeout=60)
    r.raise_for_status()
    bio = BytesIO(r.content)

    # 1) Leggi "grezzo" senza header per cercare saldo e riga intestazione
    raw0 = pd.read_excel(bio, sheet_name=(sheet_name if sheet_name else 0), header=None, engine="openpyxl")

    # saldo in A1 (0,0) tipicamente
    a1 = raw0.iloc[0, 0] if raw0.shape[0] > 0 and raw0.shape[1] > 0 else None
    balance, balance_date = extract_balance_from_text(a1)

    # trova riga header
    hdr = find_header_row(raw0)
    if hdr < 0:
        raise ValueError("Non trovo l'intestazione 'Data contabile / Descrizione / Importo' nel foglio.")

    # 2) rileggi da riga header come intestazione
    bio2 = BytesIO(r.content)  # reset stream
    df = pd.read_excel(
        bio2,
        sheet_name=(sheet_name if sheet_name else 0),
        header=hdr,
        engine="openpyxl",
    )

    return df, balance, balance_date

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Impostazioni")
include_pending = st.sidebar.checkbox("Includi 'Non contabilizzato' in KPI/Forecast", value=False)

forecast_horizon = st.sidebar.selectbox("⏳ Orizzonte forecast (mesi)", [3, 6, 9, 12], index=1)
forecast_method = st.sidebar.selectbox("Metodo forecast", ["Media ultimi N mesi", "Trend lineare"], index=0)
lookback_months = st.sidebar.selectbox("N mesi per media", [3, 6, 9, 12], index=1)

st.sidebar.divider()
target_mode = st.sidebar.selectbox("Modalità target", ["Target saldo a una data", "Target risparmio totale"], index=0)
if target_mode == "Target saldo a una data":
    target_balance = st.sidebar.number_input("🎯 Target saldo (€)", value=10000.0, step=100.0)
    target_date = st.sidebar.date_input("📅 Data target", value=add_months(date.today(), 6))
else:
    target_savings = st.sidebar.number_input("🎯 Target risparmio totale (€)", value=3000.0, step=100.0)
    months_to_target = st.sidebar.selectbox("📅 Entro quanti mesi", [3, 6, 9, 12], index=1)

# =========================
# Load ONLY Google Sheets XLSX
# =========================
st.title("🏦 Bank Spend Analytics")
st.caption("Fonte unica: Google Sheets → XLSX (via Secrets). Nessun upload file.")

sec = safe_secrets()
xlsx_url = sec.get("XLSX_URL")
sheet_name = (sec.get("SHEET_NAME") or "").strip()

if not xlsx_url:
    st.error('Manca XLSX_URL nei Secrets. Metti in `.streamlit/secrets.toml`:\n\nXLSX_URL = "https://.../pub?output=xlsx"')
    st.stop()

try:
    raw, header_balance, header_balance_date = load_xlsx_table_and_balance(xlsx_url, sheet_name if sheet_name else None)
except Exception as e:
    st.error(f"Errore caricamento XLSX: {e}")
    st.stop()

# check colonne attese
missing = [c for c in [COL_DATE, COL_DESC, COL_AMOUNT] if c not in raw.columns]
if missing:
    st.error(f"Nel foglio mancano colonne attese: {missing}. Colonne trovate: {list(raw.columns)}")
    st.stop()

st.sidebar.success("Google Sheet XLSX caricato ✅")

# =========================
# Normalize
# =========================
df = raw.copy()

df["date_raw"] = df[COL_DATE].astype(str).str.strip()
df["is_pending"] = df["date_raw"].str.lower().eq("non contabilizzato")

df["date"] = safe_date_parse(df[COL_DATE])
df["amount"] = parse_amount_series(df[COL_AMOUNT])
df["description"] = df[COL_DESC].astype(str).str.strip()

# pulizia
df = df.dropna(subset=["amount"]).copy()

df_main = df.copy()
if not include_pending:
    df_main = df_main[~df_main["is_pending"]].copy()

df_main = df_main.dropna(subset=["date"]).copy()
df_main["date"] = pd.to_datetime(df_main["date"])
df_main = df_main.sort_values("date").reset_index(drop=True)
df_main["month"] = df_main["date"].dt.to_period("M").astype(str)

if df_main.empty:
    st.warning("Dopo filtri (pending/date) non restano righe valide.")
    st.stop()

# =========================
# KPI
# =========================
income = df_main.loc[df_main["amount"] > 0, "amount"].sum()
expense = -df_main.loc[df_main["amount"] < 0, "amount"].sum()
net = income - expense

# saldo attuale: priorità saldo da intestazione, altrimenti cumulata
if header_balance is not None:
    current_balance = float(header_balance)
    note = f"da intestazione (saldo contabile {header_balance_date.strftime('%d/%m/%Y') if header_balance_date else ''})".strip()
else:
    current_balance = float(df_main["amount"].cumsum().iloc[-1])
    note = "calcolato (cumulata movimenti)"

m = df_main.groupby("month", as_index=False).agg(
    income=("amount", lambda s: s[s > 0].sum()),
    expense=("amount", lambda s: -s[s < 0].sum()),
    net=("amount", "sum"),
    n_tx=("amount", "size"),
)
m[["income", "expense", "net"]] = m[["income", "expense", "net"]].fillna(0.0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Saldo attuale", f"€ {current_balance:,.2f}", note)
k2.metric("Entrate (periodo)", f"€ {income:,.2f}")
k3.metric("Uscite (periodo)", f"€ {expense:,.2f}")
k4.metric("Netto (periodo)", f"€ {net:,.2f}")

st.markdown("---")

# =========================
# Charts
# =========================
left, right = st.columns([2, 1])

with left:
    st.subheader("Entrate vs Uscite mensili")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["month"], y=m["income"], name="Entrate"))
    fig.add_trace(go.Bar(x=m["month"], y=m["expense"], name="Uscite"))
    fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Trend Netto mensile")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=m["month"], y=m["net"], mode="lines+markers", name="Netto"))
    fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top descrizioni per spesa")
top = df_main[df_main["amount"] < 0].copy()
if len(top) == 0:
    st.info("Nessuna spesa nel periodo.")
else:
    top["expense_abs"] = -top["amount"]
    topg = top.groupby("description", as_index=False)["expense_abs"].sum().sort_values("expense_abs", ascending=False).head(12)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=topg["expense_abs"], y=topg["description"], orientation="h", name="Spesa"))
    fig3.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# =========================
# Forecast
# =========================
st.markdown("---")
st.subheader("Forecast trend e target")

series = m.sort_values("month").copy()
hist = series.tail(int(lookback_months)).copy() if len(series) >= 2 else series.copy()

def forecast_expense_monthly(method: str, horizon: int):
    if method == "Media ultimi N mesi":
        base = float(hist["expense"].mean()) if len(hist) else 0.0
        return [base for _ in range(horizon)]
    y = hist["expense"].values
    t = np.arange(len(y))
    if len(y) < 2:
        return [float(y[-1]) if len(y) else 0.0 for _ in range(horizon)]
    A = np.vstack([t, np.ones(len(t))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    t_future = np.arange(len(y), len(y) + horizon)
    preds = (a * t_future + b).tolist()
    return [max(0.0, float(p)) for p in preds]

pred_exp = forecast_expense_monthly(forecast_method, int(forecast_horizon))
avg_income = float(hist["income"].mean()) if len(hist) else 0.0
pred_inc = [avg_income for _ in range(int(forecast_horizon))]

last_month = series["month"].iloc[-1]
def month_label_from_str(ym: str, offset: int):
    y, mo = map(int, ym.split("-"))
    d0 = date(y, mo, 1)
    d1 = d0 + relativedelta(months=offset)
    return f"{d1.year:04d}-{d1.month:02d}"

future_months = [month_label_from_str(last_month, i + 1) for i in range(int(forecast_horizon))]
f = pd.DataFrame({"month": future_months, "income_fc": pred_inc, "expense_fc": pred_exp})
f["net_fc"] = f["income_fc"] - f["expense_fc"]

figf = go.Figure()
figf.add_trace(go.Scatter(x=series["month"], y=series["expense"], mode="lines+markers", name="Uscite (storico)"))
figf.add_trace(go.Scatter(x=f["month"], y=f["expense_fc"], mode="lines+markers", name="Uscite (forecast)", line=dict(dash="dash")))
figf.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(figf, use_container_width=True)

# =========================
# Target cap spesa
# =========================
st.subheader("Massimale ottimale di spesa media mensile per raggiungere il target")

if target_mode == "Target saldo a una data":
    today = date.today()
    months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
    months = max(1, months)

    expected_income_total = avg_income * months
    max_exp_total = current_balance + expected_income_total - target_balance
    max_exp_month = max(0.0, max_exp_total / months)

    st.write(f"Orizzonte: **{months} mesi** (fino a {target_date.strftime('%d %b %Y')})")
    st.write(f"Entrate medie mensili (ultimi {lookback_months} mesi): **€ {avg_income:,.2f}**")
    st.success(f"Per arrivare a saldo **€ {target_balance:,.2f}**, la spesa media mensile dovrebbe essere **≤ € {max_exp_month:,.2f}**")
else:
    months = int(months_to_target)
    expected_income_total = avg_income * months
    max_exp_total = expected_income_total - target_savings
    max_exp_month = max(0.0, max_exp_total / months)

    st.write(f"Orizzonte: **{months} mesi**")
    st.write(f"Entrate medie mensili (ultimi {lookback_months} mesi): **€ {avg_income:,.2f}**")
    st.success(f"Per risparmiare **€ {target_savings:,.2f}**, la spesa media mensile dovrebbe essere **≤ € {max_exp_month:,.2f}**")

# =========================
# Table
# =========================
st.markdown("---")
st.subheader("Movimenti (più recenti)")
view = df_main.sort_values("date", ascending=False).copy()
view["date"] = view["date"].dt.strftime("%Y-%m-%d")
view["amount"] = view["amount"].map(lambda x: f"{x:,.2f}")
st.dataframe(view[["date", "amount", "description"]].head(200), use_container_width=True, hide_index=True)