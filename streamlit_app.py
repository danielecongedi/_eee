import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import re
import requests
from io import StringIO, BytesIO

st.set_page_config(page_title="Bank Spend Analytics", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 15px; }
      .small-note { opacity: 0.8; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helpers
# -------------------------
def _lower(s):
    return str(s).strip().lower()

def _normalize_colname(s: str) -> str:
    s = _lower(s)
    for ch in ["\n", "\t", "\r"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def parse_amount_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    ss = s.astype(str).str.strip()
    ss = ss.str.replace("€", "", regex=False).str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)
    ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    has_dot = ss.str.contains(r"\.", regex=True)
    has_comma = ss.str.contains(r",", regex=True)

    out = pd.Series([np.nan] * len(ss), index=ss.index, dtype="float64")

    both = has_dot & has_comma
    if both.any():
        sub = ss[both]
        last_comma_pos = sub.str.rfind(",")
        last_dot_pos = sub.str.rfind(".")
        it_mask = last_comma_pos > last_dot_pos

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
        sub = ss[neither]
        out.loc[sub.index] = pd.to_numeric(sub, errors="coerce")

    return out

def safe_date_parse(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def add_months(d: date, n: int) -> date:
    return (datetime(d.year, d.month, 1) + relativedelta(months=n)).date()

def guess_columns(df: pd.DataFrame):
    cols = list(df.columns)
    norm_map = {_normalize_colname(c): c for c in cols}

    def pick_any(substrings):
        for sub in substrings:
            for nc, orig in norm_map.items():
                if sub in nc:
                    return orig
        return None

    col_date = pick_any(["data contabile", "data", "date", "valuta", "data operazione", "data contabile"])
    col_desc = pick_any(["descrizione", "descr", "causal", "merchant", "esercente", "beneficiario", "ordinante", "dettagli"])
    col_balance = pick_any(["saldo", "balance", "disponibile", "saldo contabile", "saldo disponibile"])

    col_amount = pick_any(["importo", "amount", "movimento", "netto", "totale", "valore"])
    col_debit = pick_any(["addebito", "uscite", "dare", "debit"])
    col_credit = pick_any(["accredito", "entrate", "avere", "credit"])

    return col_date, col_amount, col_debit, col_credit, col_desc, col_balance

def build_amount_from_debit_credit(df: pd.DataFrame, col_debit: str, col_credit: str) -> pd.Series:
    d = parse_amount_series(df[col_debit]) if col_debit else pd.Series([np.nan]*len(df), index=df.index)
    c = parse_amount_series(df[col_credit]) if col_credit else pd.Series([np.nan]*len(df), index=df.index)
    return c.fillna(0.0) - d.fillna(0.0)

def dedupe_transactions(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["date", "amount", "description", "__source_file"]
    for k in key_cols:
        if k not in df.columns:
            df[k] = ""
    return df.drop_duplicates(subset=key_cols, keep="first").copy()

def extract_balance_from_text(text: str):
    """
    Supporta stringhe tipo:
    'saldo contabile al 26/02/2026 di euro +21039,34'
    """
    first_line = text.splitlines()[0] if text else ""
    m_val = re.search(r"([+-]?\d[\d\.]*,\d+)", first_line)
    m_date = re.search(r"(\d{2}/\d{2}/\d{4})", first_line)
    bal = None
    bal_date = None
    if m_val:
        bal = float(m_val.group(1).replace(".", "").replace(",", "."))
    if m_date:
        bal_date = datetime.strptime(m_date.group(1), "%d/%m/%Y").date()
    return bal, bal_date

def normalize_gsheet_csv_url(url: str) -> str:
    """
    Accetta anche URL sbagliati tipo .../pubhtml=csv e li normalizza.
    Formati supportati:
    - .../pubhtml  -> .../pub?output=csv
    - .../pubhtml?gid=0 -> .../pub?output=csv&gid=0
    - .../export?format=csv&gid=0 -> ok
    - .../pub?output=csv -> ok
    """
    u = url.strip()
    u = u.replace("pubhtml=csv", "pub?output=csv")
    u = u.replace("/pubhtml", "/pub?output=csv")
    # se è già export o pub?output=csv, ok
    if "output=csv" in u or "format=csv" in u:
        return u
    # fallback: prova aggiungere output=csv se è /pub
    if u.endswith("/pub"):
        return u + "?output=csv"
    return u

@st.cache_data(show_spinner=False)
def load_from_gsheet(csv_url: str) -> tuple[pd.DataFrame, list]:
    """
    Ritorna (raw_df, header_balances)
    header_balances: [(source, balance, date)]
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(csv_url, headers=headers, timeout=30)
    r.raise_for_status()
    text = r.text

    bal, bal_date = extract_balance_from_text(text)

    # Provo a leggere come "movimenti" con skiprows=2 e sep=";"
    # Se fallisce, fallback su lettura standard.
    try:
        df = pd.read_csv(StringIO(text), sep=";", skiprows=2, decimal=",")
    except Exception:
        df = pd.read_csv(StringIO(text))

    df["__sheet"] = "GoogleSheets"
    df["__source_file"] = "GoogleSheets"
    return df, [("GoogleSheets", bal, bal_date)]

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Impostazioni")

st.sidebar.subheader("📥 Fonte dati")
mode = st.sidebar.radio("Sorgente", ["Google Sheets (secrets)", "Upload file"], index=0)

sheet_mode = st.sidebar.radio("Sheet (solo Excel)", ["Tutti gli sheet", "Solo questo sheet"], index=0)
sheet_name = st.sidebar.text_input("Nome sheet (se 'Solo questo sheet')", value="")

st.sidebar.divider()
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

# -------------------------
# Load data
# -------------------------
st.title("🏦 Bank Spend Analytics")
st.caption("Sorgente: Google Sheets (via Secrets) oppure Upload. KPI, trend, forecast e massimale spesa per target.")

raw = pd.DataFrame()
header_balances = []

if mode == "Google Sheets (secrets)":
    if "CSV_URL" not in st.secrets:
        st.error("Manca CSV_URL nei Secrets. Aggiungi in Secrets: CSV_URL = \"...\"")
        st.stop()

    csv_url = normalize_gsheet_csv_url(st.secrets["CSV_URL"])
    try:
        raw, header_balances = load_from_gsheet(csv_url)
        st.sidebar.success("Google Sheets caricato ✅")
        st.sidebar.caption(csv_url)
    except Exception as e:
        st.error(f"Errore caricamento Google Sheets: {e}")
        st.stop()
else:
    uploads = st.sidebar.file_uploader(
        "Carica estratti conto (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True
    )
    if not uploads:
        st.info("Carica almeno un file per iniziare.")
        st.stop()

    frames = []
    for f in uploads:
        name = getattr(f, "name", "upload")
        suffix = name.lower().split(".")[-1]

        if suffix == "csv":
            file_bytes = f.getvalue()
            text = file_bytes.decode("utf-8-sig", errors="ignore")
            bal, bal_date = extract_balance_from_text(text)
            header_balances.append((name, bal, bal_date))

            df = pd.read_csv(BytesIO(file_bytes), sep=";", skiprows=2, decimal=",")
            df["__sheet"] = "CSV"
            df["__source_file"] = name
            frames.append(df)
        else:
            if sheet_mode == "Solo questo sheet" and sheet_name.strip():
                df = pd.read_excel(f, sheet_name=sheet_name.strip(), engine="openpyxl")
                df["__sheet"] = sheet_name.strip()
                df["__source_file"] = name
                frames.append(df)
            else:
                xls = pd.read_excel(f, sheet_name=None, engine="openpyxl")
                for sh, dfi in xls.items():
                    if dfi is None or len(dfi) == 0:
                        continue
                    dfi = dfi.copy()
                    dfi["__sheet"] = sh
                    dfi["__source_file"] = name
                    frames.append(dfi)

    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

if raw.empty:
    st.warning("Nessun dato caricato.")
    st.stop()

# -------------------------
# Mapping
# -------------------------
auto_date, auto_amount, auto_debit, auto_credit, auto_desc, auto_balance = guess_columns(raw)

st.subheader("Mapping colonne")
c1, c2, c3, c4 = st.columns(4)

col_date = c1.selectbox(
    "Colonna Data",
    options=list(raw.columns),
    index=list(raw.columns).index(auto_date) if auto_date in raw.columns else 0
)

amount_mode = c2.selectbox("Modalità importo", ["Importo unico", "Addebito/Accredito separati"], index=0 if auto_amount else 1)

if amount_mode == "Importo unico":
    col_amount = c2.selectbox(
        "Colonna Importo",
        options=list(raw.columns),
        index=list(raw.columns).index(auto_amount) if auto_amount in raw.columns else 0
    )
    col_debit = None
    col_credit = None
else:
    col_debit = c2.selectbox(
        "Colonna Addebito (uscite)",
        options=["(nessuna)"] + list(raw.columns),
        index=(1 + list(raw.columns).index(auto_debit)) if auto_debit in raw.columns else 0
    )
    col_credit = c2.selectbox(
        "Colonna Accredito (entrate)",
        options=["(nessuna)"] + list(raw.columns),
        index=(1 + list(raw.columns).index(auto_credit)) if auto_credit in raw.columns else 0
    )
    col_amount = None

col_desc = c3.selectbox(
    "Colonna Descrizione",
    options=list(raw.columns),
    index=list(raw.columns).index(auto_desc) if auto_desc in raw.columns else 0
)

col_balance = c4.selectbox(
    "Colonna Saldo (opz.)",
    options=["(nessuna)"] + list(raw.columns),
    index=(1 + list(raw.columns).index(auto_balance)) if auto_balance in raw.columns else 0
)

flip_sign = st.checkbox("⚠️ Inverti segno importi (se entrate risultano negative)", value=False)

# -------------------------
# Normalize
# -------------------------
df = raw.copy()
df["date_raw"] = df[col_date].astype(str).str.strip()
df["is_pending"] = df["date_raw"].str.lower().eq("non contabilizzato")
df["date"] = safe_date_parse(df[col_date])

if amount_mode == "Importo unico":
    df["amount"] = parse_amount_series(df[col_amount])
else:
    use_debit = None if col_debit == "(nessuna)" else col_debit
    use_credit = None if col_credit == "(nessuna)" else col_credit
    df["amount"] = build_amount_from_debit_credit(df, use_debit, use_credit)

df["description"] = df[col_desc].astype(str).str.strip()

if col_balance != "(nessuna)":
    df["balance"] = parse_amount_series(df[col_balance])
else:
    df["balance"] = np.nan

df = df.dropna(subset=["amount"]).copy()

if flip_sign:
    df["amount"] = -df["amount"]

df_main = df.copy()
if not include_pending:
    df_main = df_main[~df_main["is_pending"]].copy()

df_main = df_main.dropna(subset=["date"]).copy()
df_main["date"] = pd.to_datetime(df_main["date"])
df_main = df_main.sort_values("date").reset_index(drop=True)
df_main["month"] = df_main["date"].dt.to_period("M").astype(str)
df_main = dedupe_transactions(df_main)

# -------------------------
# KPI
# -------------------------
income = df_main.loc[df_main["amount"] > 0, "amount"].sum()
expense = -df_main.loc[df_main["amount"] < 0, "amount"].sum()
net = income - expense

balance_note = ""
if df_main["balance"].notna().any():
    dfx = df_main[df_main["balance"].notna()].copy()
    last_idx = dfx.sort_values("date").index[-1]
    current_balance = float(df_main.loc[last_idx, "balance"])
    balance_note = "da colonna Saldo"
else:
    # priorità a header saldo
    best = None
    for (src, bal, bdate) in header_balances:
        if bal is None:
            continue
        if best is None:
            best = (src, bal, bdate)
        else:
            # se ho date, prendo il più recente
            if bdate and best[2] and bdate > best[2]:
                best = (src, bal, bdate)
            elif bdate and not best[2]:
                best = (src, bal, bdate)
    if best:
        current_balance = float(best[1])
        balance_note = "da intestazione (saldo contabile)"
    else:
        current_balance = float(df_main["amount"].cumsum().iloc[-1]) if len(df_main) else 0.0
        balance_note = "calcolato (cumulata movimenti)"

m = df_main.groupby("month", as_index=False).agg(
    income=("amount", lambda s: s[s > 0].sum()),
    expense=("amount", lambda s: -s[s < 0].sum()),
    net=("amount", "sum"),
    n_tx=("amount", "size"),
)
m[["income", "expense", "net"]] = m[["income", "expense", "net"]].fillna(0.0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Saldo attuale", f"€ {current_balance:,.2f}", balance_note)
k2.metric("Entrate (periodo)", f"€ {income:,.2f}")
k3.metric("Uscite (periodo)", f"€ {expense:,.2f}")
k4.metric("Netto (periodo)", f"€ {net:,.2f}")

if (~include_pending) and df["is_pending"].any():
    st.caption(f"Pending esclusi: {int(df['is_pending'].sum())} movimenti 'Non contabilizzato'.")

st.markdown("---")

# -------------------------
# Charts
# -------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Entrate vs Uscite mensili")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["month"], y=m["income"], name="Entrate"))
    fig.add_trace(go.Bar(x=m["month"], y=m["expense"], name="Uscite"))
    fig.update_layout(barmode="group", font=dict(size=11), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Trend Netto mensile")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=m["month"], y=m["net"], mode="lines+markers", name="Netto"))
    fig2.update_layout(font=dict(size=11), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top descrizioni per spesa")
top = df_main[df_main["amount"] < 0].copy()
if len(top) == 0:
    st.info("Nessuna spesa nel periodo (dati filtrati o file vuoto).")
else:
    top["expense_abs"] = -top["amount"]
    topg = top.groupby("description", as_index=False)["expense_abs"].sum().sort_values("expense_abs", ascending=False).head(12)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=topg["expense_abs"], y=topg["description"], orientation="h", name="Spesa"))
    fig3.update_layout(font=dict(size=11), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# -------------------------
# Forecast
# -------------------------
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

if len(series) == 0:
    st.warning("Dati mensili insufficienti per forecast (dopo filtri).")
else:
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
    figf.update_layout(font=dict(size=11), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(figf, use_container_width=True)

# -------------------------
# Target & cap spesa
# -------------------------
st.subheader("Massimale ottimale di spesa media mensile per raggiungere il target")

current_net_worth = current_balance

if target_mode == "Target saldo a una data":
    today = date.today()
    months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
    months = max(1, months)

    expected_income_total = avg_income * months
    max_exp_total = current_net_worth + expected_income_total - target_balance
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

st.markdown("---")

# -------------------------
# Tables
# -------------------------
st.subheader("Movimenti (più recenti)")
view = df_main.sort_values("date", ascending=False).copy()
view["date"] = view["date"].dt.strftime("%Y-%m-%d")
view["amount"] = view["amount"].map(lambda x: f"{x:,.2f}")
cols = ["date", "amount", "description", "__source_file", "__sheet"]
cols = [c for c in cols if c in view.columns]
st.dataframe(view[cols].head(200), use_container_width=True, hide_index=True)

if df["is_pending"].any():
    st.subheader("Movimenti Non contabilizzati (pending)")
    pend = df[df["is_pending"]].copy()
    pend["amount"] = pend["amount"].map(lambda x: f"{x:,.2f}")
    pend_cols = [col_date, col_desc, "amount", "__source_file"]
    pend_cols = [c for c in pend_cols if c in pend.columns]
    st.dataframe(pend[pend_cols].head(200), use_container_width=True, hide_index=True)