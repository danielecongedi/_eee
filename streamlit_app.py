from io import BytesIO
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Bank Spend Analytics", page_icon="🏦", layout="wide")

# =========================
# COLONNE ATTESE (nel file)
# =========================
COL_DATE = "Data valuta"
COL_DESC = "Descrizione"
COL_AMOUNT = "Importo"

NET_ORANGE = "#F28C28"

# =========================
# Helpers base
# =========================
def normalize_col_lookup(cols):
    return {str(c).strip().lower(): c for c in cols}

def parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "/", regex=False).str.replace("-", "/", regex=False)
    return pd.to_datetime(s, dayfirst=True, errors="coerce", infer_datetime_format=True)

def parse_amount(series: pd.Series) -> pd.Series:
    # robust IT/EN
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = s.str.replace("€", "", regex=False)
    s = s.str.replace("\u00A0", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    has_dot = s.str.contains(r"\.", regex=True)
    has_comma = s.str.contains(r",", regex=True)

    out = pd.Series(np.nan, index=s.index, dtype="float64")

    both = has_dot & has_comma
    if both.any():
        sub = s[both]
        last_is_comma = sub.str.rfind(",") > sub.str.rfind(".")
        it = sub[last_is_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        en = sub[~last_is_comma].str.replace(",", "", regex=False)
        out.loc[it.index] = pd.to_numeric(it, errors="coerce")
        out.loc[en.index] = pd.to_numeric(en, errors="coerce")

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

@st.cache_data(show_spinner=False)
def load_xlsx_from_upload(file_bytes: bytes, sheet_name: str | None = None) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=(sheet_name if sheet_name else 0), engine="openpyxl")

def month_str(dt: date) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"

def month_add(ym: str, n: int) -> str:
    y, m = map(int, ym.split("-"))
    d0 = date(y, m, 1)
    d1 = d0 + relativedelta(months=n)
    return month_str(d1)

# =========================
# Tipologie di SPESA (regole forti)
# =========================
def categorize_spend_rule(desc: str) -> str | None:
    if desc is None:
        return None
    d = str(desc).strip().lower()

    if any(k in d for k in ["movocchiali", "occhial", "ottica", "lenti", "visita", "studio medico", "medic", "dott", "clinica", "osped", "dent", "ticket", "farmac"]):
        return "Salute"
    if any(k in d for k in ["mova", "sushi"]):
        return "Sushi"
    if "netflix" in d:
        return "Netflix"
    if any(k in d for k in ["vodafone", "tim", "iliad", "wind", "fastweb"]):
        return "Telefonia"
    if any(k in d for k in ["chatgpt", "openai"]):
        return "AI / ChatGPT"
    if any(k in d for k in ["ticketone", "vivaticket", "ticketmaster"]):
        return "Concerti / Eventi"
    if any(k in d for k in ["booking", "airbnb", "trenitalia", "italo", "ryanair", "easyjet", "marino", "flixbus", "hotel", "volo"]):
        return "Viaggi"
    if any(k in d for k in ["zalando", "zara", "h&m", "hm", "decathlon", "piazza italia", "nike", "adidas"]):
        return "Abbigliamento"
    if any(k in d for k in ["mediaworld", "unieuro", "amazon", "apple", "huawei", "samsung", "iphone", "pc", "computer", "tablet"]):
        return "Tecnologia"
    if any(k in d for k in ["acqua e sapone", "acqua&sapone", "tigot", "dm", "detersiv", "sapone", "shampoo", "profumer"]):
        return "Casa / Cura persona"
    if any(k in d for k in ["coop", "conad", "esselunga", "carrefour", "lidl", "aldi", "super", "market", "ristor", "trattor", "oster", "pizzer", "bar", "gelat", "pescher", "maceller", "forno", "gastronom"]):
        return "Cibo / Spesa / Ristoranti"

    return None

# =========================
# Tipologie di PAGAMENTO (rule)
# =========================
def payment_type_rule(desc: str) -> str:
    d = ("" if desc is None else str(desc)).strip().lower()

    if any(k in d for k in ["preliev", "atm", "bancomat", "contanti"]):
        return "Prelievo"
    if any(k in d for k in ["bonifico", "giroconto", "sepa credit", "sepa trasfer"]):
        return "Bonifico"
    if any(k in d for k in ["sdd", "rid", "addebito diretto", "direct debit"]):
        return "Addebito diretto"
    if any(k in d for k in ["pagopa", "f24", "tribut", "impost", "tari"]):
        return "PagoPA / Tributi"
    if any(k in d for k in ["commission", "canone", "spese tenuta", "imposta di bollo", "bollo"]):
        return "Commissioni / Canoni"
    if any(k in d for k in ["pos", "carta", "pagamento", "mastercard", "visa"]):
        return "Carta"
    if any(k in d for k in ["srl", "spa", "ristor", "market", "super", "shop", "store", "amazon", "zalando", "vodafone", "netflix"]):
        return "Carta"
    return "Altro"

# =========================
# Clustering opzionale + titoli cluster
# =========================
def pretty_cluster_title(top_terms: list[str]) -> str:
    t = " ".join(top_terms).lower()
    if any(k in t for k in ["sushi", "mova", "ristor", "pizzer", "bar", "trattor", "oster"]):
        return "Ristorazione / Sushi"
    if any(k in t for k in ["coop", "conad", "esselunga", "carrefour", "lidl", "aldi", "super", "market"]):
        return "Spesa / Supermercato"
    if any(k in t for k in ["vodafone", "tim", "iliad", "fastweb", "wind"]):
        return "Telefonia"
    if any(k in t for k in ["netflix", "spotify", "disney", "prime"]):
        return "Streaming"
    if any(k in t for k in ["ticketone", "vivaticket", "concert", "evento"]):
        return "Concerti / Eventi"
    if any(k in t for k in ["booking", "airbnb", "trenitalia", "italo", "ryanair", "easyjet", "marino", "flixbus", "hotel", "volo"]):
        return "Viaggi"
    if any(k in t for k in ["farmac", "medic", "dott", "visita", "ottica", "occhial", "lenti", "dent", "ticket"]):
        return "Salute"
    if any(k in t for k in ["zalando", "zara", "hm", "h&m", "decathlon", "nike", "adidas", "piazza"]):
        return "Abbigliamento / Sport"
    if any(k in t for k in ["mediaworld", "unieuro", "apple", "huawei", "samsung", "amazon", "pc", "iphone"]):
        return "Tecnologia"
    top = [x for x in top_terms if x][:3]
    return "Altro – " + ", ".join(top) if top else "Altro"

def ai_cluster_unknowns_with_titles(unknown_desc: pd.Series, n_clusters: int = 6):
    texts = unknown_desc.fillna("").astype(str).str.lower().str.strip()
    texts = texts.str.replace(r"[^a-z0-9àèéìòù\s]", " ", regex=True)
    texts = texts.str.replace(r"\s+", " ", regex=True)

    uniq = texts.unique()
    if len(uniq) < 3:
        labels = pd.Series(["Altro"] * len(texts), index=unknown_desc.index)
        info = pd.DataFrame([{"cluster_id": 0, "title": "Altro", "top_terms": ""}])
        return labels, info

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
    except Exception:
        labels = pd.Series(["Altro"] * len(texts), index=unknown_desc.index)
        info = pd.DataFrame([{"cluster_id": 0, "title": "Altro", "top_terms": "sklearn non installato"}])
        return labels, info

    k = max(2, min(int(n_clusters), len(uniq)))
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(texts.tolist())

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_id = km.fit_predict(X)

    feature_names = np.array(vect.get_feature_names_out())
    centroids = km.cluster_centers_
    topn = 6

    cluster_titles = {}
    rows = []
    for cid in range(k):
        top_idx = np.argsort(centroids[cid])[::-1][:topn]
        top_terms = feature_names[top_idx].tolist()
        title = pretty_cluster_title(top_terms)
        cluster_titles[cid] = title
        rows.append({"cluster_id": cid, "title": title, "top_terms": ", ".join(top_terms)})

    labels = pd.Series([cluster_titles[int(c)] for c in cluster_id], index=unknown_desc.index)
    info = pd.DataFrame(rows).sort_values("cluster_id")
    return labels, info

# =========================
# Forecast entrate “shape”
# =========================
def forecast_with_shape(series: pd.Series, months: pd.Series, horizon: int) -> list[float]:
    df = pd.DataFrame({"month": months.astype(str), "y": series.astype(float)})
    df["dt"] = pd.to_datetime(df["month"] + "-01", errors="coerce")
    df = df.dropna(subset=["dt", "y"]).sort_values("dt")

    if len(df) < 3:
        base = float(df["y"].iloc[-1]) if len(df) else 0.0
        return [base for _ in range(horizon)]

    df["t"] = np.arange(len(df)).astype(float)
    a, b = np.polyfit(df["t"].values, df["y"].values, 1)
    df["trend"] = a * df["t"] + b
    df["trend"] = df["trend"].replace(0, np.nan)

    df["moy"] = df["dt"].dt.month
    df["ratio"] = (df["y"] / df["trend"]).replace([np.inf, -np.inf], np.nan)
    season = df.groupby("moy")["ratio"].mean().fillna(1.0)
    if season.isna().all():
        season = pd.Series({m: 1.0 for m in range(1, 13)})

    last_dt = df["dt"].iloc[-1]
    last_t = df["t"].iloc[-1]

    preds = []
    for i in range(1, horizon + 1):
        dt_f = last_dt + relativedelta(months=i)
        t_f = last_t + i
        trend_f = a * t_f + b
        s_f = float(season.get(dt_f.month, float(season.mean())))
        preds.append(max(0.0, float(trend_f) * s_f))
    return preds


# =========================
# UI: Upload obbligatorio
# =========================
st.title("🏦 Bank Spend Analytics (Excel upload)")

uploaded = st.file_uploader(
    "Trascina qui l'Excel (drag & drop) oppure clicca per selezionare",
    type=["xlsx", "xls"],
)

if uploaded is None:
    st.info("Carica un file Excel per iniziare.")
    st.stop()

try:
    raw = load_xlsx_from_upload(uploaded.getvalue(), sheet_name=None)
except Exception as e:
    st.error(f"Errore lettura Excel: {e}")
    st.stop()

# validazione colonne
cols_norm = normalize_col_lookup(raw.columns)
need = [COL_DATE, COL_DESC, COL_AMOUNT]
missing = [c for c in need if c.strip().lower() not in cols_norm]
if missing:
    st.error(f"Mancano colonne nel file: {missing}\nColonne trovate: {list(raw.columns)}")
    st.stop()

col_date = cols_norm[COL_DATE.lower()]
col_desc = cols_norm[COL_DESC.lower()]
col_amount = cols_norm[COL_AMOUNT.lower()]

# prepara df
df = raw.copy()
df["date"] = parse_date(df[col_date])
df["amount"] = parse_amount(df[col_amount])
df["description"] = df[col_desc].astype(str).str.strip()

df = df.dropna(subset=["date", "amount"]).copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

min_d = df["date"].min().date()
max_d = df["date"].max().date()

# TOP: periodo + saldo
topA, topB = st.columns([2, 1])
with topA:
    period = st.date_input(
        "Periodo analisi (da–a)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    d_from, d_to = period if isinstance(period, tuple) else (min_d, max_d)
with topB:
    current_balance = st.number_input("Saldo contabile attuale (€) (manuale)", value=0.0, step=100.0)

dfp = df[(df["date"].dt.date >= d_from) & (df["date"].dt.date <= d_to)].copy()
if dfp.empty:
    st.warning("Nel periodo selezionato non ci sono movimenti.")
    st.stop()

dfp["month"] = dfp["date"].dt.to_period("M").astype(str)

# Monthly aggregates
m = dfp.groupby("month", as_index=False).agg(
    income=("amount", lambda s: s[s > 0].sum()),
    expense=("amount", lambda s: -s[s < 0].sum()),  # positivo
    net=("amount", "sum"),
    n_tx=("amount", "size"),
).sort_values("month").reset_index(drop=True)

income_total = dfp.loc[dfp["amount"] > 0, "amount"].sum()
expense_total = -dfp.loc[dfp["amount"] < 0, "amount"].sum()
net_total = income_total - expense_total

avg_income_month = float(m["income"].mean()) if len(m) else 0.0
avg_expense_month = float(m["expense"].mean()) if len(m) else 0.0
avg_net_month = float(m["net"].mean()) if len(m) else 0.0

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Saldo attuale", f"€ {float(current_balance):,.2f}", "manuale")
k2.metric("Entrate (periodo)", f"€ {float(income_total):,.2f}")
k3.metric("Uscite (periodo)", f"€ {float(expense_total):,.2f}")
k4.metric("Netto (periodo)", f"€ {float(net_total):,.2f}")

a1, a2, a3 = st.columns(3)
a1.metric("Entrate medie mensili", f"€ {avg_income_month:,.2f}")
a2.metric("Uscite medie mensili", f"€ {avg_expense_month:,.2f}")
a3.metric("Netto medio mensile", f"€ {avg_net_month:,.2f}")

st.markdown("---")

# =========================
# 1) CUMULATA
# =========================
st.subheader("Cumulata saldo nel periodo (da movimenti)")

# Se current_balance è 0 (default), la cumulata sarà “relativa”
saldo_iniziale_stimato = float(current_balance) - float(m["net"].sum())
use_manual_start = st.checkbox("Imposta manualmente saldo iniziale del periodo", value=False)
if use_manual_start:
    saldo_iniziale = st.number_input("Saldo iniziale periodo (€)", value=saldo_iniziale_stimato, step=100.0)
else:
    saldo_iniziale = saldo_iniziale_stimato

cum = m.copy()
cum["saldo_cumulato"] = float(saldo_iniziale) + cum["net"].cumsum()

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum["month"], y=cum["saldo_cumulato"],
    mode="lines+markers",
    name="Saldo cumulato",
    line=dict(color=NET_ORANGE, width=3)
))
fig_cum.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("---")

# =========================
# 2) ENTRATE/USCITE/NETTO
# =========================
st.subheader("Entrate / Uscite mensili + Netto")
fig = go.Figure()
fig.add_trace(go.Bar(x=m["month"], y=m["income"], name="Entrate"))
fig.add_trace(go.Bar(x=m["month"], y=m["expense"], name="Uscite"))
fig.add_trace(go.Scatter(
    x=m["month"], y=m["net"],
    mode="lines+markers",
    name="Netto",
    line=dict(color=NET_ORANGE, width=3)
))
fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# =========================
# 3) MEDIE MOBILI
# =========================
st.markdown("---")
st.subheader("Entrate/Uscite medie: come cambiano nei mesi (media mobile)")

win = st.selectbox("Finestra media mobile (mesi)", [1, 3, 6, 12], index=1)
mm = m.copy()
mm["income_ma"] = mm["income"].rolling(window=win, min_periods=1).mean()
mm["expense_ma"] = mm["expense"].rolling(window=win, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=mm["month"], y=mm["income_ma"], mode="lines+markers", name=f"Entrate medie ({win}m)"))
fig_ma.add_trace(go.Scatter(x=mm["month"], y=mm["expense_ma"], mode="lines+markers", name=f"Uscite medie ({win}m)"))
fig_ma.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_ma, use_container_width=True)

st.markdown("---")

# =========================
# TIPI DI PAGAMENTO
# =========================
st.subheader("Tipologie di pagamento")

dfp["payment_type"] = dfp["description"].apply(payment_type_rule)

pay = dfp[dfp["amount"] < 0].copy()
if pay.empty:
    st.info("Nessuna uscita nel periodo per tipologie di pagamento.")
else:
    pay["expense_abs"] = -pay["amount"]
    pay_tot = pay.groupby("payment_type", as_index=False)["expense_abs"].sum().sort_values("expense_abs", ascending=False)

    cL, cR = st.columns([2, 1])
    with cL:
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Bar(x=pay_tot["expense_abs"], y=pay_tot["payment_type"], orientation="h", name="Spesa"))
        fig_pay.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pay, use_container_width=True)
    with cR:
        t = pay_tot.copy()
        t["Spesa (€)"] = t["expense_abs"].round(2)
        t = t.drop(columns=["expense_abs"])
        st.dataframe(t, use_container_width=True, hide_index=True)

    st.subheader("Uscite mensili per tipologia di pagamento")
    pay_month = pay.groupby(["month", "payment_type"], as_index=False)["expense_abs"].sum()
    pay_piv = pay_month.pivot(index="month", columns="payment_type", values="expense_abs").fillna(0.0)

    fig_pay_stack = go.Figure()
    for c in pay_piv.columns:
        fig_pay_stack.add_trace(go.Bar(x=pay_piv.index, y=pay_piv[c], name=str(c)))
    fig_pay_stack.update_layout(barmode="stack", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_pay_stack, use_container_width=True)

st.markdown("---")

# =========================
# SPESE per CATEGORIA (regole + clustering opzionale)
# =========================
st.subheader("Spese per tipologia (categorie)")

sp = dfp[dfp["amount"] < 0].copy()
if sp.empty:
    st.info("Nessuna spesa nel periodo.")
else:
    sp["expense_abs"] = -sp["amount"]
    sp["category"] = sp["description"].apply(categorize_spend_rule)

    unknown_mask = sp["category"].isna()
    cluster_info = None
    if unknown_mask.any():
        labels, cluster_info = ai_cluster_unknowns_with_titles(sp.loc[unknown_mask, "description"], n_clusters=6)
        sp.loc[unknown_mask, "category"] = labels

    cat_total = (
        sp.groupby("category", as_index=False)["expense_abs"]
        .sum()
        .sort_values("expense_abs", ascending=False)
        .reset_index(drop=True)
    )

    left, right = st.columns([2, 1])
    with left:
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(x=cat_total["expense_abs"], y=cat_total["category"], orientation="h", name="Spesa"))
        fig_cat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)
    with right:
        t = cat_total.copy()
        t["Spesa (€)"] = t["expense_abs"].round(2)
        t = t.drop(columns=["expense_abs"])
        st.dataframe(t, use_container_width=True, hide_index=True)

    if cluster_info is not None:
        with st.expander("Debug cluster (titoli + top termini)"):
            st.dataframe(cluster_info, use_container_width=True, hide_index=True)

    st.subheader("Uscite mensili per categoria (stacked)")
    cat_month = sp.groupby(["month", "category"], as_index=False)["expense_abs"].sum()
    piv = cat_month.pivot(index="month", columns="category", values="expense_abs").fillna(0.0)

    fig_stack = go.Figure()
    for c in piv.columns:
        fig_stack.add_trace(go.Bar(x=piv.index, y=piv[c], name=str(c)))
    fig_stack.update_layout(barmode="stack", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_stack, use_container_width=True)

    st.subheader("Totale periodo per categoria")
    fig_total = go.Figure()
    fig_total.add_trace(go.Bar(x=cat_total["expense_abs"], y=cat_total["category"], orientation="h"))
    fig_total.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_total, use_container_width=True)

st.markdown("---")

# =========================
# SIMULAZIONE: target saldo con shape storico entrate
# =========================
st.subheader("Simulazione: mese di raggiungimento saldo target (forecast con shape da storico)")

simA, simB, simC = st.columns(3)
with simA:
    target_balance = st.number_input("Target saldo (€)", value=25000.0, step=100.0)
with simB:
    max_monthly_expense = st.number_input("Spesa max mensile (€)", value=1500.0, step=50.0)
with simC:
    horizon = st.selectbox("Orizzonte max (mesi)", [12, 24, 36, 48, 60], index=2)

last_month = m["month"].iloc[-1]
months_future = [month_add(last_month, i) for i in range(1, int(horizon) + 1)]

income_fc = forecast_with_shape(m["income"], m["month"], horizon=int(horizon))

balances = []
bal = float(current_balance)
reach_month = None

for i, ym in enumerate(months_future):
    inc = float(income_fc[i])
    exp = float(max_monthly_expense)
    netm = inc - exp
    bal = bal + netm
    balances.append((ym, inc, exp, netm, bal))
    if reach_month is None and bal >= float(target_balance):
        reach_month = ym

sim_df = pd.DataFrame(balances, columns=["month", "income_fc", "expense_cap", "net_fc", "balance_fc"])

if reach_month is None:
    st.warning(f"Non raggiungi € {float(target_balance):,.2f} entro {horizon} mesi (entrate forecast con shape storico, spesa cap impostata).")
else:
    y, mo = map(int, reach_month.split("-"))
    st.success(f"Target € {float(target_balance):,.2f} raggiunto in **{datetime(y, mo, 1).strftime('%B %Y')}** (simulazione).")

fig_sim = go.Figure()
fig_sim.add_trace(go.Scatter(
    x=sim_df["month"], y=sim_df["balance_fc"],
    mode="lines+markers",
    name="Saldo simulato",
    line=dict(color=NET_ORANGE, width=3)
))
fig_sim.add_hline(y=float(target_balance), line_dash="dash", annotation_text="Target", annotation_position="top left")
fig_sim.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_sim, use_container_width=True)

with st.expander("Dettaglio simulazione"):
    tmp = sim_df.copy()
    for c in ["income_fc", "expense_cap", "net_fc", "balance_fc"]:
        tmp[c] = tmp[c].round(2)
    st.dataframe(tmp, use_container_width=True, hide_index=True)

st.markdown("---")

with st.expander("Movimenti filtrati (debug)"):
    st.dataframe(
        dfp[["date", "month", "description", "amount", "payment_type"]].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True
    )