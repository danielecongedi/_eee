from io import BytesIO
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Bank Analytics", page_icon="🏦", layout="wide")
st.title("🏦 Bank Analytics")

# ── Costanti ──────────────────────────────────────────────────────────────────
NET_ORANGE = "#F28C28"
N_CLUSTERS_DEFAULT = 6
HEADER_SCAN_ROWS = 40

# ── Helpers: parsing ──────────────────────────────────────────────────────────

def parse_date(series: pd.Series) -> pd.Series:
    """Normalizza una colonna in datetime, gestendo formati misti (IT / ISO)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    s = (
        series.astype(str).str.strip()
        .str.replace(".", "/", regex=False)
        .str.replace("-", "/", regex=False)
    )
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def parse_amount(series: pd.Series) -> pd.Series:
    """
    Converte una colonna importo in float, gestendo:
    - simbolo €, spazi non-breaking, parentesi negative
    - separatori IT (1.234,56) e EN (1,234.56)
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = (
        series.astype(str).str.strip()
        .str.replace("€", "", regex=False)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )

    has_dot = s.str.contains(r"\.", regex=True)
    has_comma = s.str.contains(r",", regex=True)
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    # Entrambi i separatori → determina quale è il decimale
    both = has_dot & has_comma
    if both.any():
        sub = s[both]
        last_is_comma = sub.str.rfind(",") > sub.str.rfind(".")
        it = sub[last_is_comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        en = sub[~last_is_comma].str.replace(",", "", regex=False)
        out.loc[it.index] = pd.to_numeric(it, errors="coerce")
        out.loc[en.index] = pd.to_numeric(en, errors="coerce")

    if (m := has_comma & ~has_dot).any():
        out.loc[m[m].index] = pd.to_numeric(s[m].str.replace(",", ".", regex=False), errors="coerce")

    if (m := has_dot & ~has_comma).any():
        out.loc[m[m].index] = pd.to_numeric(s[m], errors="coerce")

    if (m := ~has_dot & ~has_comma).any():
        out.loc[m[m].index] = pd.to_numeric(s[m], errors="coerce")

    return out


def month_str(dt: date) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def month_add(ym: str, n: int) -> str:
    y, m = map(int, ym.split("-"))
    return month_str(date(y, m, 1) + relativedelta(months=n))


def infer_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Individua una colonna tramite lista di candidati (case-insensitive, partial match)."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in df.columns:
        if any(k in str(c).strip().lower() for k in candidates):
            return c
    return None

# ── Caricamento Excel ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_template_excel(file_bytes: bytes) -> tuple[pd.DataFrame, str, float | None]:
    """
    Legge un estratto conto Excel con header variabile.
    Restituisce (DataFrame, nome_foglio, saldo_contabile).
    """
    xls = pd.ExcelFile(BytesIO(file_bytes))
    sheet_name = xls.sheet_names[0]
    raw_head = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name,
                             header=None, nrows=HEADER_SCAN_ROWS)

    header_row = _find_header_row(raw_head)
    saldo_value = _extract_saldo(raw_head)

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name,
                       skiprows=header_row or 0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, na=False)]
    return df, sheet_name, saldo_value


def _find_header_row(raw_head: pd.DataFrame) -> int:
    for i, row in raw_head.iterrows():
        row_str = row.dropna().astype(str).str.strip()
        joined = " | ".join(row_str.str.lower())
        if "data" in joined and "importo" in joined:
            return int(i)
    return 0


def _extract_saldo(raw_head: pd.DataFrame) -> float | None:
    for _, row in raw_head.iterrows():
        if row.astype(str).str.lower().str.contains("saldo contabile").any():
            for v in row:
                if isinstance(v, (int, float)) and not pd.isna(v):
                    return float(v)
            for v in row:
                if isinstance(v, str) and v.strip():
                    vv = v.replace("€", "").strip()
                    vv = (vv.replace(".", "").replace(",", ".")
                          if ("," in vv and "." in vv) else vv.replace(",", "."))
                    try:
                        return float(vv)
                    except ValueError:
                        pass
    return None

# ── Categorizzazione (regole) ─────────────────────────────────────────────────

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Salute",                  ["movocchiali", "occhial", "ottica", "lenti", "visita", "studio medico",
                                  "medic", "dott", "clinica", "osped", "dent", "ticket", "farmac"]),
    ("Sushi",                   ["mova", "sushi"]),
    ("Netflix",                 ["netflix"]),
    ("Telefonia",               ["vodafone", "tim", "iliad", "wind", "fastweb"]),
    ("AI / ChatGPT",            ["chatgpt", "openai"]),
    ("Concerti / Eventi",       ["ticketone", "vivaticket", "ticketmaster"]),
    ("Viaggi",                  ["booking", "airbnb", "trenitalia", "italo", "ryanair",
                                  "easyjet", "marino", "flixbus", "hotel", "volo"]),
    ("Abbigliamento",           ["zalando", "zara", "h&m", "hm", "decathlon", "piazza italia",
                                  "nike", "adidas"]),
    ("Tecnologia",              ["mediaworld", "unieuro", "amazon", "apple", "huawei",
                                  "samsung", "iphone", "pc", "computer", "tablet"]),
    ("Casa / Cura persona",     ["acqua e sapone", "acqua&sapone", "tigot", "dm", "detersiv",
                                  "sapone", "shampoo", "profumer"]),
    ("Cibo / Spesa / Ristoranti", ["coop", "conad", "esselunga", "carrefour", "lidl", "aldi",
                                   "super", "market", "ristor", "trattor", "oster", "pizzer",
                                   "bar", "gelat", "pescher", "maceller", "forno", "gastronom",
                                   "glovo", "deliveroo", "just eat"]),
]


def categorize_spend_rule(text: str) -> str | None:
    if not text:
        return None
    d = str(text).strip().lower()
    for category, keywords in _CATEGORY_RULES:
        if any(k in d for k in keywords):
            return category
    return None

# ── Clustering (fallback su descrizioni non categorizzate) ────────────────────

_CLUSTER_TITLE_RULES: list[tuple[str, list[str]]] = [
    ("Ristorazione / Sushi",    ["sushi", "mova", "ristor", "pizzer", "bar", "trattor", "oster"]),
    ("Spesa / Supermercato",    ["coop", "conad", "esselunga", "carrefour", "lidl", "aldi", "super", "market"]),
    ("Telefonia",               ["vodafone", "tim", "iliad", "fastweb", "wind"]),
    ("Streaming",               ["netflix", "spotify", "disney", "prime"]),
    ("Concerti / Eventi",       ["ticketone", "vivaticket", "concert", "evento"]),
    ("Viaggi",                  ["booking", "airbnb", "trenitalia", "italo", "ryanair", "easyjet", "hotel", "volo"]),
    ("Salute",                  ["farmac", "medic", "dott", "visita", "ottica", "occhial", "lenti", "dent", "ticket"]),
    ("Abbigliamento / Sport",   ["zalando", "zara", "hm", "h&m", "decathlon", "nike", "adidas"]),
    ("Tecnologia",              ["mediaworld", "unieuro", "apple", "huawei", "samsung", "amazon", "pc", "iphone"]),
]


def pretty_cluster_title(top_terms: list[str]) -> str:
    joined = " ".join(top_terms).lower()
    for title, keywords in _CLUSTER_TITLE_RULES:
        if any(k in joined for k in keywords):
            return title
    top = [x for x in top_terms if x][:3]
    return "Altro – " + ", ".join(top) if top else "Altro"


def ai_cluster_unknowns_with_titles(
    unknown_desc: pd.Series,
    n_clusters: int = N_CLUSTERS_DEFAULT,
) -> tuple[pd.Series, pd.DataFrame]:
    texts = (
        unknown_desc.fillna("").astype(str).str.lower().str.strip()
        .str.replace(r"[^a-z0-9àèéìòù\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )
    fallback_labels = pd.Series(["Altro"] * len(texts), index=unknown_desc.index)
    fallback_info = pd.DataFrame([{"cluster_id": 0, "title": "Altro", "top_terms": ""}])

    if len(texts.unique()) < 3:
        return fallback_labels, fallback_info

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        fallback_info.at[0, "top_terms"] = "sklearn non installato"
        return fallback_labels, fallback_info

    k = max(2, min(n_clusters, len(texts.unique())))
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(texts.tolist())
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_id = km.fit_predict(X)

    feature_names = np.array(vect.get_feature_names_out())
    rows, cluster_titles = [], {}
    for cid in range(k):
        top_terms = feature_names[np.argsort(km.cluster_centers_[cid])[::-1][:6]].tolist()
        title = pretty_cluster_title(top_terms)
        cluster_titles[cid] = title
        rows.append({"cluster_id": cid, "title": title, "top_terms": ", ".join(top_terms)})

    labels = pd.Series([cluster_titles[int(c)] for c in cluster_id], index=unknown_desc.index)
    return labels, pd.DataFrame(rows).sort_values("cluster_id")

# ── Forecast entrate (shape da storico) ───────────────────────────────────────

def forecast_with_shape(series: pd.Series, months: pd.Series, horizon: int) -> list[float]:
    """Forecast con regressione lineare + stagionalità moltiplicativa."""
    df = (
        pd.DataFrame({"month": months.astype(str), "y": series.astype(float)})
        .assign(dt=lambda d: pd.to_datetime(d["month"] + "-01", errors="coerce"))
        .dropna(subset=["dt", "y"])
        .sort_values("dt")
        .reset_index(drop=True)
    )

    if len(df) < 3:
        base = float(df["y"].iloc[-1]) if len(df) else 0.0
        return [base] * horizon

    df["t"] = np.arange(len(df), dtype=float)
    a, b = np.polyfit(df["t"], df["y"], 1)
    trend = a * df["t"] + b
    season = (
        df.assign(moy=df["dt"].dt.month, ratio=(df["y"] / trend.replace(0, np.nan)))
        .groupby("moy")["ratio"].mean().fillna(1.0)
    )

    last_dt, last_t = df["dt"].iloc[-1], df["t"].iloc[-1]
    preds = []
    for i in range(1, horizon + 1):
        dt_f = last_dt + relativedelta(months=i)
        t_f = last_t + i
        s_f = float(season.get(dt_f.month, float(season.mean())))
        preds.append(max(0.0, (a * t_f + b) * s_f))
    return preds

# ═════════════════════════════════════════════════════════════════════════════
# UI – Upload
# ═════════════════════════════════════════════════════════════════════════════

uploaded = st.file_uploader(
    "Trascina qui l'Excel (drag & drop) oppure clicca per selezionare",
    type=["xlsx", "xls"],
)
if uploaded is None:
    st.info("Carica un file Excel per iniziare.")
    st.stop()

try:
    raw, sheet_name, saldo_extracted = load_template_excel(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore lettura Excel: {e}")
    st.stop()

# ── Rilevamento colonne ───────────────────────────────────────────────────────
c_date   = infer_col(raw, ["data valuta", "valuta", "data"])
c_desc   = infer_col(raw, ["descrizione", "causale", "merchant"])
c_detail = infer_col(raw, ["dettaglio", "details", "note"])
c_amount = infer_col(raw, ["importo", "amount", "valore"])

if not all([c_date, c_desc, c_amount]):
    st.error(
        "Non riesco a identificare le colonne chiave nel file.\n\n"
        f"Colonne trovate: {list(raw.columns)}"
    )
    st.stop()

# ── Preparazione DataFrame ────────────────────────────────────────────────────
df = raw.copy()
df["date"]        = parse_date(df[c_date])
df["amount"]      = parse_amount(df[c_amount])
df["description"] = df[c_desc].astype(str).str.strip()
df["detail"]      = df[c_detail].astype(str).str.strip() if c_detail else ""
df["text_all"]    = (df["description"].fillna("") + " " + df["detail"].fillna("")).str.strip()

df = (
    df.dropna(subset=["date", "amount"])
    .assign(date=lambda d: pd.to_datetime(d["date"]))
    .sort_values("date")
    .reset_index(drop=True)
)

min_d, max_d = df["date"].min().date(), df["date"].max().date()

# ── Filtro periodo + saldo ────────────────────────────────────────────────────
col_period, col_balance = st.columns([2, 1])
with col_period:
    period = st.date_input("Periodo analisi (da–a)", value=(min_d, max_d),
                           min_value=min_d, max_value=max_d)
    d_from, d_to = period if isinstance(period, tuple) else (min_d, max_d)
with col_balance:
    current_balance = st.number_input(
        "Saldo contabile attuale (€)",
        value=float(saldo_extracted) if saldo_extracted else 0.0,
        step=100.0,
    )

dfp = df[(df["date"].dt.date >= d_from) & (df["date"].dt.date <= d_to)].copy()
if dfp.empty:
    st.warning("Nel periodo selezionato non ci sono movimenti.")
    st.stop()

dfp["month"] = dfp["date"].dt.to_period("M").astype(str)

# ── Aggregati mensili ─────────────────────────────────────────────────────────
m = (
    dfp.groupby("month", as_index=False)
    .agg(
        income=("amount", lambda s: s[s > 0].sum()),
        expense=("amount", lambda s: -s[s < 0].sum()),
        net=("amount", "sum"),
        n_tx=("amount", "size"),
    )
    .sort_values("month")
    .reset_index(drop=True)
)

income_total  = dfp.loc[dfp["amount"] > 0, "amount"].sum()
expense_total = -dfp.loc[dfp["amount"] < 0, "amount"].sum()
net_total     = income_total - expense_total

avg_income_month  = float(m["income"].mean())  if len(m) else 0.0
avg_expense_month = float(m["expense"].mean()) if len(m) else 0.0
avg_net_month     = float(m["net"].mean())     if len(m) else 0.0

# ═════════════════════════════════════════════════════════════════════════════
# KPI
# ═════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
k1.metric("Saldo attuale",     f"€ {current_balance:,.2f}",
          "da file" if saldo_extracted else "manuale")
k2.metric("Entrate (periodo)", f"€ {income_total:,.2f}")
k3.metric("Uscite (periodo)",  f"€ {expense_total:,.2f}")
k4.metric("Netto (periodo)",   f"€ {net_total:,.2f}")

a1, a2, a3 = st.columns(3)
a1.metric("Entrate medie mensili", f"€ {avg_income_month:,.2f}")
a2.metric("Uscite medie mensili",  f"€ {avg_expense_month:,.2f}")
a3.metric("Netto medio mensile",   f"€ {avg_net_month:,.2f}")

st.caption(
    f"Foglio: **{sheet_name}** | "
    f"Data=`{c_date}` · Descrizione=`{c_desc}` · "
    f"Dettaglio=`{c_detail}` · Importo=`{c_amount}`"
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 1) Cumulata saldo
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Cumulata saldo nel periodo")

saldo_stimato = float(current_balance) - float(m["net"].sum())
if st.checkbox("Imposta manualmente saldo iniziale del periodo"):
    saldo_iniziale = st.number_input("Saldo iniziale periodo (€)", value=saldo_stimato, step=100.0)
else:
    saldo_iniziale = saldo_stimato

cum = m.copy()
cum["saldo_cumulato"] = saldo_iniziale + cum["net"].cumsum()

fig_cum = go.Figure(go.Scatter(
    x=cum["month"], y=cum["saldo_cumulato"],
    mode="lines+markers", name="Saldo cumulato",
    line=dict(color=NET_ORANGE, width=3),
))
fig_cum.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_cum, use_container_width=True)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 2) Entrate / Uscite / Netto mensili
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Entrate / Uscite mensili + Netto")
fig = go.Figure([
    go.Bar(x=m["month"], y=m["income"],  name="Entrate"),
    go.Bar(x=m["month"], y=m["expense"], name="Uscite"),
    go.Scatter(x=m["month"], y=m["net"], mode="lines+markers", name="Netto",
               line=dict(color=NET_ORANGE, width=3)),
])
fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# 3) Media mobile
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Media mobile entrate / uscite")

win = st.selectbox("Finestra media mobile (mesi)", [1, 3, 6, 12], index=1)
mm = m.copy()
mm["income_ma"]  = mm["income"].rolling(window=win, min_periods=1).mean()
mm["expense_ma"] = mm["expense"].rolling(window=win, min_periods=1).mean()

fig_ma = go.Figure([
    go.Scatter(x=mm["month"], y=mm["income_ma"],  mode="lines+markers", name=f"Entrate ({win}m)"),
    go.Scatter(x=mm["month"], y=mm["expense_ma"], mode="lines+markers", name=f"Uscite ({win}m)"),
])
fig_ma.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_ma, use_container_width=True)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 4) Spese per categoria
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Spese per tipologia (categorie)")

sp = dfp[dfp["amount"] < 0].copy()
if sp.empty:
    st.info("Nessuna spesa nel periodo.")
else:
    sp["expense_abs"] = -sp["amount"]
    sp["category"]    = sp["text_all"].apply(categorize_spend_rule)

    unknown_mask = sp["category"].isna()
    cluster_info = None
    if unknown_mask.any():
        labels, cluster_info = ai_cluster_unknowns_with_titles(
            sp.loc[unknown_mask, "text_all"], n_clusters=N_CLUSTERS_DEFAULT
        )
        sp.loc[unknown_mask, "category"] = labels

    cat_total = (
        sp.groupby("category", as_index=False)["expense_abs"]
        .sum()
        .sort_values("expense_abs", ascending=False)
        .reset_index(drop=True)
    )

    left, right = st.columns([2, 1])
    with left:
        fig_cat = go.Figure(go.Bar(
            x=cat_total["expense_abs"], y=cat_total["category"],
            orientation="h",
        ))
        fig_cat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)
    with right:
        st.dataframe(
            cat_total.rename(columns={"expense_abs": "Spesa (€)"})
            .assign(**{"Spesa (€)": lambda d: d["Spesa (€)"].round(2)}),
            use_container_width=True, hide_index=True,
        )

    if cluster_info is not None:
        with st.expander("Debug cluster (titoli + top termini)"):
            st.dataframe(cluster_info, use_container_width=True, hide_index=True)

    # Stacked mensile per categoria
    st.subheader("Uscite mensili per categoria (stacked)")
    piv = (
        sp.groupby(["month", "category"], as_index=False)["expense_abs"].sum()
        .pivot(index="month", columns="category", values="expense_abs")
        .fillna(0.0)
    )
    fig_stack = go.Figure(
        [go.Bar(x=piv.index, y=piv[c], name=str(c)) for c in piv.columns]
    )
    fig_stack.update_layout(barmode="stack", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_stack, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# 5) Simulazione saldo target
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Simulazione: mese di raggiungimento saldo target")

sA, sB, sC = st.columns(3)
with sA:
    target_balance    = st.number_input("Target saldo (€)", value=25_000.0, step=100.0)
with sB:
    max_monthly_expense = st.number_input("Spesa max mensile (€)", value=1_500.0, step=50.0)
with sC:
    horizon = st.selectbox("Orizzonte max (mesi)", [12, 24, 36, 48, 60], index=2)

last_month   = m["month"].iloc[-1]
months_future = [month_add(last_month, i) for i in range(1, int(horizon) + 1)]
income_fc    = forecast_with_shape(m["income"], m["month"], horizon=int(horizon))

balances, bal, reach_month = [], float(current_balance), None
for i, ym in enumerate(months_future):
    bal += income_fc[i] - float(max_monthly_expense)
    balances.append((ym, income_fc[i], float(max_monthly_expense), income_fc[i] - max_monthly_expense, bal))
    if reach_month is None and bal >= float(target_balance):
        reach_month = ym

sim_df = pd.DataFrame(balances, columns=["month", "income_fc", "expense_cap", "net_fc", "balance_fc"])

if reach_month:
    y, mo = map(int, reach_month.split("-"))
    st.success(f"Target € {target_balance:,.2f} raggiunto in **{datetime(y, mo, 1).strftime('%B %Y')}**.")
else:
    st.warning(f"Target € {target_balance:,.2f} non raggiunto entro {horizon} mesi.")

fig_sim = go.Figure(go.Scatter(
    x=sim_df["month"], y=sim_df["balance_fc"],
    mode="lines+markers", name="Saldo simulato",
    line=dict(color=NET_ORANGE, width=3),
))
fig_sim.add_hline(y=float(target_balance), line_dash="dash",
                  annotation_text="Target", annotation_position="top left")
fig_sim.update_layout(margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig_sim, use_container_width=True)

with st.expander("Dettaglio simulazione"):
    st.dataframe(
        sim_df.assign(**{c: sim_df[c].round(2) for c in sim_df.columns if c != "month"}),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Debug: movimenti filtrati
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("Movimenti filtrati (debug)"):
    show_cols = [c for c in ["date", "month", "description", "detail", "amount", "text_all"]
                 if c in dfp.columns]
    st.dataframe(dfp[show_cols].sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)
