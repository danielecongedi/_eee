import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

# sklearn opzionale (clustering leggero)
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard Spese", layout="wide", page_icon="💳")
st.title("💳 Dashboard Bank")

MIN_ROWS_FOR_CLUSTER = 30

MACRO_TAGS_BASE = [
    "Viaggi", "Cibo", "Casa", "Auto/Trasporti", "Abbonamenti",
    "Shopping", "Salute", "Svago", "Commissioni/Banca", "Entrate", "Altro"
]

PAYMENT_TYPES = [
    "Carta", "Bonifico", "Addebito diretto", "Prelievo", "PagoPA-F24", "Commissioni", "Altro"
]

# =========================
# HELPERS
# =========================
def norm_text(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def infer_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in candidates):
            return c
    return None

def parse_amount(series: pd.Series) -> pd.Series:
    """
    Parsing robusto IT/EN senza “sproporzioni”.
    - numerico -> usa diretto
    - stringa:
      * 1.234,56 -> 1234.56
      * 1234,56  -> 1234.56
      * 1234.56  -> 1234.56 (non rimuovere punti)
      * (123,45) -> -123.45
    """
    def clean_value(val):
        if pd.isna(val):
            return np.nan

        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        s = str(val).strip()
        s = s.replace("€", "").replace("\u00a0", "").replace(" ", "")

        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        if "," in s and "." in s:
            # se la virgola è dopo il punto -> formato IT (1.234,56)
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                # formato EN con migliaia 1,234.56
                s = s.replace(",", "")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        else:
            pass

        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(clean_value)

@st.cache_data(show_spinner=False)
def load_first_sheet(file_bytes: bytes):
    xls = pd.ExcelFile(file_bytes)
    first_sheet = xls.sheet_names[0]

    raw_head = pd.read_excel(file_bytes, sheet_name=first_sheet, header=None, nrows=25)

    saldo = None
    header_row = 0

    for idx, row in raw_head.iterrows():
        row_str_lower = [str(x).lower().strip() for x in row.dropna().tolist()]

        # saldo
        if any("saldo" in val for val in row_str_lower) and saldo is None:
            for val in row.dropna().tolist():
                if isinstance(val, (int, float)):
                    saldo = float(val)
                    break
                elif isinstance(val, str):
                    tmp = parse_amount(pd.Series([val])).iloc[0]
                    if not pd.isna(tmp):
                        saldo = float(tmp)
                        break

        # header
        if any("importo" in val for val in row_str_lower) and any("data" in val for val in row_str_lower):
            header_row = idx
            break

    df = pd.read_excel(file_bytes, sheet_name=first_sheet, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
    return df, first_sheet, saldo

def month_key(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.to_period("M").astype(str)

# =========================
# CATEGORIE (REGole forti)
# =========================
CATEGORY_RULES = {
    "Cibo": [
        "bar", "rist", "pizz", "burger", "caff", "supermerc", "coop", "esselunga", "conad",
        "carrefour", "glovo", "deliveroo", "justeat", "just eat", "pam", "lidl", "md"
    ],
    "Casa": [
        "affitto", "condominio", "mutuo", "enel", "a2a", "iren", "hera", "tari",
        "luce", "gas", "acqua", "internet", "fibra", "tim", "vodafone", "wind", "fastweb"
    ],
    "Auto/Trasporti": [
        "benz", "diesel", "eni", "q8", "esso", "ip", "shell",
        "autostr", "telepass", "tren", "tram", "metro", "taxi", "uber",
        "italo", "trenitalia", "atm", "atac", "parchegg", "sosta"
    ],
    "Abbonamenti": [
        "netflix", "spotify", "prime", "amazon prime", "disney", "abbon", "subscription",
        "apple", "icloud", "google one"
    ],
    "Shopping": [
        "amazon", "ikea", "zara", "h&m", "hm", "decathlon", "mediaworld", "unieuro",
        "store", "negozio", "shopping"
    ],
    "Salute": [
        "farm", "medic", "ticket", "dent", "osped", "clinic", "analisi", "visita", "sanit"
    ],
    "Svago": [
        "cinema", "teatro", "concerto", "pub", "aperi", "discoteca", "evento", "museo"
    ],
    "Commissioni/Banca": [
        "commission", "canone", "imposta", "bollo", "spese", "fee", "prelievo", "carta",
        "interessi", "costi"
    ],
    "Viaggi": [
        "hotel", "airbnb", "flight", "ryanair", "easyjet", "booking", "expedia", "noleggio",
        "volo", "aeroporto", "traghetto"
    ],
}

def tag_category_rules(desc: str, amount: float) -> str:
    if amount is not None and amount > 0:
        return "Entrate"
    dl = (desc or "").lower()
    for cat, keys in CATEGORY_RULES.items():
        if any(k in dl for k in keys):
            return cat
    return "Altro"

# =========================
# PAGAMENTI (REGole + colonna se esiste)
# =========================
PAYMENT_RULES = {
    "PagoPA-F24": ["pagopa", "f24", "f23", "rav", "mav"],
    "Addebito diretto": ["sdd", "addebito", "rid", "direct debit", "domicilia"],
    "Bonifico": ["bonifico", "giroconto", "sepa credit transfer", "sct"],
    "Prelievo": ["preliev", "atm", "bancomat"],
    "Commissioni": ["commission", "canone", "bollo", "spese", "fee", "interessi"],
    "Carta": ["carta", "pos", "pagamento carta", "visa", "mastercard", "amex", "contactless"],
}

def tag_payment(desc: str, raw_payment: str | None) -> str:
    # se c'è una colonna tipo "Metodo" o "Tipo operazione", proviamo a mapparla
    if raw_payment:
        rp = raw_payment.lower()
        if any(k in rp for k in ["pagopa", "f24", "mav", "rav"]):
            return "PagoPA-F24"
        if any(k in rp for k in ["sdd", "addebito", "rid", "domicilia"]):
            return "Addebito diretto"
        if "bonific" in rp or "sct" in rp:
            return "Bonifico"
        if any(k in rp for k in ["preliev", "atm", "bancomat"]):
            return "Prelievo"
        if any(k in rp for k in ["commission", "canone", "bollo", "spese", "fee"]):
            return "Commissioni"
        if any(k in rp for k in ["carta", "pos", "visa", "mastercard", "amex"]):
            return "Carta"

    dl = (desc or "").lower()
    for ptype, keys in PAYMENT_RULES.items():
        if any(k in dl for k in keys):
            return ptype
    return "Altro"

# =========================
# CLUSTERING (sklearn opzionale) + TITOLI CLUSTER
# =========================
def build_tfidf(desc_list):
    vect = TfidfVectorizer(min_df=1, max_features=6000, ngram_range=(1, 2))
    X = vect.fit_transform(desc_list)
    Xn = normalize(X)
    return Xn, vect

def kmeans_labels(Xn, desired_k=12):
    n = Xn.shape[0]
    if n < MIN_ROWS_FOR_CLUSTER:
        return np.zeros(n, dtype=int)
    k = int(np.clip(desired_k, 2, max(2, int(np.sqrt(n)))))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(Xn), km

def cluster_titles_from_centroids(km, vect, topn=4):
    # titoli usando top termini dei centroidi
    feats = np.array(vect.get_feature_names_out())
    centers = km.cluster_centers_
    titles = {}
    for i in range(centers.shape[0]):
        idx = np.argsort(centers[i])[::-1][:topn]
        words = feats[idx]
        title = " / ".join([w for w in words if len(w) >= 3])
        titles[i] = title if title else f"Cluster {i}"
    return titles

# =========================
# FORECAST / SIMULAZIONE
# =========================
def monthly_aggregate(df, date_col, amt_col):
    tmp = df[df[date_col].notna()].copy()
    tmp["_ym"] = month_key(tmp[date_col])
    m = tmp.groupby("_ym")[amt_col].sum().reset_index().rename(columns={amt_col: "net"})
    return m

def month_components(df):
    """
    Crea serie mensili entrate/uscite/netto e una stima stagionalità entrate per mese dell'anno.
    """
    tmp = df[df["_date"].notna()].copy()
    tmp["_ym"] = month_key(tmp["_date"])
    tmp["_y"] = tmp["_date"].dt.year
    tmp["_m"] = tmp["_date"].dt.month

    inflow = tmp[tmp["_amount"] > 0].groupby("_ym")["_amount"].sum()
    outflow = tmp[tmp["_amount"] < 0].groupby("_ym")["_amount"].sum()  # negativo
    net = tmp.groupby("_ym")["_amount"].sum()

    mdf = pd.DataFrame({
        "_ym": sorted(set(tmp["_ym"].unique())),
    })
    mdf["entrate"] = mdf["_ym"].map(inflow).fillna(0.0)
    mdf["uscite"] = mdf["_ym"].map(outflow).fillna(0.0)
    mdf["netto"] = mdf["_ym"].map(net).fillna(0.0)

    # per trend: indice temporale 0..n-1
    mdf["t"] = np.arange(len(mdf))

    # stagionalità entrate per mese dell'anno
    # calcoliamo mean entrate per mese (1..12)
    tmp_m = tmp.copy()
    tmp_m["entrate_pos"] = np.where(tmp_m["_amount"] > 0, tmp_m["_amount"], 0.0)
    seas = tmp_m.groupby(tmp_m["_date"].dt.month)["entrate_pos"].sum()  # somma mensile per month-of-year aggregata su tutto lo storico
    # normalizzo: fattore rispetto alla media
    mean_monthly_income = mdf["entrate"].mean() if len(mdf) else 0.0
    if mean_monthly_income > 0:
        season_factor = (seas / seas.mean()).to_dict()
    else:
        season_factor = {i: 1.0 for i in range(1, 13)}

    return mdf, season_factor

def simulate_balance(
    start_balance: float,
    start_next_month: pd.Timestamp,
    target_balance: float,
    max_monthly_spend_abs: float,
    mdf_hist: pd.DataFrame,
    season_factor: dict,
    use_trend: bool = True,
    use_seasonality: bool = True,
    horizon_months: int = 60
):
    """
    Simula mese per mese:
    entrate = trend (su storico) * stagionalità (opzionale)
    uscite = -min(max_spesa, |media uscite storico|)
    """
    if len(mdf_hist) < 2:
        return pd.DataFrame()

    # Trend entrate su storico
    base_income = float(mdf_hist["entrate"].mean())
    if SKLEARN_OK and use_trend:
        X = mdf_hist[["t"]].values
        y = mdf_hist["entrate"].values
        lr = LinearRegression()
        lr.fit(X, y)
        income_pred_fn = lambda t: float(lr.predict(np.array([[t]]))[0])
    else:
        income_pred_fn = lambda t: base_income

    # baseline uscite: media valore assoluto uscite (negativo)
    avg_spend_abs = float((-mdf_hist["uscite"]).mean())
    spend_abs = min(max_monthly_spend_abs, avg_spend_abs) if avg_spend_abs > 0 else max_monthly_spend_abs

    rows = []
    bal = float(start_balance)

    # indice t futuro continua dallo storico
    t0 = int(mdf_hist["t"].max()) + 1

    current = pd.Timestamp(start_next_month).replace(day=1)
    for i in range(horizon_months):
        t = t0 + i
        inc = max(0.0, income_pred_fn(t))
        if use_seasonality:
            inc *= float(season_factor.get(int(current.month), 1.0))

        out = -float(spend_abs)  # negativo
        net = inc + out
        bal = bal + net

        rows.append({
            "mese": current.strftime("%Y-%m"),
            "entrate_sim": inc,
            "uscite_sim": out,
            "netto_sim": net,
            "saldo_sim": bal
        })

        if bal >= target_balance:
            break

        current = (current + pd.offsets.MonthBegin(1)).replace(day=1)

    return pd.DataFrame(rows)

# =========================
# UI INPUT
# =========================
st.sidebar.header("⚙️ Impostazioni")

enable_cluster = st.sidebar.toggle("Clustering 'AI' leggero (sklearn)", value=True)
if enable_cluster and not SKLEARN_OK:
    st.sidebar.warning("scikit-learn non disponibile: clustering disattivato.")
    enable_cluster = False

k_clusters = st.sidebar.slider("Numero indicativo cluster", 4, 24, 12, 1) if enable_cluster else 0
ma_window = st.sidebar.slider("Finestra medie mobili (mesi)", 2, 6, 3, 1)

debug = st.sidebar.toggle("🔎 Debug (min/max importi)", value=False)

st.sidebar.divider()
st.sidebar.subheader("🎯 Simulazione saldo target")

target_balance = st.sidebar.number_input("Saldo target (€)", value=5000.0, step=100.0)
max_spend = st.sidebar.number_input("Spesa massima mensile (assoluto, €)", value=1200.0, step=50.0, min_value=0.0)
use_trend = st.sidebar.toggle("Usa trend entrate", value=True)
use_seasonality = st.sidebar.toggle("Usa stagionalità entrate", value=True)
horizon = st.sidebar.slider("Orizzonte massimo (mesi)", 6, 120, 60, 6)

# =========================
# LOAD DATA
# =========================
uploaded = st.file_uploader("Carica Excel (movimenti nel primo foglio)", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

try:
    df, sheet_name, saldo_value = load_first_sheet(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore lettura Excel: {e}")
    st.stop()

# colonne principali
c_date = infer_col(df, ["data", "date"])
c_desc = infer_col(df, ["descrizione", "description", "causale", "dettaglio", "merchant", "nome", "desc"])
c_amt  = infer_col(df, ["importo", "amount", "valore", "movimento", "€", "eur"])

# colonna pagamento (se esiste)
c_pay = infer_col(df, ["pagamento", "payment", "metodo", "strumento", "tipo operazione", "operazione", "canale"])

if c_desc is None or c_amt is None:
    st.error(f"Non trovo le colonne chiave (Descrizione/Importo). Colonne: {list(df.columns)}")
    st.stop()

out = df.copy()
out["_amount"] = parse_amount(out[c_amt])
out = out.dropna(subset=["_amount"])

if c_date is not None:
    out["_date"] = pd.to_datetime(out[c_date], errors="coerce", dayfirst=True)
else:
    out["_date"] = pd.NaT

out["_desc"] = out[c_desc].astype(str).map(norm_text)
out["_pay_raw"] = out[c_pay].astype(str).map(norm_text) if c_pay is not None else None

if debug and len(out):
    st.sidebar.write("Min importo:", float(out["_amount"].min()))
    st.sidebar.write("Max importo:", float(out["_amount"].max()))
    st.sidebar.write("Esempi:", out["_amount"].head(8).tolist())

# =========================
# CATEGORIE + CLUSTER (opzionale) + TITOLI
# =========================
out["_category_rule"] = [tag_category_rules(d, a) for d, a in zip(out["_desc"], out["_amount"])]

# clustering su descrizioni (serve anche per titoli cluster pertinenti)
if enable_cluster:
    desc_list = out["_desc"].tolist()
    Xn, vect = build_tfidf(desc_list)
    labels, km = kmeans_labels(Xn, desired_k=k_clusters)
    out["_cluster"] = labels

    titles = cluster_titles_from_centroids(km, vect, topn=4)
    out["_cluster_title"] = out["_cluster"].map(titles)
else:
    out["_cluster"] = 0
    out["_cluster_title"] = "Cluster unico"

# categoria finale:
# - se regola != Altro => regola
# - se Altro => “Altro: <titolo cluster>” (così hai sub-categorie coerenti)
out["_category"] = np.where(
    out["_category_rule"] == "Altro",
    "Altro: " + out["_cluster_title"].astype(str),
    out["_category_rule"]
)

# =========================
# PAYMENT TYPES
# =========================
if c_pay is not None:
    pay_list = out["_pay_raw"].tolist()
else:
    pay_list = [None] * len(out)

out["_payment_type"] = [
    tag_payment(d, p) for d, p in zip(out["_desc"], pay_list)
]

# =========================
# KPI
# =========================
tot_spese = out.loc[out["_amount"] < 0, "_amount"].sum()
tot_entrate = out.loc[out["_amount"] > 0, "_amount"].sum()
netto = out["_amount"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Saldo (estratto)", f"{saldo_value:,.2f} €" if saldo_value is not None else "N/D")
c2.metric("Entrate (periodo)", f"{tot_entrate:,.2f} €")
c3.metric("Uscite (periodo)", f"{tot_spese:,.2f} €")
c4.metric("Netto (periodo)", f"{netto:,.2f} €")

st.caption(f"Foglio letto: {sheet_name} | Clustering: {'ON' if enable_cluster else 'OFF'} | sklearn: {'OK' if SKLEARN_OK else 'NO'}")
st.divider()

# =========================
# 1) GRAFICO CUMULATA PERIODO (PRIMO)
# =========================
st.subheader("✅ Cumulata del periodo (saldo relativo ai movimenti)")
if out["_date"].notna().any():
    tmp = out.dropna(subset=["_date"]).sort_values("_date")
    tmp["_cum"] = tmp["_amount"].cumsum()
    fig_cum = px.line(tmp, x="_date", y="_cum")
    fig_cum.update_layout(xaxis_title="Data", yaxis_title="Cumulata movimenti (€)")
    st.plotly_chart(fig_cum, use_container_width=True)
else:
    st.info("Colonna Data non disponibile: cumulata non mostrabile.")

# =========================
# 2) ENTRATE / USCITE / NETTO (mensile)
# =========================
st.subheader("✅ Entrate / Uscite / Netto (mensile)")
if out["_date"].notna().any():
    tmp = out.dropna(subset=["_date"]).copy()
    tmp["_ym"] = month_key(tmp["_date"])

    m_in = tmp[tmp["_amount"] > 0].groupby("_ym")["_amount"].sum()
    m_out = tmp[tmp["_amount"] < 0].groupby("_ym")["_amount"].sum()  # negativo
    m_net = tmp.groupby("_ym")["_amount"].sum()

    months = sorted(tmp["_ym"].unique())
    mdf = pd.DataFrame({"Mese": months})
    mdf["Entrate"] = mdf["Mese"].map(m_in).fillna(0.0)
    mdf["Uscite"] = mdf["Mese"].map(m_out).fillna(0.0)
    mdf["Netto"] = mdf["Mese"].map(m_net).fillna(0.0)

    # grafico grouped bar
    fig_eun = go.Figure()
    fig_eun.add_trace(go.Bar(name="Entrate", x=mdf["Mese"], y=mdf["Entrate"]))
    fig_eun.add_trace(go.Bar(name="Uscite", x=mdf["Mese"], y=mdf["Uscite"]))
    fig_eun.add_trace(go.Bar(name="Netto", x=mdf["Mese"], y=mdf["Netto"]))
    fig_eun.update_layout(barmode="group", xaxis_title="Mese", yaxis_title="€")
    st.plotly_chart(fig_eun, use_container_width=True)
else:
    st.info("Colonna Data non disponibile: vista mensile non mostrabile.")

# =========================
# 3) MEDIE MOBILI (entrate + uscite)
# =========================
st.subheader("✅ Medie mobili (entrate + uscite) — mese per mese")
if out["_date"].notna().any():
    tmp = out.dropna(subset=["_date"]).copy()
    tmp["_ym"] = month_key(tmp["_date"])

    m_in = tmp[tmp["_amount"] > 0].groupby("_ym")["_amount"].sum()
    m_out_abs = tmp[tmp["_amount"] < 0].groupby("_ym")["_amount"].sum().abs()

    months = sorted(tmp["_ym"].unique())
    mdf = pd.DataFrame({"Mese": months})
    mdf["Entrate"] = mdf["Mese"].map(m_in).fillna(0.0)
    mdf["Uscite"] = mdf["Mese"].map(m_out_abs).fillna(0.0)

    mdf[f"MA{ma_window}_Entrate"] = mdf["Entrate"].rolling(ma_window).mean()
    mdf[f"MA{ma_window}_Uscite"] = mdf["Uscite"].rolling(ma_window).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(name="Entrate", x=mdf["Mese"], y=mdf["Entrate"], mode="lines+markers"))
    fig_ma.add_trace(go.Scatter(name="Uscite", x=mdf["Mese"], y=mdf["Uscite"], mode="lines+markers"))
    fig_ma.add_trace(go.Scatter(name=f"MA{ma_window} Entrate", x=mdf["Mese"], y=mdf[f"MA{ma_window}_Entrate"], mode="lines"))
    fig_ma.add_trace(go.Scatter(name=f"MA{ma_window} Uscite", x=mdf["Mese"], y=mdf[f"MA{ma_window}_Uscite"], mode="lines"))
    fig_ma.update_layout(xaxis_title="Mese", yaxis_title="€")
    st.plotly_chart(fig_ma, use_container_width=True)
else:
    st.info("Colonna Data non disponibile: medie mobili non mostrabili.")

st.divider()

# =========================
# 4) TIPOLOGIE DI SPESA (categorie): grafico + tabella + stacked mensile + totale periodo
# =========================
st.subheader("✅ Tipologie di spesa (categorie): regole forti + clustering leggero + titoli cluster")

spese = out[out["_amount"] < 0].copy()
if len(spese) == 0:
    st.info("Nessuna uscita trovata nel periodo.")
else:
    # totale periodo per categoria
    by_cat = spese.groupby("_category")["_amount"].sum().abs().sort_values(ascending=False).reset_index()
    by_cat.columns = ["Categoria", "Totale (€)"]

    cA, cB = st.columns([1.1, 0.9])

    with cA:
        fig_cat = px.bar(by_cat.head(20), x="Categoria", y="Totale (€)")
        fig_cat.update_layout(xaxis_title="Categoria", yaxis_title="Spesa totale (€)")
        st.plotly_chart(fig_cat, use_container_width=True)

    with cB:
        st.markdown("**Tabella decrescente (top 30)**")
        st.dataframe(by_cat.head(30), use_container_width=True)

    # stacked mensile per categoria
    st.markdown("**Stacked mensile per categoria**")
    if spese["_date"].notna().any():
        sp = spese.dropna(subset=["_date"]).copy()
        sp["_ym"] = month_key(sp["_date"])
        pivot_cat = (
            sp.pivot_table(index="_ym", columns="_category", values="_amount", aggfunc="sum")
            .fillna(0.0)
            .abs()
        )
        pivot_cat = pivot_cat.loc[sorted(pivot_cat.index)]

        fig_stack_cat = go.Figure()
        for col in pivot_cat.columns:
            fig_stack_cat.add_trace(go.Bar(name=str(col), x=pivot_cat.index, y=pivot_cat[col]))
        fig_stack_cat.update_layout(barmode="stack", xaxis_title="Mese", yaxis_title="Spesa (€)")
        st.plotly_chart(fig_stack_cat, use_container_width=True)
    else:
        st.info("Colonna Data non disponibile: stacked mensile per categoria non mostrabile.")

st.divider()

# =========================
# 5) TIPOLOGIE DI PAGAMENTO: totale periodo + stacked mensile
# =========================
st.subheader("✅ Tipologie di pagamento (Carta / Bonifico / Addebito diretto / Prelievo / PagoPA-F24 / Commissioni / Altro)")

# totale periodo per tipologia pagamento (su tutto: entrate+uscite; se vuoi solo spese, filtra qui)
by_pay = out.groupby("_payment_type")["_amount"].sum().abs().sort_values(ascending=False).reset_index()
by_pay.columns = ["Tipologia pagamento", "Totale (€)"]

cP1, cP2 = st.columns([1.1, 0.9])

with cP1:
    fig_pay = px.bar(by_pay, x="Tipologia pagamento", y="Totale (€)")
    fig_pay.update_layout(xaxis_title="Tipologia pagamento", yaxis_title="Totale (€)")
    st.plotly_chart(fig_pay, use_container_width=True)

with cP2:
    st.markdown("**Tabella decrescente**")
    st.dataframe(by_pay, use_container_width=True)

st.markdown("**Stacked mensile per tipologia pagamento**")
if out["_date"].notna().any():
    tmp = out.dropna(subset=["_date"]).copy()
    tmp["_ym"] = month_key(tmp["_date"])
    pivot_pay = (
        tmp.pivot_table(index="_ym", columns="_payment_type", values="_amount", aggfunc="sum")
        .fillna(0.0)
        .abs()
    )
    pivot_pay = pivot_pay.loc[sorted(pivot_pay.index)]

    fig_stack_pay = go.Figure()
    for col in pivot_pay.columns:
        fig_stack_pay.add_trace(go.Bar(name=str(col), x=pivot_pay.index, y=pivot_pay[col]))
    fig_stack_pay.update_layout(barmode="stack", xaxis_title="Mese", yaxis_title="Totale (€)")
    st.plotly_chart(fig_stack_pay, use_container_width=True)
else:
    st.info("Colonna Data non disponibile: stacked mensile per tipologia pagamento non mostrabile.")

st.divider()

# =========================
# 6) SIMULAZIONE: mese raggiungimento saldo target (trend + stagionalità entrate) + spesa max
# =========================
st.subheader("✅ Simulazione: mese di raggiungimento saldo target")

if not out["_date"].notna().any():
    st.info("Serve la colonna Data per stimare trend/stagionalità e fare la simulazione.")
else:
    # storico mensile + stagionalità
    mdf_hist, seas = month_components(out)

    # saldo iniziale: usa saldo estratto se presente, altrimenti saldo relativo dei movimenti
    if saldo_value is not None and not pd.isna(saldo_value):
        start_balance = float(saldo_value)
        start_note = "Saldo iniziale = saldo estratto (trovato nel file)"
    else:
        # fallback: cumulata dei movimenti termina su un saldo relativo
        tmp = out.dropna(subset=["_date"]).sort_values("_date")
        start_balance = float(tmp["_amount"].cumsum().iloc[-1]) if len(tmp) else 0.0
        start_note = "Saldo iniziale = cumulata movimenti (fallback: saldo estratto non trovato)"

    # partenza dal prossimo mese rispetto all'ultima data in storico
    last_date = out.dropna(subset=["_date"])["_date"].max()
    next_month = (pd.Timestamp(last_date) + pd.offsets.MonthBegin(1)).replace(day=1)

    sim = simulate_balance(
        start_balance=start_balance,
        start_next_month=next_month,
        target_balance=float(target_balance),
        max_monthly_spend_abs=float(max_spend),
        mdf_hist=mdf_hist,
        season_factor=seas,
        use_trend=bool(use_trend),
        use_seasonality=bool(use_seasonality),
        horizon_months=int(horizon),
    )

    st.caption(start_note)
    st.caption(f"Partenza simulazione: {next_month.strftime('%Y-%m')} | Orizzonte max: {horizon} mesi")

    if sim.empty:
        st.warning("Storico insufficiente per stimare la simulazione (servono almeno 2 mesi con date).")
    else:
        # esito
        reached = sim["saldo_sim"].iloc[-1] >= float(target_balance)
        if reached:
            st.success(f"Target raggiunto in: {sim['mese'].iloc[-1]}")
        else:
            st.warning(f"Target NON raggiunto entro {len(sim)} mesi (ultimo saldo simulato: {sim['saldo_sim'].iloc[-1]:,.2f} €)")

        # grafico saldo simulato
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(name="Saldo simulato", x=sim["mese"], y=sim["saldo_sim"], mode="lines+markers"))
        fig_sim.add_hline(y=float(target_balance), line_dash="dash", annotation_text="Target", annotation_position="top left")
        fig_sim.update_layout(xaxis_title="Mese", yaxis_title="Saldo (€)")
        st.plotly_chart(fig_sim, use_container_width=True)

        # dettaglio tabellare
        st.markdown("**Dettaglio simulazione**")
        sim_view = sim.copy()
        sim_view["entrate_sim"] = sim_view["entrate_sim"].round(2)
        sim_view["uscite_sim"] = sim_view["uscite_sim"].round(2)
        sim_view["netto_sim"] = sim_view["netto_sim"].round(2)
        sim_view["saldo_sim"] = sim_view["saldo_sim"].round(2)
        st.dataframe(sim_view, use_container_width=True)

st.divider()

# =========================
# TABELLA MOVIMENTI
# =========================
st.subheader("📄 Movimenti (categoria + cluster + pagamento)")
display_cols = []
if c_date is not None:
    display_cols.append(c_date)
display_cols += [c_desc, c_amt]
if c_pay is not None:
    display_cols.append(c_pay)

view = out.copy()
view["Categoria"] = view["_category"]
view["Cluster"] = view["_cluster"]
view["Titolo cluster"] = view["_cluster_title"]
view["Pagamento"] = view["_payment_type"]

# ordina per data (se presente)
if c_date is not None and out["_date"].notna().any():
    view = view.sort_values("_date", ascending=False)

st.dataframe(
    view[display_cols + ["Categoria", "Pagamento", "Cluster", "Titolo cluster"]],
    use_container_width=True
)