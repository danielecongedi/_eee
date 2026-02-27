import streamlit as st
import pandas as pd
import numpy as np
import re

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Dashboard Spese", layout="wide", page_icon="💳")
st.title("💳 Dashboard spese (tagging leggero + clustering)")

MIN_ROWS_FOR_CLUSTER = 30

MACRO_TAGS = [
    "Viaggi", "Cibo", "Casa", "Auto/Trasporti", "Abbonamenti",
    "Shopping", "Salute", "Svago", "Commissioni/Banca", "Entrate", "Altro"
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


def parse_amount(series: pd.Series) -> pd.Series:
    """
    Parsing robusto:
    - Se la cella è numerica -> usa direttamente quel numero
    - Se stringa:
        * formato IT: 1.234,56 -> 1234.56
        * formato IT base: 1234,56 -> 1234.56
        * formato EN: 1234.56 -> 1234.56 (NON rimuovere punti!)
    """
    def clean_value(val):
        if pd.isna(val):
            return np.nan

        # Se Excel l'ha già letto come numero, non “rompere” nulla
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)

        s = str(val).strip()
        s = s.replace("€", "").replace("\u00a0", "").replace(" ", "")

        # supporto segno negativo tipo "(123,45)"
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        # Se contiene sia '.' che ',' decidiamo quale è decimale
        if "," in s and "." in s:
            # se la virgola è dopo l'ultimo punto -> IT (1.234,56)
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                # raro: 1,234.56 (EN con migliaia)
                s = s.replace(",", "")
        elif "," in s and "." not in s:
            # IT base: 1234,56
            s = s.replace(",", ".")
        else:
            # EN base: 1234.56 -> lascia così
            pass

        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(clean_value)


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


@st.cache_data(show_spinner=False)
def load_first_sheet(file_bytes: bytes):
    xls = pd.ExcelFile(file_bytes)
    first_sheet = xls.sheet_names[0]

    raw_head = pd.read_excel(file_bytes, sheet_name=first_sheet, header=None, nrows=25)

    saldo = None
    header_row = 0

    for idx, row in raw_head.iterrows():
        row_str_lower = [str(x).lower().strip() for x in row.dropna().tolist()]

        # --- RICERCA SALDO ---
        if any("saldo" in val for val in row_str_lower) and saldo is None:
            for val in row.dropna().tolist():
                if isinstance(val, (int, float)):
                    saldo = float(val)
                    break
                elif isinstance(val, str):
                    clean_val = val.replace("€", "").replace(" ", "").strip()

                    # prova parsing robusto su stringa
                    tmp = parse_amount(pd.Series([clean_val])).iloc[0]
                    if not pd.isna(tmp):
                        saldo = float(tmp)
                        break

        # --- RICERCA INTESTAZIONE DATI ---
        if any("importo" in val for val in row_str_lower) and any("data" in val for val in row_str_lower):
            header_row = idx
            break

    df = pd.read_excel(file_bytes, sheet_name=first_sheet, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]

    return df, first_sheet, saldo


def make_embeddings_tfidf(desc_list):
    vect = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    X = vect.fit_transform(desc_list)
    X = normalize(X)
    return X


def cluster_labels(matrix, desired_k=12):
    n = matrix.shape[0]
    if n < MIN_ROWS_FOR_CLUSTER:
        return np.zeros(n, dtype=int)

    k = int(np.clip(desired_k, 2, max(2, int(np.sqrt(n)))))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(matrix)


def tag_fallback(desc_list, amounts):
    rules = {
        "Cibo": ["bar", "rist", "pizz", "burger", "caff", "supermerc", "coop", "esselunga", "conad", "carrefour", "glovo", "deliveroo", "justeat", "just eat"],
        "Casa": ["affitto", "condominio", "luce", "gas", "acqua", "internet", "tim", "vodafone", "wind", "enel", "a2a", "tari"],
        "Auto/Trasporti": ["benz", "diesel", "eni", "q8", "autostr", "telepass", "tren", "tram", "metro", "taxi", "uber", "italo", "trenitalia"],
        "Abbonamenti": ["netflix", "spotify", "prime", "amazonprime", "amazon prime", "disney", "abbon", "subscription"],
        "Shopping": ["amazon", "ikea", "zara", "hm", "h&m", "decathlon", "mediaworld", "unieuro"],
        "Salute": ["farm", "medic", "ticket", "dent", "osped", "clinic"],
        "Svago": ["cinema", "teatro", "concerto", "pub", "aperi", "discoteca"],
        "Commissioni/Banca": ["commission", "canone", "imposta", "bollo", "spese", "fee", "prelievo", "bonifico", "ricarica", "carta"],
        "Viaggi": ["hotel", "airbnb", "flight", "ryanair", "easyjet", "booking", "expedia", "noleggio"],
    }

    out = []
    for d, a in zip(desc_list, amounts):
        if a is not None and a > 0:
            out.append("Entrate")
            continue

        dl = (d or "").lower()
        found = None
        for tag, keys in rules.items():
            if any(k in dl for k in keys):
                found = tag
                break
        out.append(found or "Altro")
    return out


# =========================
# APP
# =========================
uploaded = st.file_uploader("Carica Excel (movimenti nel primo foglio)", type=["xlsx", "xls"])
if not uploaded:
    st.stop()

try:
    df, sheet_name, saldo_value = load_first_sheet(uploaded.getvalue())
except Exception as e:
    st.error(f"Errore lettura Excel: {e}")
    st.stop()

c_date = infer_col(df, ["data", "date"])
c_desc = infer_col(df, ["descrizione", "description", "causale", "dettaglio", "merchant", "nome", "desc"])
c_amt  = infer_col(df, ["importo", "amount", "valore", "movimento", "€", "eur"])

if c_desc is None or c_amt is None:
    st.error(f"Non trovo le colonne chiave (Descrizione/Importo). Colonne: {list(df.columns)}")
    st.stop()

out = df.copy()

# parsing corretto importi
out["_amount"] = parse_amount(out[c_amt])
out = out.dropna(subset=["_amount"])

# data
if c_date is not None:
    out["_date"] = pd.to_datetime(out[c_date], errors="coerce", dayfirst=True)
else:
    out["_date"] = pd.NaT

# descrizione
out["_desc_raw"] = out[c_desc].astype(str).map(lambda x: re.sub(r"\s+", " ", str(x).strip()))

desc_list = out["_desc_raw"].tolist()
amounts = out["_amount"].tolist()

# debug toggle
debug = st.sidebar.toggle("🔎 Debug importi", value=False)
if debug and len(out):
    st.sidebar.write("Min importo:", float(out["_amount"].min()))
    st.sidebar.write("Max importo:", float(out["_amount"].max()))
    st.sidebar.write("Esempi importi:", out["_amount"].head(10).tolist())

# embeddings (TF-IDF) + clustering
with st.spinner("Creo embeddings (TF-IDF) + clustering..."):
    X = make_embeddings_tfidf(desc_list)
    out["_cluster"] = cluster_labels(X, desired_k=12)

# tagging leggero
with st.spinner("Assegno macro-tag (regole leggere)..."):
    out["_macro_tag"] = tag_fallback(desc_list, amounts)

# cluster -> macro dominante
cluster_macro = (
    out.groupby("_cluster")["_macro_tag"]
      .agg(lambda s: s.value_counts().index[0])
      .to_dict()
)
out["_cluster_macro_tag"] = out["_cluster"].map(cluster_macro)

# =========================
# KPI
# =========================
c1, c2, c3, c4 = st.columns(4)
tot_spese = out.loc[out["_amount"] < 0, "_amount"].sum()
tot_entrate = out.loc[out["_amount"] > 0, "_amount"].sum()
netto = out["_amount"].sum()

c1.metric("Saldo (estratto)", f"{saldo_value:,.2f} €" if saldo_value is not None else "N/D")
c2.metric("Entrate (periodo)", f"{tot_entrate:,.2f} €")
c3.metric("Uscite (periodo)", f"{tot_spese:,.2f} €")
c4.metric("Netto (entrate+uscite)", f"{netto:,.2f} €")

st.caption(f"Foglio letto: {sheet_name} | Saldo individuato automaticamente")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("📈 Netto mensile")
    if out["_date"].notna().any():
        tmp = out.copy()
        tmp["_year_month"] = tmp["_date"].dt.to_period("M").astype(str)
        monthly = tmp.groupby("_year_month")["_amount"].sum().reset_index()
        fig = px.bar(monthly, x="_year_month", y="_amount")
        fig.update_layout(xaxis_title="Mese", yaxis_title="Netto (€/mese)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Colonna Data non disponibile: grafico mensile non mostrabile.")

with right:
    st.subheader("📊 Spese per macro attività")
    tmp = out[out["_amount"] < 0].copy()
    if len(tmp):
        by_tag = (
            tmp.groupby("_macro_tag")["_amount"]
            .sum().abs()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig2 = px.bar(by_tag, x="_macro_tag", y="_amount")
        fig2.update_layout(xaxis_title="Macro attività", yaxis_title="Spesa (€)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Nessuna uscita trovata.")

st.divider()

st.subheader("📄 Movimenti (tag + cluster)")
display_cols = []
if c_date is not None:
    display_cols.append(c_date)
display_cols += [c_desc, c_amt]

view = out.copy()
view["Macro attività"] = view["_macro_tag"]
view["Cluster"] = view["_cluster"]
view["Tag cluster"] = view["_cluster_macro_tag"]

if c_date is not None and view["_date"].notna().any():
    view = view.sort_values("_date", ascending=False)

st.dataframe(
    view[display_cols + ["Macro attività", "Cluster", "Tag cluster"]],
    use_container_width=True
)