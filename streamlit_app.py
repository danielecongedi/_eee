import streamlit as st
import pandas as pd
import numpy as np
import re
import sys
import importlib.util

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Dashboard Spese", layout="wide", page_icon="💳")
st.title("💳 Dashboard spese (AI tagging)")

MIN_ROWS_FOR_CLUSTER = 30

MACRO_TAGS = [
    "Viaggi", "Cibo", "Casa", "Auto/Trasporti", "Abbonamenti",
    "Shopping", "Salute", "Svago", "Commissioni/Banca", "Entrate", "Altro"
]


# =========================
# DIAGNOSTICA IMPORT
# =========================
def can_import(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


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
    s = series.astype(str).str.replace("€", "", regex=False).str.strip()
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


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

        # RICERCA SALDO
        if any("saldo" in val for val in row_str_lower) and saldo is None:
            for val in row.dropna().tolist():
                if isinstance(val, (int, float)):
                    saldo = float(val)
                    break
                elif isinstance(val, str):
                    clean_val = val.replace("€", "").strip()
                    if clean_val.count('.') <= 1 and clean_val.count(',') == 0:
                        pass
                    elif clean_val.count(',') == 1 and clean_val.count('.') == 0:
                        clean_val = clean_val.replace(',', '.')
                    else:
                        clean_val = clean_val.replace(".", "").replace(",", ".")
                    try:
                        saldo = float(clean_val)
                        break
                    except ValueError:
                        continue

        # RICERCA INTESTAZIONE
        if any("importo" in val for val in row_str_lower) and any("data" in val for val in row_str_lower):
            header_row = idx
            break

    df = pd.read_excel(file_bytes, sheet_name=first_sheet, skiprows=header_row)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False, na=False)]

    return df, first_sheet, saldo


def cluster_labels(matrix, desired_k=12):
    n = matrix.shape[0]
    if n < MIN_ROWS_FOR_CLUSTER:
        return np.zeros(n, dtype=int)

    k = int(np.clip(desired_k, 2, max(2, int(np.sqrt(n)))))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init=10 più compatibile
    return km.fit_predict(matrix)


# =========================
# AI: PESANTE (se disponibile) + FALLBACK
# =========================
@st.cache_resource(show_spinner=False)
def load_heavy_models():
    """
    Carica sentence-transformers + zero-shot.
    Se non sono installati, questa funzione NON va chiamata.
    """
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return embed_model, zsc


def ai_tag_zero_shot(desc_list, amounts, zsc, labels):
    """
    Zero-shot per macro-tag.
    Dedup descrizioni per velocizzare.
    """
    uniq = list(dict.fromkeys(desc_list))
    mapping = {}

    chunk = 32
    for i in range(0, len(uniq), chunk):
        batch = uniq[i:i + chunk]
        out = zsc(batch, candidate_labels=labels, multi_label=False)
        if isinstance(out, dict):
            out = [out]
        for text, res in zip(batch, out):
            mapping[text] = res["labels"][0]

    tagged = []
    for d, a in zip(desc_list, amounts):
        if a is not None and a > 0:
            tagged.append("Entrate")
        else:
            tagged.append(mapping.get(d, "Altro"))
    return tagged


def ai_tag_fallback(desc_list, amounts):
    """
    Fallback leggero (no transformers): regole + heuristics.
    Non è “LLM”, ma evita blocchi e ti dà tagging usabile.
    """
    # keyword minime (puoi ampliarle quando vuoi)
    rules = {
        "Cibo": ["bar", "rist", "pizz", "burger", "caff", "supermerc", "coop", "esselunga", "conad", "carrefour", "glovo", "deliveroo", "just eat"],
        "Casa": ["affitto", "condominio", "luce", "gas", "acqua", "internet", "tim", "vodafone", "wind", "enel", "a2a", "tari"],
        "Auto/Trasporti": ["benz", "diesel", "eni", "q8", "es", "autostr", "telepass", "tren", "tram", "metro", "taxi", "uber"],
        "Abbonamenti": ["netflix", "spotify", "prime", "amazon prime", "disney", "abbon", "subscription"],
        "Shopping": ["amazon", "ikea", "zara", "h&m", "decathlon", "mediaworld", "unieuro"],
        "Salute": ["farm", "medic", "ticket", "dent", "osped", "clinic"],
        "Svago": ["cinema", "teatro", "concerto", "bar", "pub", "viaggio", "hotel", "bnb", "airbnb"],
        "Commissioni/Banca": ["commission", "canone", "imposta", "bollo", "spese", "fee", "prelievo", "bonifico"],
        "Viaggi": ["hotel", "airbnb", "flight", "ryanair", "easyjet", "trenitalia", "italo", "booking", "expedia"],
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


def make_embeddings_heavy(desc_list, embed_model):
    emb = embed_model.encode(desc_list, show_progress_bar=False)
    emb = normalize(np.array(emb))
    return emb


def make_embeddings_tfidf(desc_list):
    vect = TfidfVectorizer(min_df=1, max_features=5000, ngram_range=(1, 2))
    X = vect.fit_transform(desc_list)
    # normalizziamo per KMeans
    X = normalize(X)
    return X


# =========================
# SIDEBAR: CONTROLLI
# =========================
st.sidebar.header("⚙️ Controlli & diagnostica")

use_heavy = st.sidebar.toggle(
    "Usa AI pesante (sentence-transformers + zero-shot) se disponibile",
    value=True
)

st.sidebar.caption("Se disattivi: TF-IDF + regole (leggero, sempre stabile).")

st.sidebar.subheader("🧪 Ambiente")
st.sidebar.write("Python:", sys.executable)
st.sidebar.write("Versione:", sys.version.split()[0])

pkgs = {
    "torch": can_import("torch"),
    "transformers": can_import("transformers"),
    "sentence_transformers": can_import("sentence_transformers"),
    "sklearn": can_import("sklearn"),
    "openpyxl": can_import("openpyxl"),
}
st.sidebar.write("Import check:", pkgs)

heavy_available = pkgs["torch"] and pkgs["transformers"] and pkgs["sentence_transformers"]


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

# colonne
c_date = infer_col(df, ["data", "date"])
c_desc = infer_col(df, ["descrizione", "description", "causale", "dettaglio", "merchant", "nome", "desc"])
c_amt = infer_col(df, ["importo", "amount", "valore", "movimento", "€", "eur"])

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

out["_desc_raw"] = out[c_desc].astype(str).map(norm_text)

desc_list = out["_desc_raw"].tolist()
amounts = out["_amount"].tolist()

# =========================
# EMBEDDINGS + CLUSTER + TAG
# =========================
if use_heavy and heavy_available:
    with st.spinner("Carico modelli AI (pesanti)..."):
        try:
            embed_model, zsc = load_heavy_models()
        except Exception as e:
            st.warning(f"AI pesante non caricabile, uso fallback. Dettaglio: {e}")
            use_heavy = False

if use_heavy and heavy_available:
    with st.spinner("Creo embeddings (sentence-transformers)..."):
        emb = make_embeddings_heavy(desc_list, embed_model)
    out["_cluster"] = cluster_labels(emb, desired_k=12)

    with st.spinner("Assegno macro-tag (zero-shot)..."):
        out["_macro_tag"] = ai_tag_zero_shot(desc_list, amounts, zsc, MACRO_TAGS)

else:
    with st.spinner("Creo embeddings (TF-IDF fallback)..."):
        X = make_embeddings_tfidf(desc_list)
    out["_cluster"] = cluster_labels(X, desired_k=12)

    with st.spinner("Assegno macro-tag (fallback leggero)..."):
        out["_macro_tag"] = ai_tag_fallback(desc_list, amounts)

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
    st.subheader("🍽️ Spese per macro attività")
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