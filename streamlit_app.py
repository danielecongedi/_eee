import streamlit as st
import pandas as pd
import numpy as np
import re

import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# --- AI ---
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Dashboard Spese", layout="wide", page_icon="💳")
st.title("💳 Dashboard spese (AI tagging)")

SALDO_CELL = "F1"
MIN_ROWS_FOR_CLUSTER = 30

# Macro-tag candidati (QUI sì: solo le classi, non keyword)
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

def excel_cell_to_idx(cell: str):
    cell = cell.upper().strip()
    col_letters = re.findall(r"[A-Z]+", cell)[0]
    row_number = int(re.findall(r"\d+", cell)[0])
    col_idx = 0
    for ch in col_letters:
        col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
    col_idx -= 1
    row_idx = row_number - 1
    return row_idx, col_idx

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
    df = pd.read_excel(file_bytes, sheet_name=first_sheet)
    df.columns = [str(c).strip() for c in df.columns]

    raw = pd.read_excel(file_bytes, sheet_name=first_sheet, header=None)
    r, c = excel_cell_to_idx(SALDO_CELL)
    saldo = None
    if r < raw.shape[0] and c < raw.shape[1]:
        v = raw.iat[r, c]
        if isinstance(v, str):
            vv = v.replace("€", "").strip().replace(".", "").replace(",", ".")
            try:
                saldo = float(vv)
            except:
                saldo = None
        elif pd.notna(v):
            try:
                saldo = float(v)
            except:
                saldo = None

    return df, first_sheet, saldo

@st.cache_resource(show_spinner=False)
def load_models():
    # Embedding model (ottimo e veloce)
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Zero-shot classifier (robusto per label set personalizzate)
    zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return embed_model, zsc

def ai_tag_descriptions(desc_list, amounts, zsc, labels):
    """
    Zero-shot per macro-tag.
    Ottimizzazione: dedup delle descrizioni per non classificare N volte le stesse stringhe.
    """
    # dedup
    uniq = list(dict.fromkeys(desc_list))
    mapping = {}

    # Batch “a chunk” per non rallentare troppo
    chunk = 32
    for i in range(0, len(uniq), chunk):
        batch = uniq[i:i+chunk]
        # multi_label=False: sceglie 1 label
        out = zsc(batch, candidate_labels=labels, multi_label=False)
        if isinstance(out, dict):
            out = [out]
        for text, res in zip(batch, out):
            mapping[text] = res["labels"][0]

    # applico e fix entrate/uscite: se amount > 0 -> Entrate (override)
    tagged = []
    for d, a in zip(desc_list, amounts):
        if a is not None and a > 0:
            tagged.append("Entrate")
        else:
            tagged.append(mapping.get(d, "Altro"))
    return tagged

def cluster_embeddings(embeddings, desired_k=12):
    n = embeddings.shape[0]
    if n < MIN_ROWS_FOR_CLUSTER:
        return np.zeros(n, dtype=int)
    k = int(np.clip(desired_k, 2, max(2, int(np.sqrt(n)))))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(embeddings)

# =========================
# APP
# =========================
uploaded = st.file_uploader("Carica Excel (movimenti nel primo foglio + saldo in F1)", type=["xlsx", "xls"])
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
c_amt  = infer_col(df, ["importo", "amount", "valore", "movimento", "€", "eur"])

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

# AI models
with st.spinner("Carico modelli AI (solo la prima volta può metterci un po')..."):
    embed_model, zsc = load_models()

# embeddings
with st.spinner("Creo embeddings delle descrizioni..."):
    emb = embed_model.encode(out["_desc_raw"].tolist(), show_progress_bar=False)
    emb = normalize(np.array(emb))

# clustering su embeddings
out["_cluster"] = cluster_embeddings(emb, desired_k=12)

# AI tagging zero-shot
with st.spinner("Assegno macro-tag con AI (zero-shot)..."):
    out["_macro_tag"] = ai_tag_descriptions(
        out["_desc_raw"].tolist(),
        out["_amount"].tolist(),
        zsc,
        MACRO_TAGS
    )

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

c1.metric("Saldo (Excel F1)", f"{saldo_value:,.2f} €" if saldo_value is not None else "N/D")
c2.metric("Entrate (periodo)", f"{tot_entrate:,.2f} €")
c3.metric("Uscite (periodo)", f"{tot_spese:,.2f} €")
c4.metric("Netto (entrate+uscite)", f"{netto:,.2f} €")

st.caption(f"Foglio letto: **{sheet_name}** | Saldo da cella **{SALDO_CELL}**")

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
    st.subheader("🍽️ Spese per macro attività (AI)")
    tmp = out[out["_amount"] < 0].copy()
    if len(tmp):
        by_tag = tmp.groupby("_macro_tag")["_amount"].sum().abs().sort_values(ascending=False).head(12).reset_index()
        fig2 = px.bar(by_tag, x="_macro_tag", y="_amount")
        fig2.update_layout(xaxis_title="Macro attività", yaxis_title="Spesa (€)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Nessuna uscita trovata.")

st.divider()

st.subheader("📄 Movimenti (AI tag + cluster)")
display_cols = []
if c_date is not None:
    display_cols.append(c_date)
display_cols += [c_desc, c_amt]

view = out.copy()
view["Macro attività (AI)"] = view["_macro_tag"]
view["Cluster"] = view["_cluster"]
view["Tag cluster"] = view["_cluster_macro_tag"]

if c_date is not None and view["_date"].notna().any():
    view = view.sort_values("_date", ascending=False)

st.dataframe(view[display_cols + ["Macro attività (AI)", "Cluster", "Tag cluster"]], use_container_width=True)