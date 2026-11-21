"""
Indicateurs de commerce extérieur - UN Comtrade + Banque mondiale
Lancer :  streamlit run app.py
"""
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
WB_URL = "https://api.worldbank.org/v2/country/{codes}/indicator/NY.GDP.MKTP.CD"
CACHE_DIR = Path(".comtrade_cache")
REPORTER_BATCH = 10      # nb de pays déclarants par appel API
MAX_YEARS_PER_CALL = 12  # limite Comtrade sur le nb de périodes

# nom -> (code Comtrade, ISO3).  Attention : Comtrade utilise 251 pour la France (pas 250) ; Italie = 380 depuis la mise à jour (381 = ancien code).
COUNTRIES = {
    # pays des notebooks 1 à 7
    "Morocco": (504, "MAR"), "Portugal": (620, "PRT"), "Tunisia": (788, "TUN"),
    "Greece": (300, "GRC"), "Egypt": (818, "EGY"), "Turkey": (792, "TUR"),
    # Golfe
    "Saudi Arabia": (682, "SAU"), "Bahrain": (48, "BHR"), "United Arab Emirates": (784, "ARE"),
    "Kuwait": (414, "KWT"), "Oman": (512, "OMN"), "Qatar": (634, "QAT"),
    # Union européenne (27)
    "Germany": (276, "DEU"), "Austria": (40, "AUT"), "Belgium": (56, "BEL"),
    "Bulgaria": (100, "BGR"), "Cyprus": (196, "CYP"), "Croatia": (191, "HRV"),
    "Denmark": (208, "DNK"), "Spain": (724, "ESP"), "Estonia": (233, "EST"),
    "Finland": (246, "FIN"), "France": (251, "FRA"), "Hungary": (348, "HUN"),
    "Ireland": (372, "IRL"), "Italy": (380, "ITA"), "Lithuania": (440, "LTU"),
    "Latvia": (428, "LVA"), "Luxembourg": (442, "LUX"), "Malta": (470, "MLT"),
    "Netherlands": (528, "NLD"), "Poland": (616, "POL"), "Czechia": (203, "CZE"),
    "Romania": (642, "ROU"), "Slovakia": (703, "SVK"), "Slovenia": (705, "SVN"),
    "Sweden": (752, "SWE"),
    # quelques pays supplémentaires
    "Algeria": (12, "DZA"), "Libya": (434, "LBY"), "Jordan": (400, "JOR"),
    "Lebanon": (422, "LBN"), "Mauritania": (478, "MRT"), "Senegal": (686, "SEN"),
    "Cote d'Ivoire": (384, "CIV"), "Israel": (376, "ISR"), "China": (156, "CHN"),
    "India": (699, "IND"), "USA": (842, "USA"), "United Kingdom": (826, "GBR"),
}
CODE2NAME = {v[0]: k for k, v in COUNTRIES.items()}
CODE2NAME[381] = "Italy"   # ancien code Comtrade, au cas où
NAME2ISO = {k: v[1] for k, v in COUNTRIES.items()}

GOLFE = ["Saudi Arabia", "Bahrain", "United Arab Emirates", "Kuwait", "Oman", "Qatar"]
UE = ["Germany", "Austria", "Belgium", "Bulgaria", "Cyprus", "Croatia", "Denmark", "Spain",
      "Estonia", "Finland", "France", "Greece", "Hungary", "Ireland", "Italy", "Lithuania",
      "Latvia", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Czechia",
      "Romania", "Slovakia", "Slovenia", "Sweden"]
DEFAULT_COUNTRIES = ["Morocco", "Portugal", "Tunisia", "Greece", "Egypt", "Turkey"]
REGION_PRESETS = {"Golfe": GOLFE, "Union européenne": UE}

IND_IP = "1 · Pénétration des importations (IP)"
IND_TDI = "2 · Dépendance commerciale (TDI)"
IND_NTB = "3 · Solde commercial normalisé"
IND_COV = "4 · Ratio de couverture (X/M)"
IND_EXP = "5 · Exportations et croissance annuelle"
IND_EP = "6 · Propension à exporter (EP)"
IND_MPM = "7 · Propension marginale à importer (MPM)"
IND_XS = "8/11/13 · Part des exportations intra-régionales (XS)"
IND_MS = "9/12 · Part des importations intra-régionales (MS)"
IND_TS = "10/14/15 · Part du commerce intra-régional (TS)"
IND_ALL = "10/14 · Vue combinée XS + MS + TS"
COUNTRY_INDS = [IND_IP, IND_TDI, IND_NTB, IND_COV, IND_EXP, IND_EP, IND_MPM]
REGION_INDS = [IND_XS, IND_MS, IND_TS, IND_ALL]
GDP_INDS = {IND_IP, IND_TDI, IND_EP, IND_MPM}


# ----------------------------------------------------------------------------
# Couche données : API Comtrade
# ----------------------------------------------------------------------------
class DataError(Exception):
    pass


def _chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def clean_key(k):
    """Enlève espaces, retours à la ligne et guillemets souvent collés avec la clé."""
    return "".join(str(k).split()).strip("\"'")


def _comtrade_get(params, key, retries=4):
    headers = {"Ocp-Apim-Subscription-Key": clean_key(key)}
    for attempt in range(retries):
        r = requests.get(COMTRADE_URL, params=params, headers=headers, timeout=90)
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", ""))
            except ValueError:
                wait = 15 * (attempt + 1)
            time.sleep(min(wait, 120))
            continue
        if r.status_code in (401, 403):
            raise DataError("Clé Comtrade refusée (401/403). Vérifiez la clé et l'abonnement au produit gratuit.")
        if r.status_code >= 400:
            raise DataError(f"Erreur Comtrade {r.status_code} : {r.text[:300]}")
        return r.json().get("data") or []
    raise DataError("Limite de requêtes Comtrade atteinte (429). Réessayez plus tard (500 appels/jour).")


def _call_comtrade(key, reporters, partners, years):
    rows = []
    for yrs in _chunks(years, MAX_YEARS_PER_CALL):
        params = {
            "reporterCode": ",".join(map(str, reporters)),
            "partnerCode": ",".join(map(str, partners)),
            "period": ",".join(map(str, yrs)),
            "flowCode": "M,X",
            "cmdCode": "TOTAL",
            "partner2Code": 0, "customsCode": "C00", "motCode": 0,
            "maxRecords": 100000, "format": "json", "includeDesc": "false",
        }
        rows += _comtrade_get(params, key)
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame(columns=["year", "reporter_code", "partner_code", "flow", "value"])
    df = pd.DataFrame(rows)
    need = ["refYear", "reporterCode", "partnerCode", "flowCode", "primaryValue"]
    if any(c not in df.columns for c in need):
        raise DataError(f"Colonnes inattendues dans la réponse Comtrade : {list(df.columns)[:15]}")
    df = df[need].rename(columns={"refYear": "year", "reporterCode": "reporter_code",
                                  "partnerCode": "partner_code", "flowCode": "flow",
                                  "primaryValue": "value"})
    df["flow"] = df["flow"].map({"M": "Import", "X": "Export"})
    return df.dropna(subset=["flow"])


def _cache_file(rc, partners, years):
    h = hashlib.md5(f"{rc}|{sorted(partners)}|{sorted(years)}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{rc}_{h}.csv"


def fetch_trade(key, reporter_codes, partner_codes, years):
    """Un fichier de cache par pays déclarant : un pays déjà téléchargé ne coûte plus d'appel."""
    CACHE_DIR.mkdir(exist_ok=True)
    years, partners = sorted(years), sorted(partner_codes)
    frames, missing = [], []
    for rc in reporter_codes:
        f = _cache_file(rc, partners, years)
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            missing.append(rc)
    for batch in _chunks(missing, REPORTER_BATCH):
        try:
            df = _call_comtrade(key, batch, partners, years)
        except DataError:
            if len(batch) == 1:
                raise
            df = pd.concat([_call_comtrade(key, [rc], partners, years) for rc in batch])
        for rc in batch:
            sub = df[df["reporter_code"] == rc]
            if not sub.empty:
                sub.to_csv(_cache_file(rc, partners, years), index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["year", "reporter_code", "partner_code", "flow", "value"])
    out = pd.concat(frames, ignore_index=True)
    out["reporter"] = out["reporter_code"].map(CODE2NAME)
    out["partner"] = out["partner_code"].map(lambda c: "World" if c == 0 else CODE2NAME.get(c))
    return out.dropna(subset=["reporter", "partner"])


def api_world(key, countries, y0, y1):
    df = fetch_trade(key, [COUNTRIES[c][0] for c in countries], [0], range(y0, y1 + 1))
    df["iso3"] = df["reporter"].map(NAME2ISO)
    return df[["year", "reporter", "iso3", "partner", "flow", "value"]]


def api_bilateral(key, members, y0, y1):
    codes = [COUNTRIES[c][0] for c in members]
    df = fetch_trade(key, codes, codes + [0], range(y0, y1 + 1))
    return df[["year", "reporter", "partner", "flow", "value"]]


# ----------------------------------------------------------------------------
# Couche données : CSV locaux
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Lecture du CSV (données mondiales)…")
def csv_world(path, mtime):
    d = pd.read_csv(path, encoding="latin1", low_memory=False)
    d = d[["RefYear", "ReporterDesc", "ReporterISO", "FlowDesc", "PartnerDesc", "PrimaryValue"]]
    d.columns = ["year", "reporter", "iso3", "flow", "partner", "value"]
    d["value"] = pd.to_numeric(d["value"].astype(str).str.replace(",", ""), errors="coerce")
    d = d[(d["partner"] == "World") & d["flow"].isin(["Export", "Import"])]
    return d.reset_index(drop=True)


@st.cache_data(show_spinner="Lecture du CSV (données bilatérales)…")
def csv_bilateral(path, mtime):
    d = pd.read_csv(path, encoding="latin1", low_memory=False,
                    usecols=["reporter", "partner", "trade_value_usd", "year", "trade_flow"])
    d = d.rename(columns={"trade_value_usd": "value", "trade_flow": "flow"})
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d[d["flow"].isin(["Export", "Import"])].reset_index(drop=True)


# ----------------------------------------------------------------------------
# PIB (Banque mondiale)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Téléchargement du PIB (Banque mondiale)…", ttl=24 * 3600)
def fetch_gdp(iso3_list, y0, y1):
    url = WB_URL.format(codes=";".join(iso3_list))
    r = requests.get(url, params={"format": "json", "per_page": 5000, "date": f"{y0}:{y1}"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    if len(js) < 2 or not js[1]:
        raise DataError("La Banque mondiale n'a renvoyé aucune donnée de PIB.")
    df = pd.DataFrame([{"iso3": x["countryiso3code"], "year": int(x["date"]), "gdp": x["value"]} for x in js[1]])
    return df.dropna(subset=["gdp"])


# ----------------------------------------------------------------------------
# Calcul des indicateurs
# ----------------------------------------------------------------------------
def wide_country(world, countries, with_gdp):
    d = world[world["reporter"].isin(countries)]
    p = d.pivot_table(index=["reporter", "iso3", "year"], columns="flow", values="value", aggfunc="sum").reset_index()
    for c in ("Export", "Import"):
        if c not in p:
            p[c] = np.nan
    if with_gdp:
        y0, y1 = int(p["year"].min()), int(p["year"].max())
        gdp = fetch_gdp(tuple(sorted(p["iso3"].dropna().unique())), y0, y1)
        p = p.merge(gdp, on=["iso3", "year"], how="left")
    return p.sort_values(["reporter", "year"]).reset_index(drop=True)


def country_indicator(ind, p):
    p = p.copy()
    X, M = p["Export"], p["Import"]
    if ind == IND_IP:
        p["value"] = M / (p["gdp"] - X + M) * 100
    elif ind == IND_TDI:
        p["value"] = (X + M) / p["gdp"] * 100
    elif ind == IND_NTB:
        p["value"] = (X - M) / (X + M)
    elif ind == IND_COV:
        p["value"] = X / M
    elif ind == IND_EP:
        p["value"] = X / p["gdp"] * 100
    elif ind == IND_MPM:
        g = p.groupby("reporter")
        p["value"] = g["Import"].diff() / g["gdp"].diff()
    elif ind == IND_EXP:
        p["value"] = X
        p["growth"] = p.groupby("reporter")["Export"].pct_change(fill_method=None) * 100
    return p


def intra_share(bil, members, flows, y0, y1):
    d = bil[bil["reporter"].isin(members) & bil["flow"].isin(flows) &
            bil["partner"].isin(members + ["World"]) & bil["year"].between(y0, y1)]
    intra = d[d["partner"] != "World"].groupby("year")["value"].sum()
    world = d[d["partner"] == "World"].groupby("year")["value"].sum()
    n = d[d["partner"] == "World"].groupby("year")["reporter"].nunique()
    out = pd.DataFrame({"share": intra / world.replace(0, np.nan) * 100, "n_reporters": n}).reset_index()
    return out.dropna(subset=["share"])


# ----------------------------------------------------------------------------
# Graphiques
# ----------------------------------------------------------------------------
def _style(fig, ytitle, hline=None):
    fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title=ytitle,
                      xaxis_title="Année", xaxis=dict(tickmode="linear", dtick=1),
                      legend_title_text="")
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="gray")
    return fig


def line(df, y, color, title, ytitle, hline=None):
    return _style(px.line(df, x="year", y=y, color=color, markers=True, title=title), ytitle, hline)


def table_download(df, name, pivot_cols=None):
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ Télécharger (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{name}.csv", mime="text/csv", key=f"dl_{name}")


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Indicateurs de commerce extérieur", page_icon="📊", layout="wide")
st.title("📊 Indicateurs de commerce extérieur")


def get_key():
    """Cherche la clé et renvoie (clé, infos de diagnostic). La clé n'est jamais affichée."""
    info = []
    try:
        k = st.secrets.get("COMTRADE_KEY", "")
    except Exception:
        k = ""
    info.append(f"st.secrets : {'clé trouvée' if k else 'rien'}")
    if not k:
        base = Path(__file__).resolve().parent
        info.append(f"Dossier de app.py : {base}")
        info.append(f"Dossier de lancement : {Path.cwd()}")
        for c in (base / ".streamlit" / "secrets.toml", base / "secrets.toml"):
            info.append(f"{c.relative_to(base)} : {'existe' if c.exists() else 'absent'}")
            if c.exists() and not k:
                try:
                    import tomllib
                    text = c.read_text(encoding="utf-8-sig")
                    if "=" not in text:                      # fichier contenant seulement la clé
                        k = text.strip()
                    else:
                        d = tomllib.loads(text)
                        info.append("noms trouvés dans le fichier : " + ", ".join(d) if d else "fichier vide")
                        k = d.get("COMTRADE_KEY") or next(
                            (v for n, v in d.items() if isinstance(v, str) and "key" in n.lower()), "")
                except Exception as e:
                    info.append(f"fichier illisible : {e}")
    if not k:
        k = os.environ.get("COMTRADE_KEY", "")
        info.append(f"variable d'environnement : {'clé trouvée' if k else 'rien'}")
    k = "".join(str(k).split()).strip("\"'")
    info.append(f"Résultat : {'clé détectée (' + str(len(k)) + ' caractères)' if k else 'AUCUNE clé détectée'}")
    return k, info


with st.sidebar.form("params"):
    st.header("Paramètres")
    source = st.radio("Source des données", ["API UN Comtrade", "Fichiers CSV"])
    key_default, key_info = get_key()
    key_input = st.text_input("Clé Comtrade (si non définie dans secrets / variable d'environnement)",
                              type="password", value="", placeholder="déjà configurée" if key_default else "")
    csv_world_path = st.text_input("CSV mondial (indicateurs 1-7)", "TradeData_all-world-16-20.csv")
    csv_bil_path = st.text_input("CSV bilatéral (indicateurs 8-15)", "Section3_dataset.csv")
    y0, y1 = st.slider("Période affichée", 2010, 2025, (2016, 2020))
    inds = st.multiselect("Indicateurs", COUNTRY_INDS + REGION_INDS, default=[IND_IP, IND_XS])
    countries = st.multiselect("Pays (indicateurs 1 à 7)", sorted(COUNTRIES), default=DEFAULT_COUNTRIES)
    regions = st.multiselect("Régions (indicateurs 8 à 15)", list(REGION_PRESETS) + ["Groupe personnalisé"],
                             default=list(REGION_PRESETS))
    custom = st.multiselect("Membres du groupe personnalisé", sorted(COUNTRIES))
    submitted = st.form_submit_button("Calculer", type="primary")

with st.sidebar.expander("🔍 Diagnostic de la clé"):
    for line_ in key_info:
        st.text(line_)

if st.sidebar.button("🗑️ Vider le cache API"):
    for f in CACHE_DIR.glob("*.csv"):
        f.unlink()
    st.sidebar.success("Cache vidé.")

if submitted:
    st.session_state["p"] = dict(source=source, key=clean_key(key_input or key_default), wpath=csv_world_path,
                                 bpath=csv_bil_path, y0=y0, y1=y1, inds=inds, countries=countries,
                                 regions=regions, custom=custom)
if "p" not in st.session_state:
    st.info("Choisissez la source, les indicateurs et les pays dans le panneau de gauche, puis cliquez sur **Calculer**.")
    st.stop()

P = st.session_state["p"]
use_api = P["source"] == "API UN Comtrade"
if use_api and not P["key"]:
    st.error("Aucune clé Comtrade : ajoutez `COMTRADE_KEY` dans `.streamlit/secrets.toml`, "
             "dans une variable d'environnement, ou collez-la dans le champ du panneau.")
    st.stop()
if not P["inds"]:
    st.warning("Sélectionnez au moins un indicateur.")
    st.stop()

sel_country_inds = [i for i in P["inds"] if i in COUNTRY_INDS]
sel_region_inds = [i for i in P["inds"] if i in REGION_INDS]
Y0, Y1 = P["y0"], P["y1"]

groups = {}
for r in P["regions"]:
    members = P["custom"] if r == "Groupe personnalisé" else REGION_PRESETS[r]
    if members:
        groups[r] = list(members)

try:
    # ---- données par pays (indicateurs 1 à 7) ----
    wide = None
    if sel_country_inds:
        if not P["countries"]:
            st.warning("Sélectionnez au moins un pays pour les indicateurs 1 à 7.")
        else:
            need_gdp = any(i in GDP_INDS for i in sel_country_inds)
            with st.spinner("Chargement des données par pays…"):
                if use_api:
                    world = api_world(P["key"], P["countries"], Y0 - 1, Y1)   # 1 an de recul pour croissance / MPM
                else:
                    p = Path(P["wpath"])
                    world = csv_world(str(p), p.stat().st_mtime)
                wide = wide_country(world, P["countries"], need_gdp)
            missing = [c for c in P["countries"] if c not in set(wide["reporter"])]
            if missing:
                st.warning("Aucune donnée pour : " + ", ".join(missing))

    # ---- données bilatérales (indicateurs 8 à 15) ----
    bil = None
    if sel_region_inds:
        if not groups:
            st.warning("Sélectionnez au moins une région (ou des membres pour le groupe personnalisé).")
        else:
            all_members = sorted({m for g in groups.values() for m in g})
            with st.spinner("Chargement des données bilatérales…"):
                if use_api:
                    bil = {}
                    for r, mem in groups.items():
                        bil[r] = api_bilateral(P["key"], mem, Y0, Y1)
                else:
                    p = Path(P["bpath"])
                    full = csv_bilateral(str(p), p.stat().st_mtime)
                    bil = {r: full for r in groups}
except DataError as e:
    st.error(str(e))
    st.stop()
except (requests.RequestException, FileNotFoundError, KeyError, ValueError) as e:
    msg = str(e)
    if P.get("key"):
        msg = msg.replace(P["key"], "***")
    st.error(f"Impossible de charger les données : {msg}")
    st.stop()

tabs = st.tabs(P["inds"])
for tab, ind in zip(tabs, P["inds"]):
    with tab:
        # ===== indicateurs par pays =====
        if ind in COUNTRY_INDS:
            if wide is None or wide.empty:
                st.info("Pas de données.")
                continue
            res = country_indicator(ind, wide)
            res = res[res["year"].between(Y0, Y1)]
            if ind in GDP_INDS and res["gdp"].isna().all():
                st.error("PIB indisponible pour ces pays/années (Banque mondiale).")
                continue
            if ind == IND_EXP:
                res = res.assign(export_bn=res["Export"] / 1e9)
                st.plotly_chart(line(res, "export_bn", "reporter", "Valeur des exportations", "Milliards USD"),
                                width="stretch")
                fig = px.bar(res.dropna(subset=["growth"]), x="year", y="growth", color="reporter", barmode="group",
                             title="Taux de croissance annuel des exportations")
                st.plotly_chart(_style(fig, "Croissance (%)", 0), width="stretch")
                cols = ["reporter", "year", "Export", "growth"]
            else:
                titles = {
                    IND_IP: ("Indicateur de pénétration des importations", "IP (%)", None),
                    IND_TDI: ("Taux de dépendance commerciale", "TDI (%)", None),
                    IND_NTB: ("Solde commercial normalisé", "(X − M) / (X + M)", 0),
                    IND_COV: ("Ratio de couverture export / import", "X / M", 1),
                    IND_EP: ("Propension à exporter", "EP (%)", None),
                    IND_MPM: ("Propension marginale à importer", "MPM (ΔM / ΔPIB)", 0),
                }
                t, yt, hl = titles[ind]
                st.plotly_chart(line(res, "value", "reporter", f"{t} ({Y0}-{Y1})", yt, hl), width="stretch")
                cols = ["reporter", "year", "value"]
            tbl = res[cols].round(4).rename(columns={"value": ind.split("· ")[1]})
            table_download(tbl, "indicateur_" + ind.split(" ")[0].replace("/", "-"))

        # ===== indicateurs régionaux =====
        else:
            if not bil:
                st.info("Pas de données.")
                continue
            measures = {IND_XS: [("XS", ["Export"])], IND_MS: [("MS", ["Import"])],
                        IND_TS: [("TS", ["Export", "Import"])],
                        IND_ALL: [("XS", ["Export"]), ("MS", ["Import"]), ("TS", ["Export", "Import"])]}[ind]
            frames = []
            for r, mem in groups.items():
                found = set(bil[r]["reporter"].unique())
                absent = [m for m in mem if m not in found]
                if absent:
                    st.warning(f"{r} : pays absents des données : {', '.join(absent)}")
                for code, flows in measures:
                    s = intra_share(bil[r], [m for m in mem if m in found], flows, Y0, Y1)
                    s["region"], s["mesure"] = r, code
                    frames.append(s)
            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if out.empty:
                st.info("Aucune donnée pour cette sélection.")
                continue
            out["share"] = out["share"].round(2)
            if ind == IND_XS:
                fig = px.bar(out, x="year", y="share", color="region", barmode="group", text="share",
                             title=f"Part des exportations intra-régionales ({Y0}-{Y1})")
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                st.plotly_chart(_style(fig, "Part des exportations (%)"), width="stretch")
            elif ind in (IND_MS, IND_TS):
                lab = "importations" if ind == IND_MS else "échanges (X+M)"
                st.plotly_chart(line(out, "share", "region", f"Part des {lab} intra-régionaux ({Y0}-{Y1})",
                                     "Part (%)"), width="stretch")
            else:
                for r in groups:
                    sub = out[out["region"] == r]
                    st.plotly_chart(line(sub, "share", "mesure", f"Commerce intra-régional – {r} ({Y0}-{Y1})",
                                         "Part du commerce (%)"), width="stretch")
            st.caption("`n_reporters` = nombre de pays déclarants avec un total « World » cette année-là "
                       "(un pays sans donnée est ignoré, comme dans vos notebooks).")
            table_download(out[["region", "mesure", "year", "share", "n_reporters"]],
                           "regional_" + ind.split(" ")[0].replace("/", "-"))
