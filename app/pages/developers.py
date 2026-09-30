"""
developers.py — Developer Intelligence Hub v3
Competition-grade Steam Market Intelligence decision-support system.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

from src.config import PROJECT_ROOT

HUB_DIR  = PROJECT_ROOT / "models" / "dev_hub_v3"
ACCENT   = "#00f2fe"
ACCENT2  = "#4facfe"
SUCCESS_CLR = "#00ff88"
WARN_CLR    = "#ff7b72"
MED_CLR     = "#f59e0b"
TIER_CFG    = {
    "High":   {"color": SUCCESS_CLR, "icon": "🟢", "label": "High Commercial Traction"},
    "Medium": {"color": MED_CLR,     "icon": "🟡", "label": "Medium Commercial Traction"},
    "Low":    {"color": WARN_CLR,    "icon": "🔴", "label": "Low Commercial Traction"},
}
HUMAN = {
    "release_year":"Release Year","price":"Planned Price",
    "languages_count":"Language Support","cat_full_controller_support":"Controller Support",
    "dlc_count":"DLC Planned","full_audio_languages_count":"Full Audio Languages",
    "release_quarter":"Release Quarter","patforms_count":"Platforms",
    "genre_enc":"Genre","is_indie":"Studio (Indie)","tag_indie":"Indie Tag",
    "tag_action":"Action Tag","cat_single_player":"Single-player",
    "cat_multi_player":"Multiplayer","cat_co_op":"Co-op",
    "genre_simulation":"Simulation Genre","genre_rpg":"RPG Genre",
    "genre_casual":"Casual Genre","genre_adventure":"Adventure Genre",
    "genre_strategy":"Strategy Genre","genre_count":"Genre Count",
    "tag_simulation":"Simulation Tag","tag_rpg":"RPG Tag",
}
ALL_TAGS = [
    "Action","Adventure","RPG","Simulation","Strategy","Casual","Indie",
    "Open World","Story Rich","Roguelike","Survival","Horror","Sci-fi",
    "Fantasy","Puzzle","Platformer","Shooter","Sports","Racing",
]
PLAYTIME = {"< 2h": 1, "2–5h": 3, "5–10h": 7, "10–20h": 15, "20–50h": 35, "50h+": 70}
CAND_PRICES = [4.99, 9.99, 14.99, 19.99, 24.99, 29.99, 39.99, 59.99]
STD_PRICES  = [0.99,1.99,2.99,4.99,7.99,9.99,12.99,14.99,17.99,19.99,24.99,29.99,34.99,39.99,49.99,59.99]


@st.cache_resource(show_spinner="Loading Developer Intelligence models (V3)...")
def _load_hub():
    if not HUB_DIR.exists():
        return None
    d = {}
    for name in ["genre_encoder","genre_list","m01_demand_rf","m01_shap_explainer",
                 "m02_kmeans","m02_scaler","m02_feature_cols","m02_nn",
                 "m03_success_clf","m03_shap_explainer","m03_feature_cols"]:
        p = HUB_DIR / f"{name}.pkl"
        if p.exists():
            d[name] = joblib.load(p)
    for name in ["feature_meta","cluster_profiles"]:
        p = HUB_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                d[name] = json.load(f)
    p_nn_meta = HUB_DIR / "m02_nn_meta.parquet"
    if p_nn_meta.exists():
        d["m02_nn_meta"] = pd.read_parquet(p_nn_meta)
    return d


def _css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
.dh-title {
    font-family:'Inter',sans-serif; font-size:2.4rem; font-weight:800;
    background:linear-gradient(90deg,#00f2fe,#4facfe 60%,#00ff88);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;
}
.dh-sub { color:#8b9dc3; font-size:1rem; margin-bottom:28px; }
.gp {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:14px; padding:22px; margin-bottom:18px;
}
.gp-accent { border-color:rgba(0,242,254,0.25) !important; }
.metric-box { text-align:center; padding:14px 8px; }
.metric-val { font-size:1.9rem; font-weight:700; font-family:'Inter',sans-serif; }
.metric-lbl { font-size:0.72rem; color:#8b9dc3; text-transform:uppercase; letter-spacing:1.2px; margin-top:4px; }
.risk-item {
    padding:8px 12px; background:rgba(255,123,114,0.08);
    border-left:3px solid #ff7b72; border-radius:4px; margin:5px 0; font-size:0.9rem;
}
.good-item {
    padding:8px 12px; background:rgba(0,255,136,0.08);
    border-left:3px solid #00ff88; border-radius:4px; margin:5px 0; font-size:0.9rem;
}
.info-item {
    padding:8px 12px; background:rgba(79,172,254,0.08);
    border-left:3px solid #4facfe; border-radius:4px; margin:5px 0; font-size:0.9rem;
}
.shared-banner {
    background:linear-gradient(135deg,rgba(0,242,254,0.07),rgba(79,172,254,0.04));
    border:1px solid rgba(0,242,254,0.18); border-radius:12px; padding:14px 18px; margin-bottom:18px;
}
</style>
""", unsafe_allow_html=True)


def _metric(label, value, color=ACCENT):
    return (f'<div class="metric-box">'
            f'<div class="metric-val" style="color:{color}">{value}</div>'
            f'<div class="metric-lbl">{label}</div></div>')


def _build_row(profile: dict, ge, genre_list, num_feats, price=None):
    """Construct feature dict from shared profile for model inference."""
    row = {f: 0.0 for f in num_feats + ["genre_enc"]}
    g = profile.get("genre", "Other")
    if g in genre_list:
        row["genre_enc"] = float(ge.transform([[g]])[0][0])
    # Genre binary flags
    gmap = {"Action":"genre_action","Adventure":"genre_adventure","Casual":"genre_casual",
            "RPG":"genre_rpg","Simulation":"genre_simulation","Strategy":"genre_strategy"}
    if g in gmap:
        row[gmap[g]] = 1.0
    tags_lc = [t.lower() for t in profile.get("tags", [])]
    row["genre_count"] = 1.0 + len(profile.get("tags", []))
    # Tag flags
    for tf in ["tag_indie","tag_action","tag_casual","tag_adventure",
               "tag_rpg","tag_simulation","tag_strategy","tag_singleplayer"]:
        tname = tf.replace("tag_","")
        if tname in tags_lc:
            row[tf] = 1.0
    row["is_indie"] = 1.0 if profile.get("studio") == "Indie" else 0.0
    if profile.get("studio") == "Indie":
        row["tag_indie"] = 1.0
    # Category flags
    row["cat_single_player"] = float(profile.get("sp", True))
    row["cat_multi_player"]  = float(profile.get("mp", False))
    row["cat_co_op"]         = float(profile.get("coop", False))
    row["cat_full_controller_support"] = float(profile.get("controller", False))
    if profile.get("sp", True):
        row["tag_singleplayer"] = 1.0
    # Meta
    row["patforms_count"]             = float(sum([profile.get("win",True), profile.get("mac",False), profile.get("lin",False)]))
    row["languages_count"]            = float(profile.get("langs", 5))
    row["full_audio_languages_count"] = float(max(1, profile.get("langs", 5) // 4))
    row["has_english"]                = 1.0
    row["dlc_count"]                  = float(profile.get("dlc", 0))
    row["release_year"]               = float(profile.get("rel_year", 2026))
    row["release_quarter"]            = float(profile.get("rel_q", 2))
    if price is not None:
        row["price"] = float(price)
    return row


def _shap_top(explainer, row_df, cols, n=6):
    """Get top-N SHAP feature importances for a single row."""
    try:
        import shap
        sv = explainer.shap_values(row_df)
        if isinstance(sv, list):
            sv = np.abs(np.array(sv)).mean(axis=0)
        if len(np.array(sv).shape) == 2:
            sv = np.array(sv)[0]
        imp = pd.Series(np.abs(sv), index=cols).sort_values(ascending=False)
        return imp.head(n)
    except Exception:
        return pd.Series(dtype=float)


# =============================================================================
def render(df_app=None, models=None) -> None:
    _css()
    st.markdown('<div class="dh-title">Developer Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="dh-sub">Pre-launch Steam decision-support system built on 107,000+ historical game profiles.</div>', unsafe_allow_html=True)

    hub = _load_hub()
    if hub is None or "m01_demand_rf" not in hub:
        st.error("Developer Hub v3 models not found. Run: `python scripts/train_dev_hub_v3.py`")
        return

    ge          = hub["genre_encoder"]
    genre_list  = hub["genre_list"]
    m01         = hub["m01_demand_rf"]
    m01_shap    = hub["m01_shap_explainer"]
    m02_km      = hub["m02_kmeans"]
    m02_sc      = hub["m02_scaler"]
    m02_cols    = hub["m02_feature_cols"]
    m02_nn      = hub["m02_nn"]
    m02_nn_meta = hub["m02_nn_meta"]
    m03         = hub["m03_success_clf"]
    m03_shap    = hub["m03_shap_explainer"]
    m03_cols    = hub["m03_feature_cols"]
    fmeta       = hub["feature_meta"]
    cluster_prf = hub.get("cluster_profiles", {})

    num_feats  = fmeta["num_feats"]
    base_cols  = fmeta["base_cols"]
    top_genres = fmeta["top_genres"]

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED GAME PROFILE
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📋  Shared Game Profile — set once, used by all models", expanded=True):
        st.markdown('<div class="shared-banner">Define your pre-launch game features. The models will only use pre-launch available variables to ensure no data leakage.</div>', unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            sh_genre        = st.selectbox("Primary Genre", top_genres + ["Other"], key="sh_genre")
            sh_tags         = st.multiselect("Key Tags (up to 8)", ALL_TAGS, default=["Action","Indie"], max_selections=8, key="sh_tags")
            sh_playtime_str = st.selectbox("Expected Playtime", list(PLAYTIME.keys()), index=3, key="sh_pt")
            sh_playtime     = PLAYTIME[sh_playtime_str]
        with pc2:
            st.markdown("**Player Features**")
            sh_sp   = st.checkbox("Single-player",     value=True,  key="sh_sp")
            sh_mp   = st.checkbox("Multiplayer",       value=False, key="sh_mp")
            sh_coop = st.checkbox("Co-op",             value=False, key="sh_coop")
            sh_pvp  = st.checkbox("PvP",               value=False, key="sh_pvp")
            sh_ctrl = st.checkbox("Controller Support",value=False, key="sh_ctrl")
            sh_studio = st.radio("Studio Tier", ["Indie","AA","AAA"], horizontal=True, key="sh_studio")
        with pc3:
            st.markdown("**Target Platforms**")
            sh_win  = st.checkbox("Windows", value=True,  key="sh_win")
            sh_mac  = st.checkbox("macOS",   value=False, key="sh_mac")
            sh_lin  = st.checkbox("Linux",   value=False, key="sh_lin")
            dlc_map = {"None": 0, "1–2 DLC": 1, "3+ DLC": 4}
            sh_dlc_str = st.radio("DLC Planned", list(dlc_map.keys()), horizontal=True, key="sh_dlc")
            sh_dlc  = dlc_map[sh_dlc_str]
            sh_langs = st.slider("Language Support", 1, 25, 5, key="sh_langs")
            col_y, col_q = st.columns(2)
            with col_y: sh_year = st.number_input("Release Year", 2025, 2030, 2026, key="sh_year")
            with col_q: sh_q    = st.selectbox("Quarter", ["Q1","Q2","Q3","Q4"], index=1, key="sh_q")

    sh_profile = dict(
        genre=sh_genre, tags=sh_tags, sp=sh_sp, mp=sh_mp, coop=sh_coop,
        pvp=sh_pvp, controller=sh_ctrl, studio=sh_studio,
        win=sh_win, mac=sh_mac, lin=sh_lin,
        dlc=sh_dlc, langs=sh_langs, rel_year=int(sh_year),
        rel_q=int(sh_q.replace("Q",""))
    )

    tabs = st.tabs([
        "💰 01 · Demand & Pricing",
        "🗺️ 02 · Market Segment",
        "🏆 03 · Commercial Traction",
        "⚡ 04 · Scenario Engine",
    ])

    # =========================================================================
    # TAB 01 — PRICE OPTIMIZATION
    # =========================================================================
    with tabs[0]:
        st.markdown("### 01 · Price & Demand Regression")
        st.markdown("*Predicts historical demand/ownership and evaluates price scenarios.*")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="gp">', unsafe_allow_html=True)
            st.markdown("##### Candidate Price Window")
            p1_min, p1_max = st.select_slider(
                "Test range ($)",
                options=CAND_PRICES,
                value=(4.99, 39.99),
                key="t1_range",
            )
            run_t1 = st.button("Run Demand Simulation", type="primary",
                               use_container_width=True, key="run_t1")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if run_t1:
                prices = [p for p in CAND_PRICES if p1_min <= p <= p1_max] or [p1_min]
                results = []
                for cp in prices:
                    prow = _build_row(sh_profile, ge, genre_list, num_feats, price=cp)
                    pdf = pd.DataFrame([prow])[base_cols].fillna(0).astype(float)
                    pred_log_owners = float(m01.predict(pdf)[0])
                    pred_owners = np.expm1(pred_log_owners)
                    rev_proxy = pred_owners * cp
                    results.append({
                        "Price": cp,
                        "Expected Owners": pred_owners,
                        "Revenue Proxy": rev_proxy,
                        "pdf": pdf # Store for SHAP
                    })
                
                # Find best price by revenue proxy
                best = max(results, key=lambda x: x["Revenue Proxy"])
                best_price = best["Price"]
                
                st.markdown('<div class="gp gp-accent">', unsafe_allow_html=True)
                st.markdown("#### Simulation Results")
                mc1, mc2, mc3 = st.columns(3)
                mc1.markdown(_metric("Recommended Price", f"${best_price:.2f}", ACCENT), unsafe_allow_html=True)
                mc2.markdown(_metric("Max Revenue Proxy", f"${best['Revenue Proxy']:,.0f}", ACCENT2), unsafe_allow_html=True)
                mc3.markdown(_metric("Est. Owners (at Rec Price)", f"{best['Expected Owners']:,.0f}", "#8b9dc3"), unsafe_allow_html=True)

                st.markdown("##### Price Scenario Analysis")
                rows_tbl = []
                for r in results:
                    diff_rev = (r["Revenue Proxy"] - best["Revenue Proxy"]) / best["Revenue Proxy"] * 100
                    rows_tbl.append({
                        "Candidate Price": f"${r['Price']:.2f}",
                        "Est. Owners": f"{r['Expected Owners']:,.0f}",
                        "Gross Rev Proxy": f"${r['Revenue Proxy']:,.0f}",
                        "vs Optimal Rev": f"{diff_rev:+.1f}%",
                    })
                st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

                # Charts
                st.markdown("##### Demand vs. Revenue Trade-off")
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(
                    x=[f"${r['Price']:.2f}" for r in results],
                    y=[r['Expected Owners'] for r in results],
                    name="Est. Owners", marker_color="#8b9dc3", yaxis="y1"
                ))
                fig1.add_trace(go.Scatter(
                    x=[f"${r['Price']:.2f}" for r in results],
                    y=[r['Revenue Proxy'] for r in results],
                    name="Revenue Proxy", mode="lines+markers",
                    line=dict(color=ACCENT, width=3),
                    marker=dict(size=10, color=[SUCCESS_CLR if r['Price']==best_price else ACCENT for r in results]),
                    yaxis="y2"
                ))
                fig1.update_layout(
                    height=300, margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Owners", showgrid=False),
                    yaxis2=dict(title="Revenue Proxy ($)", overlaying="y", side="right", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig1, use_container_width=True)

                # SHAP explanation
                shap_imp = _shap_top(m01_shap, best["pdf"], base_cols)
                if not shap_imp.empty:
                    st.markdown("##### Primary Drivers of Demand")
                    st.caption("These pre-launch features had the strongest influence on the model's demand prediction.")
                    for feat, val in shap_imp.items():
                        if feat == "price": continue
                        hname = HUMAN.get(feat, feat)
                        st.markdown(f'<div class="info-item">📊 <b>{hname}</b> — SHAP impact: {val:.3f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Set your candidate price window and click **Run Demand Simulation**.")

    # =========================================================================
    # TAB 02 — MARKET SEGMENTATION
    # =========================================================================
    with tabs[1]:
        st.markdown("### 02 · K-Means Market Archetypes & Competitor Discovery")
        st.markdown("*Identifies overlapping Steam market archetypes.*")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="gp">', unsafe_allow_html=True)
            st.markdown("##### Target Price for Archetype")
            t2_price = st.slider("Target Price ($)", 0.99, 60.0, 14.99, key="t2_price")
            run_t2   = st.button("Identify Market Archetype", type="primary",
                                  use_container_width=True, key="run_t2")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if run_t2:
                row2 = _build_row(sh_profile, ge, genre_list, num_feats, price=t2_price)
                row2_df = pd.DataFrame([row2])[m02_cols].fillna(0).astype(float)
                row2_sc = m02_sc.transform(row2_df)
                cid     = str(int(m02_km.predict(row2_sc)[0]))
                cp2     = cluster_prf.get(cid, {})

                if not cp2:
                    st.error("Cluster profile not found. Re-run training script.")
                else:
                    st.markdown('<div class="gp gp-accent">', unsafe_allow_html=True)
                    st.markdown(f"#### Market Archetype: **{cp2['segment_name']}**")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.markdown(_metric("Median Archetype Price", f"${cp2['median_price']:.2f}", ACCENT), unsafe_allow_html=True)
                    mc2.markdown(_metric("Dominant Genre", f"{cp2['top_genres'][0]}", ACCENT2), unsafe_allow_html=True)
                    mc3.markdown(_metric("Archetype Size", f"{cp2['size']:,} games", "#8b9dc3"), unsafe_allow_html=True)

                    # Closest games (Nearest Neighbors)
                    st.markdown("##### Closest Historical Games")
                    distances, indices = m02_nn.kneighbors(row2_sc)
                    nn_df = m02_nn_meta.iloc[indices[0]].copy()
                    st.dataframe(nn_df[["name", "price", "primary_genre", "owners_mid"]], use_container_width=True, hide_index=True)

                    st.markdown("##### Segment Profile")
                    char1, char2 = st.columns(2)
                    with char1:
                        st.markdown(f"**Top Genres:** {', '.join(cp2.get('top_genres', []))}")
                        st.markdown(f"**Multiplayer Focus:** {cp2['pct_mp']*100:.0f}% MP, {cp2['pct_coop']*100:.0f}% Co-op")
                        st.markdown(f"**Top Tags:** {', '.join(cp2.get('top_tags', []))}")
                    with char2:
                        st.markdown(f"**Indie Presence:** {cp2['pct_indie']*100:.0f}%")
                        st.markdown(f"**Avg DLCs:** {cp2['avg_dlc']:.1f}")
                        pdiff = t2_price - cp2["median_price"]
                        if pdiff > 2:
                            st.markdown(f'<div class="info-item">🎯 Pricing ${pdiff:.2f} above segment median. Differentiation required.</div>', unsafe_allow_html=True)
                        elif pdiff < -2:
                            st.markdown(f'<div class="info-item">🎯 Pricing ${-pdiff:.2f} below segment median. Volume strategy.</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="info-item">🎯 Pricing aligns with segment median.</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Set a target price and click **Identify Market Segment**.")

    # =========================================================================
    # TAB 03 — SUCCESS PREDICTION
    # =========================================================================
    with tabs[2]:
        st.markdown("### 03 · Historical Commercial Traction Classification")
        st.markdown("*Predicts historical commercial-traction tiers from pre-launch metadata.*")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="gp">', unsafe_allow_html=True)
            st.markdown("##### Planned Price")
            t3_price = st.slider("Planned Price ($)", 0.99, 60.0, 14.99, key="t3_price")
            run_t3   = st.button("Predict Traction Tier", type="primary",
                                  use_container_width=True, key="run_t3")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if run_t3:
                row3    = _build_row(sh_profile, ge, genre_list, num_feats, price=t3_price)
                row3_df = pd.DataFrame([row3])[m03_cols].fillna(0).astype(float)
                tier_n  = int(m03.predict(row3_df)[0])
                probs   = m03.predict_proba(row3_df)[0]
                tier    = ["Low", "Medium", "High"][tier_n]
                tc      = TIER_CFG[tier]

                st.markdown('<div class="gp gp-accent">', unsafe_allow_html=True)
                st.markdown("#### Commercial Traction Prediction")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.markdown(_metric("Predicted Tier", f"{tc['icon']} {tier}", tc["color"]), unsafe_allow_html=True)
                mc2.markdown(_metric("P(Low)",    f"{probs[0]*100:.1f}%", WARN_CLR),    unsafe_allow_html=True)
                mc3.markdown(_metric("P(Medium)", f"{probs[1]*100:.1f}%", MED_CLR),     unsafe_allow_html=True)
                mc4.markdown(_metric("P(High)",   f"{probs[2]*100:.1f}%", SUCCESS_CLR), unsafe_allow_html=True)

                fig3p = go.Figure(go.Bar(
                    x=["Low", "Medium", "High"],
                    y=[p * 100 for p in probs],
                    marker_color=[WARN_CLR, MED_CLR, SUCCESS_CLR],
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition="auto",
                ))
                fig3p.update_layout(
                    height=180, margin=dict(l=0,r=0,t=20,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Probability (%)", showlegend=False,
                )
                st.plotly_chart(fig3p, use_container_width=True)

                # SHAP Risk/Strength analysis dynamically
                shap_imp = _shap_top(m03_shap, row3_df, m03_cols, n=8)
                if not shap_imp.empty:
                    st.markdown("##### Core Influencing Factors")
                    st.caption("SHAP feature importances driving this specific prediction:")
                    
                    strengths, risks = [], []
                    # Simple heuristic: if probability of high/med is good, treat top drivers as strengths, else risks
                    for feat, val in shap_imp.items():
                        hname = HUMAN.get(feat, feat)
                        fval = row3.get(feat, 0)
                        msg = f"{hname} = {fval}"
                        if probs[2] > 0.4:  # Likely high traction
                            strengths.append(msg)
                        else:
                            risks.append(msg)
                            
                    col_s, col_r = st.columns(2)
                    with col_s:
                        if strengths:
                            st.markdown("**Key Positives:**")
                            for s in strengths[:4]: st.markdown(f'<div class="good-item">✅ {s}</div>', unsafe_allow_html=True)
                    with col_r:
                        if risks:
                            st.markdown("**Potential Risks:**")
                            for r in risks[:4]: st.markdown(f'<div class="risk-item">⚠️ {r}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Set your planned price and click **Predict Traction Tier**.")

    # =========================================================================
    # TAB 04 — SCENARIO ENGINE
    # =========================================================================
    with tabs[3]:
        st.markdown("### 04 · Live Scenario Engine")
        st.markdown("*Simulates alternative configurations using the trained models.*")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown('<div class="gp">', unsafe_allow_html=True)
            t4_obj = st.radio(
                "Optimization Objective",
                ["Balanced (Demand + Success Prob)", "Maximize Demand (Owners)", "Maximize Revenue Proxy", "Maximize High-Tier Prob"],
                key="t4_obj",
            )
            run_t4 = st.button("Run Scenario Analysis", type="primary",
                               use_container_width=True, key="run_t4")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            if run_t4:
                candidates = CAND_PRICES

                results = []
                for cp in candidates:
                    r4 = _build_row(sh_profile, ge, genre_list, num_feats, price=cp)
                    
                    # M01 Demand
                    pdf = pd.DataFrame([r4])[base_cols].fillna(0).astype(float)
                    pred_log_owners = float(m01.predict(pdf)[0])
                    owners = np.expm1(pred_log_owners)
                    rev = owners * cp
                    
                    # M03 Traction
                    cdf = pd.DataFrame([r4])[m03_cols].fillna(0).astype(float)
                    probs = m03.predict_proba(cdf)[0]
                    tier_idx = int(m03.predict(cdf)[0])
                    p_high = float(probs[2])
                    
                    # Score based on objective
                    if "Balanced" in t4_obj:
                        score = (owners / 50000) * 0.5 + p_high * 0.5
                    elif "Demand" in t4_obj:
                        score = owners
                    elif "Revenue" in t4_obj:
                        score = rev
                    else:
                        score = p_high
                        
                    results.append({
                        "Price": cp,
                        "Est. Owners": owners,
                        "Revenue Proxy": rev,
                        "P(High)": p_high,
                        "Score": score,
                        "Tier": ["Low", "Medium", "High"][tier_idx]
                    })

                rdf = pd.DataFrame(results)
                best  = rdf.loc[rdf["Score"].idxmax()]
                alts  = rdf[rdf["Price"] != best["Price"]].sort_values("Score", ascending=False)
                alt1  = alts.iloc[0] if len(alts) >= 1 else best
                alt2  = alts.iloc[1] if len(alts) >= 2 else alt1

                st.markdown('<div class="gp gp-accent">', unsafe_allow_html=True)
                st.markdown("#### Scenario Results")
                bc1, bc2, bc3 = st.columns(3)
                bc1.markdown(_metric("Optimal Sweet Spot", f"${best['Price']:.2f}", ACCENT), unsafe_allow_html=True)
                bc2.markdown(_metric("Expected Demand", f"{best['Est. Owners']:,.0f}", ACCENT2), unsafe_allow_html=True)
                bc3.markdown(_metric("P(High Traction)", f"{best['P(High)']*100:.1f}%", SUCCESS_CLR), unsafe_allow_html=True)

                st.markdown("##### Strategy Alternatives")
                st.markdown(
                    f'<div class="good-item">🥇 <b>Sweet Spot: ${best["Price"]:.2f}</b> '
                    f'— {best["Tier"]} tier | Demand: {best["Est. Owners"]:,.0f} '
                    f'| Rev Proxy: ${best["Revenue Proxy"]:,.0f}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="info-item">🥈 <b>Alt #1: ${alt1["Price"]:.2f}</b> '
                    f'— {alt1["Tier"]} tier | Demand: {alt1["Est. Owners"]:,.0f} '
                    f'| Rev Proxy: ${alt1["Revenue Proxy"]:,.0f}</div>', unsafe_allow_html=True)
                if alt2["Price"] != alt1["Price"]:
                    st.markdown(
                        f'<div class="info-item">🥉 <b>Alt #2: ${alt2["Price"]:.2f}</b> '
                        f'— {alt2["Tier"]} tier | Demand: {alt2["Est. Owners"]:,.0f} '
                        f'| Rev Proxy: ${alt2["Revenue Proxy"]:,.0f}</div>', unsafe_allow_html=True)

                st.markdown("##### Complete Scenario Comparison")
                tbl_rows = []
                for _, r in rdf.iterrows():
                    lbl = "✅ Sweet Spot" if r["Price"] == best["Price"] else ""
                    tbl_rows.append({
                        "Scenario Price": f"${r['Price']:.2f}",
                        "Traction Tier": r['Tier'],
                        "Est. Demand": f"{r['Est. Owners']:,.0f}",
                        "Revenue Proxy": f"${r['Revenue Proxy']:,.0f}",
                        "P(High)": f"{r['P(High)']*100:.1f}%",
                        "": lbl
                    })
                st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Select an optimization objective and click **Run Scenario Analysis**.")

    st.markdown("---")

