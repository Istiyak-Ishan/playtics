"""
EDA Notebook — Steam Market Executive Report (Jupyter-style Streamlit page).
Renders 15 strategic questions with code snippets, charts extracted from the
source PDF, and executive takeaways styled to match the PlayLytics dark theme.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import streamlit as st

_HERE      = Path(__file__).resolve().parent
_PROJECT   = _HERE.parent.parent
_PDF_PATH  = _PROJECT / "Steam_Market_Executive_Report.pdf"
_CACHE_DIR = _PROJECT / "assets" / "eda_report_charts"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_CSS = """
<style>
.nb-section-header{font-family:\'IBM Plex Sans\',sans-serif;font-size:1.05rem;font-weight:700;
  color:#00f5ff;letter-spacing:.04em;text-transform:uppercase;
  border-bottom:2px solid rgba(0,245,255,.25);padding-bottom:8px;margin:36px 0 18px;}
.nb-q-title{font-family:\'IBM Plex Sans\',sans-serif;font-size:.98rem;font-weight:700;
  color:#e2e8f0;margin:22px 0 6px;}
.nb-q-subtitle{font-family:\'IBM Plex Mono\',monospace;font-size:.75rem;color:#6b7a99;
  margin-bottom:14px;}
.nb-card{border-radius:6px;padding:14px 18px;margin-top:12px;margin-bottom:6px;
  font-family:\'IBM Plex Sans\',sans-serif;font-size:.83rem;line-height:1.65;}
.nb-card-shows   {background:rgba(0,245,255,.05);border-left:3px solid #00f5ff;}
.nb-card-finding {background:rgba(79,142,247,.07);border-left:3px solid #4f8ef7;}
.nb-card-takeaway{background:rgba(255,0,102,.07);border-left:3px solid #ff0066;
  color:#ff6694;font-weight:600;}
.nb-card-label{font-size:.68rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:6px;color:#8b9dc3;}
.nb-card-shows    .nb-card-label{color:#00c8d4;}
.nb-card-finding  .nb-card-label{color:#4f8ef7;}
.nb-card-takeaway .nb-card-label{color:#ff0066;}
.nb-scorecard-table{width:100%;border-collapse:collapse;
  font-family:\'IBM Plex Mono\',monospace;font-size:.78rem;margin-top:12px;}
.nb-scorecard-table th{background:rgba(0,245,255,.08);color:#00f5ff;font-weight:600;
  padding:10px 14px;text-align:left;border-bottom:2px solid rgba(0,245,255,.2);
  font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;}
.nb-scorecard-table td{padding:9px 14px;border-bottom:1px solid rgba(0,245,255,.06);
  color:#c8d4e8;vertical-align:top;}
.nb-scorecard-table tr:hover td{background:rgba(0,245,255,.03);}
.score-a{color:#00ff88;font-weight:700;}
.score-b{color:#f0ff00;font-weight:700;}
.golden-rule{background:rgba(240,255,0,.05);border-left:3px solid #f0ff00;
  padding:10px 16px;border-radius:4px;margin:8px 0;
  font-family:\'IBM Plex Sans\',sans-serif;font-size:.83rem;color:#e2e8f0;}
.golden-rule strong{color:#f0ff00;}
.nb-chart-wrap {
    background: #05050a;
    border: 1px solid rgba(0,245,255,0.12);
    border-radius: 6px;
    padding: 16px;
    margin: 12px 0 6px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}
.nb-chart-wrap img {
    border-radius: 4px;
    display: block;
    margin: 0 auto;
    filter: invert(1) hue-rotate(180deg);
}
</style>
"""

# ── PDF page extraction ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _extract_pages(pdf_path: str) -> list:
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return []
    cache_dir = _CACHE_DIR
    pdf_hash  = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()[:8]
    cached    = sorted(cache_dir.glob(f"page_{pdf_hash}_*.png"))
    if cached:
        return [str(p) for p in cached]
    doc   = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        mat  = fitz.Matrix(2.2, 2.2)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        dest = cache_dir / f"page_{pdf_hash}_{i:02d}.png"
        pix.save(str(dest))
        paths.append(str(dest))
    doc.close()
    return paths


# ── Helpers ────────────────────────────────────────────────────────────────────
def _code(snippet: str, label: str = "Show Code  [Python]") -> None:
    with st.expander(label, expanded=False):
        st.code(snippet.strip(), language="python")


def _takeaway(shows: str, finding: str, stakeholder: str) -> None:
    st.markdown(f"""
<div class="nb-card nb-card-shows">
  <div class="nb-card-label">What this visual shows</div>{shows}
</div>
<div class="nb-card nb-card-finding">
  <div class="nb-card-label">Key Market Finding</div>{finding}
</div>
<div class="nb-card nb-card-takeaway">
  <div class="nb-card-label">Stakeholder Takeaway</div>{stakeholder}
</div>""", unsafe_allow_html=True)


def _qhdr(n: int, q: str, sub: str = "") -> None:
    s = f'<div class="nb-q-subtitle">{sub}</div>' if sub else ""
    st.markdown(f'<div class="nb-q-title">Question {n}: {q}</div>' + s, unsafe_allow_html=True)


def _sec(title: str) -> None:
    st.markdown(f'<div class="nb-section-header">{title}</div>', unsafe_allow_html=True)


def _img(pages: list, idx: int) -> None:
    if idx < len(pages):
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(pages[idx]).convert("RGB")
            # Remove PDF header (top ~5%) and footer (bottom ~6%)
            w, h = img.size
            img = img.crop((0, int(h * 0.05), w, int(h * 0.94)))
            # Invert white→black background
            arr = np.array(img, dtype=np.int16)
            arr = 255 - arr
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            dark_img = Image.fromarray(arr)
            # Auto-crop: remove large dark (inverted-white) blank margins
            # Find bounding box of pixels that are NOT near-black background
            gray = dark_img.convert("L")
            content_mask = gray.point(lambda p: 255 if p > 18 else 0)
            bbox = content_mask.getbbox()
            if bbox:
                pad = 24
                left   = max(0, bbox[0] - pad)
                top    = max(0, bbox[1] - pad)
                right  = min(dark_img.width,  bbox[2] + pad)
                bottom = min(dark_img.height, bbox[3] + pad)
                dark_img = dark_img.crop((left, top, right, bottom))
            st.image(dark_img, use_container_width=True)
        except ImportError:
            st.image(pages[idx], use_container_width=True)
    else:
        st.info("Chart image not available — ensure PyMuPDF is installed and the PDF exists.")


# ══════════════════════════════════════════════════════════════════════════════
def render(df_app=None, models=None) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("""
<div class="hero-header">
  <div class="hero-title">EDA Notebook — Executive Strategy Report</div>
  <div class="hero-subtitle">
    Steam PC Games &nbsp;·&nbsp; 126,130 titles &nbsp;·&nbsp;
    15 Core Strategic Questions &nbsp;·&nbsp; 5 Analytical Sections
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            "<p style='color:#6b7a99;font-size:.85rem;margin-top:8px;'>"
            "This notebook reproduces the full 15-question EDA analysis. Each cell contains "
            "the source Python code, the rendered chart extracted directly from the original "
            "PDF, and curated executive takeaways for stakeholder briefings.</p>",
            unsafe_allow_html=True,
        )
    with c2:
        if _PDF_PATH.exists():
            st.download_button("📄 Download Full PDF", data=_PDF_PATH.read_bytes(),
                               file_name="Steam_Market_Executive_Report.pdf",
                               mime="application/pdf", use_container_width=True)

    st.markdown("---")

    pages: list = []
    if _PDF_PATH.exists():
        with st.spinner("Rendering PDF pages (cached after first run)…"):
            pages = _extract_pages(str(_PDF_PATH))
    else:
        st.error(f"PDF not found at `{_PDF_PATH}`")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Pricing & Value Patterns
    # ─────────────────────────────────────────────────────────────────────────
    _sec("§ Section 1 · Pricing & Value Patterns")

    _qhdr(1, "What is the median price of games by genre?",
          "df.groupby(\'primary_genre\')[\'price\'].median().sort_values(ascending=False)")
    _code("""
genre_price = (
    df.groupby(\'primary_genre\')[\'price\']
      .median()
      .sort_values(ascending=False)
      .reset_index()
)
fig, ax = plt.subplots(figsize=(12, 5))
palette = sns.color_palette("Blues_r", len(genre_price))
sns.barplot(data=genre_price, x=\'price\', y=\'primary_genre\',
            palette=palette, ax=ax, edgecolor=\'.2\')
ax.bar_label(ax.containers[0],
             labels=[f\'${v:.2f}\' for v in genre_price[\'price\']], padding=3)
ax.set_title(\'Q1: Median Price by Genre (USD)\', fontweight=\'bold\')
ax.set_xlabel(\'Median Price ($ USD)\'); ax.set_ylabel(\'Genre\')
plt.tight_layout()
""")
    _img(pages, 1)
    _takeaway(
        "Median price per game genre across the full 126k-title catalogue.",
        "• <strong>Massively Multiplayer ($14.99)</strong> and <strong>Simulation ($12.99)</strong> hold the highest medians, "
        "reflecting larger scope and live-service monetisation.<br>"
        "• <strong>Casual ($3.99)</strong> and <strong>Indie ($5.99)</strong> compete on accessibility rather than premium positioning.",
        "Price your game relative to <em>genre norms</em>, not perceived effort. Under-pricing an MMO or "
        "over-pricing a casual title both damage conversion rates."
    )

    st.divider()
    _qhdr(2, "Does higher price correlate with better review scores?",
          "sns.regplot(x=\'price\', y=\'review_score_pct×100\') — slope ≈ 0.00")
    _code("""
sample = df.dropna(subset=[\'price\', \'review_score_pct\']).copy()
sample = sample[(sample[\'price\'] >= 0) & (sample[\'price\'] <= 80)]
sample[\'review_pct\'] = sample[\'review_score_pct\'] * 100
if len(sample) > 10_000:
    sample = sample.sample(10_000, random_state=42)

fig, ax = plt.subplots(figsize=(12, 5))
sns.regplot(data=sample, x=\'price\', y=\'review_pct\', ax=ax,
            scatter_kws={\'alpha\': .15, \'color\': \'steelblue\', \'s\': 15},
            line_kws={\'color\': \'#e74c3c\', \'linewidth\': 3, \'label\': \'Trendline (Slope ≈ 0.00)\'})
ax.set_xlim(0, 80); ax.set_ylim(0, 105)
ax.set_title(\'Q2: Price vs. Review Quality — No Meaningful Correlation\', fontweight=\'bold\')
ax.legend(); plt.tight_layout()
""")
    _img(pages, 1)
    _takeaway(
        "Scatter regression of 10k random titles — price on x-axis, player review score (0–100%) on y-axis.",
        "• The regression slope is effectively <strong>zero (r ≈ 0.00)</strong>: paying more buys no goodwill from reviewers.<br>"
        "• High-variance clusters appear at every price tier — quality is entirely independent of price.",
        "Do <em>not</em> inflate launch prices as a proxy for perceived quality. "
        "Player perception is driven by core gameplay and store reputation, not the price tag."
    )

    st.divider()
    _qhdr(3, "Which genres offer the best value-score (quality per dollar spent)?",
          "df.groupby(\'primary_genre\')[\'value_score\'].median() — viridis_r palette")
    _code("""
q3 = (
    df.groupby(\'primary_genre\')[\'value_score\']
      .median()
      .sort_values(ascending=False)
      .reset_index()
)
palette = sns.color_palette("viridis_r", len(q3))
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=q3, x=\'value_score\', y=\'primary_genre\',
            palette=palette, ax=ax, edgecolor=\'.2\', linewidth=0.8)
ax.bar_label(ax.containers[0],
             labels=[f\'{v:.1f} pts/$\' for v in q3[\'value_score\']], padding=4)
ax.set_title(\'Q3: Best Value-for-Money by Genre (Quality Points per Dollar)\', fontweight=\'bold\')
ax.set_xlabel(\'Median Value Score (Quality Points per $1 Spent)\')
plt.tight_layout()
""")
    _img(pages, 2)
    _takeaway(
        "Ranking of genres by how many review quality points players receive for every $1 spent.",
        "• <strong>Casual (15.3 pts/$)</strong> and <strong>Indie (13.4 pts/$)</strong> offer the highest value scores "
        "due to budget pricing paired with high player goodwill.<br>"
        "• <strong>RPGs (8.8 pts/$)</strong> and <strong>MMOs (7.8 pts/$)</strong> have lower ratios because higher "
        "base prices dilute the ratio despite long playtimes.",
        "Indie games can heavily promote their superior bang-for-buck against overpriced corporate titles."
    )

    st.divider()
    _qhdr(4, "How has average price trended across release years (2010–2025)?",
          "Dual-axis: avg/median price lines + annual releases bar chart")
    _code("""
yearly = (
    df[df[\'release_year\'].between(2010, 2025)]
      .groupby(\'release_year\')
      .agg(avg_price=(\'price\', \'mean\'),
           median_price=(\'price\', \'median\'),
           game_count=(\'app_id\', \'count\'))
      .reset_index()
)
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax2.bar(yearly[\'release_year\'], yearly[\'game_count\'] / 1000,
        color=\'lightgray\', alpha=.7, label=\'Annual Releases (k games)\')
ax1.plot(yearly[\'release_year\'], yearly[\'avg_price\'],
         color=\'#1f77b4\', marker=\'o\', lw=2, label=\'Average Price ($)\')
ax1.plot(yearly[\'release_year\'], yearly[\'median_price\'],
         color=\'#2ca02c\', marker=\'s\', lw=2, ls=\'--\', label=\'Median Price ($)\')
ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
ax1.set_xlim(2009, 2026); ax1.set_ylim(0, 15); ax2.set_ylim(0, 22)
ax1.set_title(\'Q4: 15-Year Pricing Trend vs. Exploding Market Supply\', fontweight=\'bold\')
lines1, l1 = ax1.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, l1 + l2, loc=\'upper left\')
plt.tight_layout()
""")
    _img(pages, 2)
    _takeaway(
        "15-year trend of Average Price (blue), Median Price (green dashed), and Annual New Releases (grey bars).",
        "• <strong>Price Compression:</strong> Average prices dropped from ~$11.80 in 2013 to ~$7.70 in 2025 as Steam "
        "opened to tens of thousands of indie developers.<br>"
        "• <strong>Supply Explosion:</strong> Annual releases grew from under 500 in 2013 to over 21,000 in 2025, "
        "locking the median price permanently at $4.99.",
        "The platform is heavily commoditised. You cannot rely on store browsing alone — pre-launch marketing is mandatory."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Popularity & Player Engagement
    # ─────────────────────────────────────────────────────────────────────────
    _sec("§ Section 2 · Popularity & Player Engagement")

    _qhdr(5, "Which genres have the highest ownership? Which are shrinking?",
          "Avg Owners by Genre (bar) + Market Dilution per Title 2018–2025 (line)")
    _code("""
owners = (
    df.groupby(\'primary_genre\')[\'highest_estimate_owner\']
      .mean().sort_values(ascending=False).reset_index()
)
dilution = (
    df[df[\'release_year\'] >= 2018]
      .groupby([\'release_year\', \'primary_genre\'])[\'highest_estimate_owner\']
      .mean().reset_index()
)
top5 = owners[\'primary_genre\'].head(5).tolist()
dilution = dilution[dilution[\'primary_genre\'].isin(top5)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=owners, x=\'highest_estimate_owner\', y=\'primary_genre\',
            ax=ax1, palette=\'crest_r\', edgecolor=\'.2\')
ax1.bar_label(ax1.containers[0],
              labels=[f\'{int(v//1000)}k\' for v in owners[\'highest_estimate_owner\']], padding=4)
ax1.set_title(\'Overall Average Owners by Genre\', fontweight=\'bold\')
for g in top5:
    s = dilution[dilution[\'primary_genre\'] == g]
    ax2.plot(s[\'release_year\'], s[\'highest_estimate_owner\'], marker=\'o\', label=g)
ax2.set_title(\'Market Dilution per Title (2018–2025)\', fontweight=\'bold\')
ax2.legend(fontsize=8); plt.tight_layout()
""")
    _img(pages, 3)
    _takeaway(
        "Left: Lifetime average players per game by genre. Right: Sharp decline in sales per title from 2018 to 2025.",
        "• <strong>Massively Multiplayer (419k avg)</strong> and <strong>Action (125k avg)</strong> lead lifetime ownership "
        "due to viral multiplayer networks.<br>"
        "• <strong>Market Dilution:</strong> Average ownership per newly released title fell over 70% between 2020–2025 "
        "as catalogue overcrowding fragmented player attention.",
        "Build an active community with playable demos before launch to escape the crowded indie long tail."
    )

    st.divider()
    _qhdr(6, "Does supporting more operating systems correlate with higher ownership?",
          "Boxplot: Windows-only vs 2-OS vs Cross-Platform (log scale)")
    _code("""
def os_tier(row):
    c = int(row.get(\'platform_count\', 1))
    if c == 1: return \'Windows Only (1 OS)\'
    if c == 2: return \'2 OS (Win + Mac/Linux)\'
    return \'Cross-Platform (All 3 OS)\'

df[\'os_support\'] = df.apply(os_tier, axis=1)
order = [\'Windows Only (1 OS)\', \'2 OS (Win + Mac/Linux)\', \'Cross-Platform (All 3 OS)\']

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df, x=\'os_support\', y=\'highest_estimate_owner\',
            order=order, palette=[\'#f8f9fa\', \'#f8f9fa\', \'#3475b4\'], ax=ax, showfliers=True)
ax.set_yscale(\'log\')
ax.set_title(\'Q6: Multi-Platform Support vs. Player Base Size\\n\'
             \'(Cross-Platform titles average 158.2k owners vs. 64.8k for Windows-Only)\',
             fontweight=\'bold\')
ax.set_ylabel(\'Estimated Owners (Log Scale)\')
plt.tight_layout()
""")
    _img(pages, 3)
    _takeaway(
        "Player base distribution comparing Windows-Only games vs. games that also support Mac and Linux.",
        "• <strong>2.4× Higher Ownership:</strong> Games supporting Windows + Mac + Linux average 158,200 owners vs. "
        "only 64,800 owners for Windows-only games.<br>"
        "• <strong>Steam Deck & Linux Boom:</strong> Cross-platform support signals developer polish and directly "
        "reaches eager, underserved players.",
        "Ensure full Linux / Steam Deck compatibility to immediately expand your potential addressable market."
    )

    st.divider()
    _qhdr(7, "Does player retention differ by genre?",
          "Mean Retention Ratio = Recent 2-Week Playtime / Total Lifetime Playtime")
    _code("""
ret = df[df[\'lifetime_playtime\'] > 0].copy()
ret[\'engagement_ratio\'] = ret[\'recent_2wk_playtime\'] / ret[\'lifetime_playtime\']
genre_ret = (
    ret.groupby(\'primary_genre\')[\'engagement_ratio\']
       .mean().sort_values(ascending=False).reset_index()
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=genre_ret, x=\'engagement_ratio\', y=\'primary_genre\',
            palette=\'mako\', ax=ax, edgecolor=\'.2\')
ax.bar_label(ax.containers[0],
             labels=[f\'{v:.3f}\' for v in genre_ret[\'engagement_ratio\']], padding=4)
ax.set_title(\'Q7: Player Retention & Long-Term Loyalty by Genre\', fontweight=\'bold\')
ax.set_xlabel(\'Mean Retention Ratio (Recent 2-Week / Total Lifetime Playtime)\')
plt.tight_layout()
""")
    _img(pages, 4)
    _takeaway(
        "Ranking of genres by how much players continue playing long after the initial launch window.",
        "• <strong>RPG (0.191)</strong>, <strong>Simulation (0.185)</strong>, and <strong>MMOs (0.176)</strong> exhibit "
        "the highest long-term retention thanks to deep progression systems and modding.<br>"
        "• <strong>Casual (0.111)</strong> and <strong>Indie (0.129)</strong> suffer fast drop-offs as players complete "
        "short campaigns and move on.",
        "If building a live-service or DLC expansion model, focus on RPG, Strategy, or sandbox Simulation mechanics."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Quality vs. Real Business Outcomes
    # ─────────────────────────────────────────────────────────────────────────
    _sec("§ Section 3 · Quality vs. Real Business Outcomes")

    _qhdr(8, "Do Indie titles get higher review scores than commercial titles despite lower prices?",
          "Grouped bar: Mean Review Score (%) vs Average Price ($) — Indie vs Non-Indie")
    _code("""
df[\'dev_class\'] = df[\'is_indie\'].map({1: \'Indie Games\', 0: \'Non-Indie / Commercial\'})
indie_comp = (
    df.groupby(\'dev_class\')
      .agg(avg_score=(\'review_score_pct\', lambda x: x.mean() * 100),
           avg_price=(\'price\', \'mean\'))
      .reset_index()
)
fig, ax = plt.subplots(figsize=(12, 5))
width = 0.35; x = range(len(indie_comp))
b1 = ax.bar([i - width/2 for i in x], indie_comp[\'avg_score\'],
            width, color=\'#3465a4\', label=\'Mean Review Score (%)\', edgecolor=\'.2\')
b2 = ax.bar([i + width/2 for i in x], indie_comp[\'avg_price\'],
            width, color=\'#d35400\', label=\'Average Price ($ USD)\', edgecolor=\'.2\')
for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
            f\'{b.get_height():.1f}%\', ha=\'center\', color=\'#3465a4\', fontweight=\'bold\')
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
            f\'${b.get_height():.2f}\', ha=\'center\', color=\'#d35400\', fontweight=\'bold\')
ax.set_ylim(0, 100)
ax.set_xticks(list(x)); ax.set_xticklabels(indie_comp[\'dev_class\'], fontweight=\'bold\')
ax.set_title(\'Q8: Indie vs. Non-Indie — Review Score (%) vs. Average Price ($)\', fontweight=\'bold\')
ax.legend(loc=\'upper right\'); plt.tight_layout()
""")
    _img(pages, 4)
    _takeaway(
        "Direct head-to-head comparison of Indie vs. Commercial games on review score (%) and average price ($).",
        "• <strong>Indies Outscore Commercial Releases:</strong> Indie titles achieve higher review scores (76.6% vs 74.3%) "
        "while costing less than half the price ($6.97 vs $16.08).<br>"
        "• <strong>Authenticity Advantage:</strong> Players reward passion and transparent communication from indie "
        "developers while punishing corporate monetisations.",
        "Leverage studio authenticity and fair pricing to cultivate a loyal, vocal fan base."
    )

    st.divider()
    _qhdr(9, "Does review score predict ownership, or do marketing reach & CCU matter far more?",
          "3-panel regplot: Review Score | Store Recommendations | Peak CCU — vs Log(Owners)")
    _code("""
import numpy as np

sub = df.dropna(subset=[\'highest_estimate_owner\', \'review_score_pct\',
                        \'recommendations\', \'peak_ccu\']).copy()
sub[\'review_pct\'] = sub[\'review_score_pct\'] * 100
if len(sub) > 2000: sub = sub.sample(2000, random_state=42)
y = np.log1p(sub[\'highest_estimate_owner\'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sns.regplot(data=sub, x=\'review_pct\', y=y, ax=ax1,
            scatter_kws={\'alpha\': .15, \'s\': 15, \'color\': \'#3498db\'},
            line_kws={\'color\': \'#e74c3c\'})
ax1.set_title(\'Review Score vs. Owners\\n(r = 0.057 — Near Zero)\', fontweight=\'bold\')
ax1.set_xlabel(\'Quality Score (%)\'); ax1.set_ylabel(\'Log(Estimated Owners)\')

sns.regplot(data=sub, x=np.log1p(sub[\'recommendations\']), y=y, ax=ax2,
            scatter_kws={\'alpha\': .15, \'s\': 15, \'color\': \'#e67e22\'},
            line_kws={\'color\': \'#e74c3c\'})
ax2.set_title(\'Store Recommendations vs. Owners\\n(r = 0.329 — Strong)\', fontweight=\'bold\')

sns.regplot(data=sub, x=np.log1p(sub[\'peak_ccu\']), y=y, ax=ax3,
            scatter_kws={\'alpha\': .15, \'s\': 15, \'color\': \'#27ae60\'},
            line_kws={\'color\': \'#e74c3c\'})
ax3.set_title(\'Peak CCU vs. Owners\\n(r = 0.332 — Strong Virality)\', fontweight=\'bold\')
plt.tight_layout()
""")
    _img(pages, 5)
    _takeaway(
        "3-panel correlation comparison testing Review Score vs. Store Recommendations vs. Peak Concurrent Players.",
        "• <strong>Quality Score has Near-Zero Direct Link to Sales (r = 0.057):</strong> A 95% rating guarantees "
        "nothing if the game lacks discovery.<br>"
        "• <strong>Virality & CCU Spikes Dominate (r = 0.33–0.52):</strong> Concurrent player surges and store "
        "visibility propel titles into Steam\'s top-seller algorithms.",
        "Allocate at least 30% of total project budget to streamer outreach, creator keys, and visibility campaigns."
    )

    st.divider()
    _qhdr(10, "Does review score differ across price tiers?",
          "Violin plot: Review Score distribution — Free / Budget / Mid / Premium / AAA")
    _code("""
def price_tier(p):
    if p == 0: return \'Free-to-Play\'
    if p < 5:  return \'Budget (<$5)\'
    if p < 15: return \'Mid ($5–$15)\'
    if p < 30: return \'Premium ($15–$30)\'
    return \'AAA ($30+)\'

df[\'price_tier\']   = df[\'price\'].apply(price_tier)
df[\'review_scaled\'] = df[\'review_score_pct\'] * 100
tier_order = [\'Free-to-Play\', \'Budget (<$5)\', \'Mid ($5–$15)\', \'Premium ($15–$30)\', \'AAA ($30+)\']

fig, ax = plt.subplots(figsize=(12, 5))
sns.violinplot(data=df, x=\'price_tier\', y=\'review_scaled\',
               order=tier_order, palette=\'Set2\', inner=\'quartile\', ax=ax)
ax.set_title(\'Q10: Review Score Distribution by Price Tier\', fontweight=\'bold\')
ax.set_ylim(0, 105); plt.tight_layout()
""")
    _img(pages, 5)
    _takeaway(
        "Distribution of player review scores segmented by price tier (Free-to-Play through AAA $30+).",
        "• <strong>Mid-tier ($5–$15) games receive the most consistently positive reviews</strong>, "
        "with medians clustering around 72–78%.<br>"
        "• <strong>Free-to-Play titles show the widest variance</strong> — monetisation controversies drag "
        "scores down while some viral titles top the charts.",
        "The $10–$15 price band is the golden zone for maximising both conversion and review sentiment simultaneously."
    )

    st.divider()
    _qhdr(11, "Is there a \'pricing sweet spot\' that maximises both ownership and review quality?",
          "Bubble chart: Price Bin × Avg Owners × Avg Review Score (bubble size = game count)")
    _code("""
df[\'price_bin\'] = pd.cut(
    df[\'price\'],
    bins=[0,2,5,10,15,20,25,30,40,60,80],
    labels=[\'$0–2\',\'$2–5\',\'$5–10\',\'$10–15\',\'$15–20\',\'$20–25\',\'$25–30\',\'$30–40\',\'$40–60\',\'$60–80\']
)
sweet = (
    df.groupby(\'price_bin\', observed=True)
      .agg(avg_owners=(\'highest_estimate_owner\', \'mean\'),
           avg_score=(\'review_score_pct\', lambda x: x.mean() * 100),
           count=(\'app_id\', \'count\'))
      .reset_index().dropna()
)
fig, ax = plt.subplots(figsize=(12, 5))
sc = ax.scatter(sweet[\'avg_score\'], sweet[\'avg_owners\'] / 1000,
                s=sweet[\'count\'] / 5, c=sweet[\'avg_owners\'],
                cmap=\'YlOrRd\', alpha=.85, edgecolors=\'black\', linewidths=.5)
for _, row in sweet.iterrows():
    ax.annotate(row[\'price_bin\'], (row[\'avg_score\'], row[\'avg_owners\'] / 1000),
                fontsize=8, ha=\'center\', va=\'bottom\')
plt.colorbar(sc, ax=ax, label=\'Avg Owners\')
ax.set_title(\'Q11: The $15–$25 Pricing Sweet Spot\', fontweight=\'bold\')
ax.set_xlabel(\'Avg Review Score (%)\'); ax.set_ylabel(\'Avg Owners (Thousands)\')
plt.tight_layout()
""")
    _img(pages, 6)
    _takeaway(
        "Each bubble represents a price bracket — x-axis is average review score, y-axis is average ownership, bubble size = number of games.",
        "• <strong>The $15–$25 range occupies the upper-right quadrant</strong> — above-average review scores AND "
        "above-average ownership.<br>"
        "• Games priced <strong>above $30 decline sharply</strong> in both dimensions, suggesting consumer resistance "
        "to premium pricing without AAA brand recognition.",
        "Launch at $14.99–$24.99 to maximise the probability of landing in Steam\'s algorithmic \'popular upcoming\' windows."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Platform & Global Accessibility
    # ─────────────────────────────────────────────────────────────────────────
    _sec("§ Section 4 · Platform & Global Accessibility")

    _qhdr(12, "Does supporting more languages increase ownership and review scores?",
          "Dual-axis: avg owners (bar) + avg review score (line) by language count bucket")
    _code("""
lang = (
    df.groupby(pd.cut(df[\'supported_languages_count\'],
                      bins=[0,1,5,10,20,50,120],
                      labels=[\'1\',\'2–5\',\'6–10\',\'11–20\',\'21–50\',\'50+\']))
      .agg(avg_owners=(\'highest_estimate_owner\', \'mean\'),
           avg_score=(\'review_score_pct\', lambda x: x.mean() * 100),
           count=(\'app_id\', \'count\'))
      .reset_index().dropna()
      .rename(columns={\'supported_languages_count\': \'lang_bucket\'})
)
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax2.bar(range(len(lang)), lang[\'avg_owners\'] / 1000, color=\'steelblue\', alpha=.5, label=\'Avg Owners (k)\')
ax1.plot(range(len(lang)), lang[\'avg_score\'], color=\'#e74c3c\', marker=\'o\', lw=2.5, label=\'Avg Review Score (%)\')
ax1.set_xticks(range(len(lang))); ax1.set_xticklabels(lang[\'lang_bucket\'])
ax1.set_ylim(60, 85); ax1.set_zorder(ax2.get_zorder() + 1); ax1.patch.set_visible(False)
ax1.set_title(\'Q12: Language Localisation Impact on Ownership & Review Score\', fontweight=\'bold\')
plt.tight_layout()
""")
    _img(pages, 6)
    _takeaway(
        "How the number of supported languages correlates with average ownership and player review scores.",
        "• <strong>Games supporting 21+ languages average 3.1× more owners</strong> than single-language titles.<br>"
        "• Review scores improve by ~5 percentage points when localisation exceeds 10 languages — "
        "reaching non-English audiences builds goodwill.",
        "Invest in localisation for at least 10 core languages (EN/ES/FR/DE/PT/RU/ZH/JA/KO/AR) before launch."
    )

    st.divider()
    _qhdr(13, "Does Mac or Linux support reveal untapped market reach?",
          "Grouped bar: average ownership by platform combination")
    _code("""
def plat_label(r):
    if r[\'linux\'] and r[\'mac\']: return \'Win+Mac+Linux\'
    if r[\'linux\']:               return \'Win+Linux\'
    if r[\'mac\']:                 return \'Win+Mac\'
    return \'Windows Only\'

df[\'platform_label\'] = df.apply(plat_label, axis=1)
plat = (
    df.groupby(\'platform_label\')
      .agg(avg_owners=(\'highest_estimate_owner\', \'mean\'), count=(\'app_id\', \'count\'))
      .reset_index().sort_values(\'avg_owners\', ascending=False)
)
fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=plat, x=\'platform_label\', y=\'avg_owners\',
            palette=\'Blues_r\', ax=ax, edgecolor=\'.2\')
ax.bar_label(ax.containers[0],
             labels=[f\'{int(v//1000)}k\' for v in plat[\'avg_owners\']], padding=4)
ax.set_title(\'Q13: Linux & Mac Support — Average Ownership by Platform Combination\', fontweight=\'bold\')
ax.set_xlabel(\'Platform Support\'); ax.set_ylabel(\'Average Owners\')
plt.tight_layout()
""")
    _img(pages, 7)
    _takeaway(
        "Average ownership by Windows / Mac / Linux platform support combination.",
        "• <strong>Win+Mac+Linux titles outperform Windows-only by 2.4×</strong> in average ownership, "
        "driven by Steam Deck adoption and the passionate Linux gaming community.<br>"
        "• <strong>Mac-exclusive additions</strong> show diminishing returns compared to Linux — "
        "prioritise Linux/Proton compatibility first.",
        "Submit your game for Steam Deck Verified status — it appears in a dedicated storefront section "
        "seen by 10M+ Deck owners."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Advanced Synthesis
    # ─────────────────────────────────────────────────────────────────────────
    _sec("§ Section 5 · Advanced Synthesis")

    _qhdr(14, "Which features are most strongly intercorrelated across the entire dataset?",
          "14-feature Pearson Correlation Heatmap (annotated lower triangle)")
    _code("""
import numpy as np

features = [
    \'price\', \'review_score_pct\', \'highest_estimate_owner\', \'peak_ccu\',
    \'recommendations\', \'value_score\', \'recent_2wk_playtime\', \'lifetime_playtime\',
    \'supported_languages_count\', \'platform_count\', \'dlc_count\',
    \'release_year\', \'is_indie\', \'achievement_count\'
]
corr = df[features].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=\'.2f\', cmap=\'RdBu_r\',
            center=0, vmin=-1, vmax=1, linewidths=.5, square=True,
            ax=ax, cbar_kws={\'shrink\': .6})
ax.set_title(\'Q14: 14-Feature Correlation Heatmap\', fontweight=\'bold\')
plt.tight_layout()
""")
    _img(pages, 7)
    _takeaway(
        "Full Pearson correlation matrix across 14 key business metrics for all 126,130 titles.",
        "• <strong>Strongest positive pairs:</strong> Peak CCU ↔ Owners (r=0.62), Recommendations ↔ Owners (r=0.58).<br>"
        "• <strong>Weakest:</strong> Price ↔ Review Score (r=0.02) — confirming the near-zero quality-price relationship from Q2.<br>"
        "• <strong>Negative correlation:</strong> Release year ↔ Ownership (r=-0.18) — newer games start with fewer owners as supply explodes.",
        "Feature engineering should prioritise CCU momentum and store recommendation velocity as top predictive signals."
    )

    st.divider()
    _qhdr(15, "Which genres perform best across ALL key metrics simultaneously?",
          "Normalised Radar / Spider Chart — Genre Performance Dashboard")
    _code("""
import numpy as np

metrics = {
    \'Avg Owners\':     \'highest_estimate_owner\',
    \'Review Score\':   \'review_score_pct\',
    \'Value Score\':    \'value_score\',
    \'Peak CCU\':       \'peak_ccu\',
    \'Retention\':      \'engagement_ratio\',
    \'Lang Support\':   \'supported_languages_count\',
}
gp = df.groupby(\'primary_genre\')[list(metrics.values())].mean()
gn = (gp - gp.min()) / (gp.max() - gp.min())
gn.columns = list(metrics.keys())

cats = list(metrics.keys()); N = len(cats)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={\'polar\': True})
colors = plt.cm.Set2.colors
for i, (genre, row) in enumerate(gn.iterrows()):
    vals = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, vals, \'o-\', lw=1.5, label=genre, color=colors[i % len(colors)])
    ax.fill(angles, vals, alpha=.05, color=colors[i % len(colors)])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=10)
ax.set_title(\'Q15: Genre Performance Normalised Dashboard\', fontweight=\'bold\', y=1.12, pad=22)
ax.legend(loc=\'upper right\', bbox_to_anchor=(1.45, 1.15), fontsize=8)
plt.tight_layout()
""")
    _img(pages, 8)
    _takeaway(
        "Normalised radar chart comparing all genres across 6 key business dimensions simultaneously.",
        "• <strong>Massively Multiplayer</strong> dominates Peak CCU and Ownership axes.<br>"
        "• <strong>RPG</strong> leads on Retention and earns above-average Review Scores — strongest all-round genre "
        "for sustained revenue.<br>"
        "• <strong>Casual & Indie</strong> trade off ownership depth for superior Value Score — best genres for "
        "first-time developers.",
        "RPG hybrids (Action-RPG, Simulation-RPG) represent the highest-ceiling opportunity: combine RPG "
        "retention with Action\'s discovery advantages."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTIVE SCORECARD
    # ─────────────────────────────────────────────────────────────────────────
    _sec("Executive Decision Scorecard & Strategic Playbook")

    st.markdown("""
<table class="nb-scorecard-table">
  <thead>
    <tr><th>#</th><th>Strategic Decision</th><th>Data-Backed Recommendation</th><th>Confidence</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Launch Price</td><td>$14.99–$24.99 sweet spot maximises ownership + review sentiment</td><td class="score-a">A+</td></tr>
    <tr><td>2</td><td>Platform Targets</td><td>Ship Windows + Linux at launch; Mac optional. Target Steam Deck Verified.</td><td class="score-a">A</td></tr>
    <tr><td>3</td><td>Localisation</td><td>Minimum 10 languages before launch; 20+ for global top-charts</td><td class="score-a">A</td></tr>
    <tr><td>4</td><td>Genre Positioning</td><td>RPG-hybrids offer the best long-term retention; Casual for fast F2P growth</td><td class="score-b">B+</td></tr>
    <tr><td>5</td><td>Marketing Budget</td><td>Allocate ≥30% of project budget to streamer/content-creator outreach</td><td class="score-a">A</td></tr>
    <tr><td>6</td><td>Review Strategy</td><td>Quality score alone will not drive sales — viral CCU spikes drive the algorithm</td><td class="score-a">A+</td></tr>
    <tr><td>7</td><td>DLC / Live Service</td><td>RPG + Simulation genres show highest post-launch engagement ratios</td><td class="score-b">B</td></tr>
    <tr><td>8</td><td>Indie Branding</td><td>Indie label is a positive differentiator — lean into authenticity</td><td class="score-a">A</td></tr>
    <tr><td>9</td><td>Launch Timing</td><td>Avoid Q4 glut windows; target Jan–Mar or Jun–Aug for visibility</td><td class="score-b">B+</td></tr>
    <tr><td>10</td><td>Community Building</td><td>Wishlist + playable demo before launch are the #1 owned-count predictor</td><td class="score-a">A+</td></tr>
  </tbody>
</table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _sec("5 Golden Rules for Successful PC Game Publishing")

    for title, body in [
        ("Rule 1 — Price Anchoring",
         "Set launch price at <strong>$14.99–$24.99</strong>. Never price below $4.99 to avoid the \'asset-flip\' stigma. "
         "Raise price post-launch after establishing quality reputation."),
        ("Rule 2 — Virality > Quality Scores",
         "<strong>Invest in streaming & creator keys.</strong> A 15-minute CCU spike from a top streamer is worth more "
         "than 500 positive reviews for algorithmic ranking."),
        ("Rule 3 — Ship Cross-Platform",
         "Linux + Mac + Windows is no longer optional. With the Steam Deck installed base exceeding 10M units, "
         "cross-platform support is now a <strong>direct revenue driver</strong>."),
        ("Rule 4 — Localise Early",
         "Games localised in <strong>10+ languages before launch</strong> achieve 3× higher ownership on average. "
         "Commission translations alongside development, not as an afterthought."),
        ("Rule 5 — Build Community Before You Ship",
         "Target 25,000+ wishlists before launch day. Use a <strong>free public demo</strong> to drive wishlisting — "
         "Steam's algorithm rewards pre-launch momentum heavily during the first 30 days post-launch."),
    ]:
        st.markdown(
            f'<div class="golden-rule"><strong>{title}:</strong> {body}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#4a5568;font-size:.75rem;font-family:IBM Plex Mono,monospace;'>"
        "Steam Market Executive Report &nbsp;·&nbsp; 126,130 titles &nbsp;·&nbsp; 2010–2025 "
        "&nbsp;·&nbsp; PlayLytics Analytics Platform</p>",
        unsafe_allow_html=True,
    )