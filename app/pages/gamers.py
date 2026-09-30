import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from src.similarity import find_similar_games
from src.config import ACCENT_COLORS

def _render_comparison_radar(target: dict, comps: list, df: pd.DataFrame):
    fig = go.Figure()
    radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                  "average_playtime_forever", "languages_count", "patforms_count"]
    
    genre = target.get("primary_genre")
    genre_df = df[df["primary_genre"] == genre] if genre else df
    if len(genre_df) < 10:
        genre_df = df
        
    baselines = {}
    for col in radar_cols:
        max_val = genre_df[col].max()
        if pd.isna(max_val) or max_val <= 0: max_val = df[col].max()
        if pd.isna(max_val) or max_val <= 0: max_val = 1.0
        baselines[col] = max_val

    # Render comps
    for i, comp in enumerate(comps):
        normalized_vals = []
        for col in radar_cols:
            val = float(comp.get(col, 0))
            if val <= 0:
                norm = 0
            else:
                norm = (np.log1p(val) / np.log1p(baselines[col])) * 100
                norm = min(norm, 100)
            normalized_vals.append(norm)
            
        normalized_vals.append(normalized_vals[0])
        labels = ["Price", "Review Score", "Total Reviews", "Peak CCU", "Playtime", "Languages", "Platforms", "Price"]
        color = ACCENT_COLORS[(i + 1) % len(ACCENT_COLORS)]
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_vals,
            theta=labels,
            fill='none',
            name=comp.get("name", f"Comp {i+1}"),
            line=dict(color=color, dash='dot')
        ))
        
    # Render target (more prominent)
    target_norm = []
    for col in radar_cols:
        val = float(target.get(col, 0))
        if val <= 0:
            norm = 0
        else:
            norm = (np.log1p(val) / np.log1p(baselines[col])) * 100
            norm = min(norm, 100)
        target_norm.append(norm)
        
    target_norm.append(target_norm[0])
    
    fig.add_trace(go.Scatterpolar(
        r=target_norm,
        theta=labels,
        fill='toself',
        name=target.get("name", "Target"),
        line=dict(color=ACCENT_COLORS[0], width=3),
        fillcolor=ACCENT_COLORS[0].replace(')', ', 0.3)').replace('rgb', 'rgba') if 'rgb' in ACCENT_COLORS[0] else ACCENT_COLORS[0] + '4D'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#555e6e")),
            angularaxis=dict(tickfont=dict(color="#9da3ae"))
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#9da3ae"),
        margin=dict(t=40, b=40, l=40, r=40),
        height=450
    )
    return fig

def _render_genre_landscape(target: dict, df: pd.DataFrame):
    genre = target.get("primary_genre")
    if not genre or pd.isna(genre):
        return None
    
    # Filter to same genre, cap to 500 games to avoid browser lag
    genre_df = df[(df["primary_genre"] == genre) & (df["price"] <= 100) & (df["review_score_pct"] > 0)].copy()
    if len(genre_df) > 500:
        genre_df = genre_df.sample(500, random_state=42)
        
    fig = go.Figure()
    
    # Background games
    fig.add_trace(go.Scatter(
        x=genre_df["price"],
        y=genre_df["review_score_pct"] * 100,
        mode="markers",
        marker=dict(
            size=genre_df["total_review"].fillna(0) / 5000 + 5,
            sizemode='area',
            sizemin=4,
            color="rgba(157, 163, 174, 0.4)",
            line=dict(width=0)
        ),
        text=genre_df["name"],
        name=f"Other {genre} Games",
        hoverinfo="text+x+y"
    ))
    
    # Target game
    target_price = target.get("price", 0)
    target_score = target.get("review_score_pct", 0) * 100
    
    fig.add_trace(go.Scatter(
        x=[target_price],
        y=[target_score],
        mode="markers+text",
        marker=dict(
            size=18,
            symbol="diamond",
            color=ACCENT_COLORS[0],
            line=dict(color="white", width=2)
        ),
        text=[target.get("name")],
        textposition="top center",
        name="Searched Game",
        hoverinfo="text+x+y",
        cliponaxis=False
    ))
    
    fig.update_layout(
        title=f"Price vs Quality in {genre}",
        xaxis_title="Price ($)",
        yaxis_title="Review Score (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#9da3ae"),
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def _render_playtime_dist(target: dict, df: pd.DataFrame):
    genre = target.get("primary_genre")
    if not genre or pd.isna(genre):
        return None
        
    genre_df = df[(df["primary_genre"] == genre) & (df["average_playtime_forever"] > 0)].copy()
    genre_df["playtime_hrs"] = genre_df["average_playtime_forever"] / 60
    
    # Remove extreme outliers for the plot
    if len(genre_df) > 10:
        cap = genre_df["playtime_hrs"].quantile(0.95)
        genre_df = genre_df[genre_df["playtime_hrs"] <= cap]
        
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        x=genre_df["playtime_hrs"],
        name=genre,
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(157, 163, 174, 0.2)",
        line_color="rgba(157, 163, 174, 0.7)",
        showlegend=False,
    ))
    
    target_playtime = target.get("average_playtime_forever", 0) / 60
    
    fig.add_trace(go.Scatter(
        x=[target_playtime],
        y=[genre], # Aligns with the violin plot's category
        mode="markers+text",
        marker=dict(
            size=16,
            symbol="diamond",
            color=ACCENT_COLORS[0],
            line=dict(color="white", width=2)
        ),
        text=[target.get("name")],
        textposition="top center",
        name="Searched Game",
        hoverinfo="text+x",
        cliponaxis=False
    ))
    
    fig.update_layout(
        title=f"Avg Playtime in {genre}",
        xaxis_title="Playtime (Hours)",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#9da3ae"),
        margin=dict(t=40, b=20, l=20, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, max(genre_df["playtime_hrs"].max(), target_playtime) * 1.25]),
        yaxis=dict(showticklabels=False, gridcolor="rgba(255,255,255,0.05)")
    )
    return fig

def render(df):
    st.markdown("<div class='hero-header'><div class='hero-title'>Gamer's Hub</div><div class='hero-subtitle'>Discover games, check their value, and find cheaper alternatives.</div></div>", unsafe_allow_html=True)
    
    popular_games = df.sort_values("total_review", ascending=False)["name"].dropna().head(15000).tolist()
    target_name = st.selectbox(
        "Search for a Game", 
        options=popular_games, 
        index=None, 
        placeholder="e.g. Cyberpunk 2077, Stardew Valley…"
    )
    
    if not target_name:
        return
        
    target_row = df[df["name"] == target_name]
    if target_row.empty:
        st.warning("Game not found in dataset.")
        return
        
    target_dict = target_row.iloc[0].to_dict()
    
    st.markdown(f"### {target_name} Summary")
    
    # 1. Enhanced Summary & Playtime Stats
    col1, col2, col3, col4, col5 = st.columns(5)
    price = target_dict.get('price', 0)
    review_score = target_dict.get('review_score_pct', 0) * 100
    peak_ccu = target_dict.get('peak_ccu', 0)
    avg_playtime = target_dict.get('average_playtime_forever', 0) / 60
    median_playtime = target_dict.get('median_playtime_hours', 0)
    
    col1.metric("Price", f"${price:.2f}" if price > 0 else "Free to Play")
    col2.metric("Review Score", f"{review_score:.1f}%")
    col3.metric("Peak Players", f"{peak_ccu:,.0f}")
    col4.metric("Avg Playtime", f"{avg_playtime:.1f} hrs")
    col5.metric("Median Playtime", f"{median_playtime:.1f} hrs")
    
    st.markdown("---")
    
    # 2. & 3. Badges and Multiplayer Health
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'>Features & Accessibility</div>", unsafe_allow_html=True)
        badges = []
        if target_dict.get('windows') == 1: badges.append("🖥️ Windows")
        if target_dict.get('mac') == 1: badges.append("🍏 Mac")
        if target_dict.get('linux') == 1: badges.append("🐧 Linux")
        if target_dict.get('cat_full_controller_support') == 1: badges.append("🎮 Controller Support")
        
        audio_langs = target_dict.get('full_audio_languages_count', 0)
        if audio_langs > 1: badges.append(f"🗣️ {audio_langs} Audio Languages")
        
        if badges:
            for badge in badges:
                st.markdown(f"✅ {badge}")
        else:
            st.markdown("No extra accessibility features found.")
            
    with col_b:
        is_mp = target_dict.get('cat_multi_player', 0) == 1 or target_dict.get('cat_co_op', 0) == 1
        if is_mp:
            st.markdown("<div class='section-header'>Multiplayer Health</div>", unsafe_allow_html=True)
            if peak_ccu > 10000:
                health, color = "Thriving Community", "var(--success)"
            elif peak_ccu > 1000:
                health, color = "Active Playerbase", "var(--accent)"
            elif peak_ccu > 100:
                health, color = "Cult Following", "var(--warning)"
            else:
                health, color = "Dead Multiplayer", "var(--danger)"
                
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 6px; border: 1px solid {color}; text-align: center; background: rgba(0,0,0,0.15);'>
                <strong style='color: {color}; font-size: 1.1rem;'>{health}</strong>
                <p style='margin:0; font-size: 0.8rem; color: var(--text-muted);'>Peak CCU: {peak_ccu:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='section-header'>Multiplayer Health</div>", unsafe_allow_html=True)
            st.markdown("<p style='color: var(--text-muted);'>Single-player game.</p>", unsafe_allow_html=True)

    # 4. True Value Engine
    st.markdown("<div class='section-header'>Value Assessment</div>", unsafe_allow_html=True)
    
    price_tier = target_dict.get('price_tier', 'Unknown')
    
    # Calculate a custom value score taking price tier into account
    if price == 0:
        if review_score >= 80:
            val_status, val_color = "Excellent Deal (Free)", "var(--success)"
            val_desc = "It's free and highly rated! You can't beat that price-to-quality ratio."
        elif review_score >= 60:
            val_status, val_color = "Fair Free Game", "var(--accent)"
            val_desc = "It's free and offers a decent experience despite mixed reviews."
        else:
            val_status, val_color = "Poor Quality (Free)", "var(--danger)"
            val_desc = "Even at no cost, the very low review score suggests it may not be worth your time."
    else:
        # Cost-per-hour heuristic
        playtime = median_playtime if median_playtime > 0 else avg_playtime
        hour_type = "median hour" if median_playtime > 0 else "average hour"
        
        if playtime > 0:
            cost_per_hour = price / playtime
            
            if cost_per_hour > 7.0:
                val_status, val_color = "Pricey for Playtime", "var(--danger)"
                val_desc = f"At approx <strong>${cost_per_hour:.2f}/{hour_type}</strong>, this is quite expensive for the amount of time players typically spend. Consider waiting for a sale."
            elif cost_per_hour > 3.0:
                if review_score >= 80:
                    val_status, val_color = "Fair Premium Price", "var(--accent)"
                    val_desc = f"At approx <strong>${cost_per_hour:.2f}/{hour_type}</strong>, it's a bit pricey but community reviews suggest it's a high-quality experience."
                else:
                    val_status, val_color = "Wait for Sale", "var(--warning)"
                    val_desc = f"At approx <strong>${cost_per_hour:.2f}/{hour_type}</strong> with mixed reviews, it might be better to pick this up at a discount."
            else:
                if review_score >= 85:
                    val_status, val_color = "Excellent Deal", "var(--success)"
                    val_desc = f"At approx <strong>${cost_per_hour:.2f}/{hour_type}</strong> with very positive reviews, this is a steal!"
                elif review_score >= 60:
                    val_status, val_color = "Good Value", "var(--accent)"
                    val_desc = f"At approx <strong>${cost_per_hour:.2f}/{hour_type}</strong>, you get solid playtime for your money."
                else:
                    val_status, val_color = "Poor Quality", "var(--danger)"
                    val_desc = "It's cheap per hour, but the low review score suggests the time spent might not be enjoyable."
        else:
            if price > 30:
                val_status, val_color = "Wait for Sale", "var(--warning)"
                val_desc = "No playtime data available. Given the high price tag, proceed with caution or wait for a sale."
            else:
                val_status, val_color = "Fair Price", "var(--accent)"
                val_desc = "No playtime data available, but the price is relatively low."
        
    st.markdown(f"""
    <div style='padding: 16px; border-radius: 8px; border: 1px solid {val_color}; background: rgba(0,0,0,0.2);'>
        <h4 style='color: {val_color}; margin: 0 0 8px 0;'>{val_status}</h4>
        <p style='margin: 0; font-size: 0.9rem;'>{val_desc}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4.5 Market Context & Genre Landscape
    st.markdown("<div class='section-header'>Market Context & Genre Landscape</div>", unsafe_allow_html=True)
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        fig_landscape = _render_genre_landscape(target_dict, df)
        if fig_landscape:
            st.plotly_chart(fig_landscape, use_container_width=True)
    with plot_col2:
        fig_playtime = _render_playtime_dist(target_dict, df)
        if fig_playtime:
            st.plotly_chart(fig_playtime, use_container_width=True)

    # 5. & 6. Vibe Matcher & Hidden Gems
    st.markdown("<div class='section-header'>Discovery & Cheaper Alternatives</div>", unsafe_allow_html=True)
    
    # Filters
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        st.markdown("<br/>", unsafe_allow_html=True)
        hidden_gems = st.checkbox("💎 Show Hidden Gems Only", help="Find highly-rated games with smaller player bases.")
    with col_f2:
        tag_options = {
            "Action": "genre_action", "RPG": "genre_rpg", "Strategy": "genre_strategy", 
            "Indie": "is_indie", "Multiplayer": "cat_multi_player", "Co-op": "cat_co_op",
            "Adventure": "genre_adventure", "Simulation": "genre_simulation", "Casual": "genre_casual"
        }
        selected_tags = st.multiselect("Must include tags:", options=list(tag_options.keys()))
        
    with st.spinner("Finding similar but cheaper games..."):
        sim_df = find_similar_games(target_dict, df, n=1500) # get a large pool to filter from
        
        # Base filter: Cheaper and good reviews
        if price > 0:
            filtered_df = sim_df[(sim_df["price"] < price) & (sim_df["price"] > 0) & (sim_df["review_score_pct"] > 0.70)]
        else:
            filtered_df = sim_df[(sim_df["price"] == 0) & (sim_df["review_score_pct"] > 0.70)]
            
        # Hidden Gems filter
        if hidden_gems:
            filtered_df = filtered_df[(filtered_df["review_score_pct"] >= 0.85) & (filtered_df["total_review"] < 15000)]
            
        # Vibe Matcher (Tags) filter
        for tag in selected_tags:
            col_name = tag_options[tag]
            filtered_df = filtered_df[filtered_df[col_name] == 1]
            
        final_df = filtered_df.head(5)
        
    if final_df.empty:
        st.info("No alternatives found matching these specific filters. Try removing some tags!")
    else:
        display_df = final_df[["name", "primary_genre", "price", "review_score_pct", "similarity_score"]].copy()
        display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}" if x > 0 else "Free")
        display_df["review_score_pct"] = display_df["review_score_pct"].apply(lambda x: f"{x*100:.1f}%")
        display_df["similarity_score"] = display_df["similarity_score"].apply(lambda x: f"{x*100:.1f}%")
        display_df = display_df.rename(columns={
            "name": "Game",
            "primary_genre": "Genre",
            "price": "Price",
            "review_score_pct": "Review Score",
            "similarity_score": "Similarity Match"
        })

        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("<div class='section-header'>Feature Comparison vs Alternatives</div>", unsafe_allow_html=True)
        radar_cols = ["price", "review_score_pct", "total_review", "peak_ccu", 
                      "average_playtime_forever", "languages_count", "patforms_count"]
        
        comps = final_df.head(3).to_dict(orient="records")
        fig = _render_comparison_radar(target_dict, comps, df)
        st.plotly_chart(fig, use_container_width=True)
