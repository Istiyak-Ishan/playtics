"""
similarity.py — Similar Game Engine for the Steam Market Intelligence Platform.

Uses cosine similarity on a standardised feature matrix.
Returns the N most similar games with scores and comparison attributes.

Excluded from features: app_id, name, URLs, image URLs, identifiers.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SIMILARITY_FEATURES, SIMILARITY_N_DEFAULT

# ── Module-level cache ─────────────────────────────────────────────────────────
_feature_matrix: np.ndarray | None = None
_scaler_sim: StandardScaler | None = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_indexed_df: pd.DataFrame | None = None


DISPLAY_COLS = [
    "name", "price", "review_score_pct", "owners_mid",
    "recommendations", "peak_ccu", "average_playtime_forever",
    "languages_count", "patforms_count", "age_by_years",
    "primary_genre", "price_tier",
]

# Weight multiplier for TF-IDF tags so gameplay outweighs commercial stats
TFIDF_WEIGHT = 3.0


def _parse_tags(val) -> str:
    """Parse tags/genres list or string into space-separated underscore-joined words."""
    if pd.isna(val) or not str(val).strip():
        return ""
    val_str = str(val).strip()
    
    if val_str.startswith("["):
        try:
            items = json.loads(val_str.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            items = [x.strip().strip("'").strip('"') for x in val_str.strip("[]").split(',')]
    else:
        items = [x.strip() for x in val_str.split(',')]
        
    return " ".join([x.lower().replace(" ", "_").replace("-", "_") for x in items if x])


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, TfidfVectorizer, pd.DataFrame]:
    """
    Build a standardised feature matrix from the dataset.
    Combines scaled numerical features with TF-IDF weighted text features (tags + genres).
    """
    available = [f for f in SIMILARITY_FEATURES if f in df.columns]
    
    keep_cols = available + ["app_id"]
    if "tags" in df.columns: keep_cols.append("tags")
    if "genres" in df.columns: keep_cols.append("genres")
        
    clean = df[list(set(keep_cols))].dropna(subset=available).copy()
    
    # 1. Numerical & Binary Features
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(clean[available])
    
    # 2. TF-IDF Text Features (tags and genres)
    tfidf = TfidfVectorizer(max_features=200)
    
    text_series = pd.Series([""] * len(clean), index=clean.index)
    if "tags" in clean.columns:
        text_series += clean["tags"].apply(_parse_tags) + " "
    if "genres" in clean.columns:
        text_series += clean["genres"].apply(_parse_tags)
        
    tfidf_matrix = tfidf.fit_transform(text_series).toarray()
    
    # Weight the TF-IDF matrix heavily
    tfidf_matrix = tfidf_matrix * TFIDF_WEIGHT
    
    # Combine
    final_matrix = np.hstack([num_matrix, tfidf_matrix])
    
    return final_matrix, scaler, tfidf, clean.reset_index(drop=True)


def _ensure_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, TfidfVectorizer, pd.DataFrame]:
    """Build or return cached feature matrix."""
    global _feature_matrix, _scaler_sim, _tfidf_vectorizer, _indexed_df
    if _feature_matrix is None or _indexed_df is None:
        _feature_matrix, _scaler_sim, _tfidf_vectorizer, _indexed_df = _build_feature_matrix(df)
    return _feature_matrix, _scaler_sim, _tfidf_vectorizer, _indexed_df


def find_similar_games(
    game_profile: dict | pd.Series,
    df: pd.DataFrame,
    n: int = SIMILARITY_N_DEFAULT,
    exclude_app_id: int | None = None,
) -> pd.DataFrame:
    """
    Return the N most similar games to the given profile.
    """
    matrix, scaler, tfidf, idx_df = _ensure_matrix(df)
    available = [f for f in SIMILARITY_FEATURES if f in df.columns]

    medians = df[available].median()
    if isinstance(game_profile, pd.Series):
        game_profile = game_profile.to_dict()

    # 1. Process Numerical Query
    query_values = []
    for f in available:
        val = game_profile.get(f)
        if pd.isna(val) or val is None:
            val = medians[f]
        query_values.append(float(val))
    
    query_values = np.array(query_values).reshape(1, -1)
    query_num = scaler.transform(query_values)

    # 2. Process Text Query (tags and genres)
    tags_val = str(game_profile.get("tags", ""))
    genres_val = str(game_profile.get("genres", ""))
    text_query = _parse_tags(tags_val) + " " + _parse_tags(genres_val)
    
    query_tfidf = tfidf.transform([text_query]).toarray()
    query_tfidf = query_tfidf * TFIDF_WEIGHT
    
    # 3. Combine and Compute Similarity
    query_final = np.hstack([query_num, query_tfidf])
    sims = cosine_similarity(query_final, matrix)[0]

    idx_df = idx_df.copy()
    idx_df["similarity_score"] = sims

    if exclude_app_id is not None:
        idx_df = idx_df[idx_df["app_id"] != exclude_app_id]

    top = idx_df.nlargest(n, "similarity_score")[["app_id", "similarity_score"]]

    # Merge back all columns from df to allow robust downstream filtering
    result = top.merge(df.drop_duplicates("app_id"), on="app_id", how="left")
    result["similarity_score"] = result["similarity_score"].round(4)
    return result.reset_index(drop=True)


def find_similar_by_appid(
    app_id: int,
    df: pd.DataFrame,
    n: int = SIMILARITY_N_DEFAULT,
) -> pd.DataFrame:
    """Find similar games given an app_id present in df."""
    row = df[df["app_id"] == app_id]
    if row.empty:
        raise ValueError(f"app_id {app_id} not found in dataset.")
    profile = row.iloc[0].to_dict()
    return find_similar_games(profile, df, n=n, exclude_app_id=app_id)
