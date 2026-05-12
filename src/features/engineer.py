"""Feature engineering module for e-commerce user behavior analysis.

Generates 29 behavioral, temporal, and engagement features from raw event data
to capture user conversion intent and purchasing patterns.
"""
import pandas as pd
import numpy as np

from config import SESSION_GAP_SECONDS


def engineer_features(events: pd.DataFrame) -> pd.DataFrame:
    """Engineer 29 features from raw e-commerce event data.
    
    Feature categories:
    - Behavioral counts: views, carts, transactions, total events
    - Item diversity: unique items, diversity ratio
    - Temporal: recency, span, hourly patterns, weekday patterns
    - Session metrics: session count, gaps, active days, events per day
    - Engagement: return visitor flag, funnel depth, repeat item views
    - Derived ratios: view-to-cart, cart-to-purchase, conversion rates
    
    Args:
        events: DataFrame from load_events() with columns:
                [visitorid, timestamp, event, itemid]
    
    Returns:
        User-level feature matrix with shape (n_users, 31) including:
        - visitorid: user identifier
        - purchased: binary target (1 if user converted)
        - 29 feature columns
    """
    print("\n" + "=" * 60)
    print("[2] Feature Engineering  (29 features)")
    print("=" * 60)

    uid    = "visitorid"
    max_ts = events["timestamp"].max()
    ev     = _add_time_cols(events)

    feat = pd.concat(
        [
            _behavioral_counts(ev, uid),
            _item_diversity(ev, uid),
            _temporal_features(ev, uid, max_ts),
            _session_features(ev, uid),
            _engagement_features(ev, uid),
        ],
        axis=1,
    ).fillna(0)

    feat = _add_derived_ratios(feat)
    feat["purchased"] = (feat["total_trans"] > 0).astype(int)
    feat = feat.reset_index()

    print(f"  Users            : {len(feat):,}")
    print(f"  Features created : {feat.shape[1] - 2}")  # minus visitorid and purchased
    print(f"  Buyers           : {feat['purchased'].sum():,} ({feat['purchased'].mean() * 100:.3f}%)")
    return feat


def _add_time_cols(ev: pd.DataFrame) -> pd.DataFrame:
    """Add temporal aggregation columns from timestamp.
    
    Args:
        ev: Event dataframe with 'timestamp' column
        
    Returns:
        DataFrame with additional temporal columns: hour, weekday, date, is_weekend, funnel
    """
    ev = ev.copy()
    ev["hour"]       = ev["timestamp"].dt.hour
    ev["weekday"]    = ev["timestamp"].dt.dayofweek
    ev["date"]       = ev["timestamp"].dt.date
    ev["is_weekend"] = (ev["weekday"] >= 5).astype(int)
    ev["funnel"]     = ev["event"].map({"view": 1, "addtocart": 2, "transaction": 3})
    return ev


def _behavioral_counts(ev: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Count events by type for each user.
    
    Args:
        ev: Event dataframe with 'event' column and user identifier
        uid: Name of user ID column
        
    Returns:
        DataFrame with columns: total_views, total_carts, total_trans, total_events
    """
    views   = ev[ev["event"] == "view"].groupby(uid).size().rename("total_views")
    carts   = ev[ev["event"] == "addtocart"].groupby(uid).size().rename("total_carts")
    trans   = ev[ev["event"] == "transaction"].groupby(uid).size().rename("total_trans")
    total_e = ev.groupby(uid).size().rename("total_events")
    return pd.concat([views, carts, trans, total_e], axis=1)


def _item_diversity(ev: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Calculate item diversity metrics for each user.
    
    Args:
        ev: Event dataframe with 'itemid' column and user identifier
        uid: Name of user ID column
        
    Returns:
        DataFrame with columns: unique_items, item_diversity
    """
    total_e   = ev.groupby(uid).size().rename("total_events")
    u_items   = ev.groupby(uid)["itemid"].nunique().rename("unique_items")
    diversity = (u_items / (total_e + 1)).rename("item_diversity")
    return pd.concat([u_items, diversity], axis=1)


def _temporal_features(ev: pd.DataFrame, uid: str, max_ts: pd.Timestamp) -> pd.DataFrame:
    """Calculate temporal behavior patterns for each user.
    
    Args:
        ev: Event dataframe with temporal columns: hour, weekday, is_weekend, timestamp
        uid: Name of user ID column
        max_ts: Maximum timestamp in dataset (for recency calculation)
        
    Returns:
        DataFrame with columns: recency_days, span_days, avg_hour, hour_std,
                               avg_weekday, weekend_ratio
    """
    last_ts       = ev.groupby(uid)["timestamp"].max()
    first_ts      = ev.groupby(uid)["timestamp"].min()
    recency       = ((max_ts - last_ts).dt.days).rename("recency_days")
    span          = ((last_ts - first_ts).dt.days + 1).rename("span_days")
    avg_hour      = ev.groupby(uid)["hour"].mean().rename("avg_hour")
    hour_std      = ev.groupby(uid)["hour"].std().fillna(0).rename("hour_std")
    avg_weekday   = ev.groupby(uid)["weekday"].mean().rename("avg_weekday")
    weekend_ratio = ev.groupby(uid)["is_weekend"].mean().rename("weekend_ratio")
    return pd.concat([recency, span, avg_hour, hour_std, avg_weekday, weekend_ratio], axis=1)


def _session_features(ev: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Calculate session-based metrics for each user.
    
    Sessions are defined by time gaps > SESSION_GAP_SECONDS (default 30 min).
    
    Args:
        ev: Event dataframe with timestamp and date columns
        uid: Name of user ID column
        
    Returns:
        DataFrame with columns: session_count, avg_gap_min, active_days,
                               events_per_active_day
    """
    ev_s             = ev.sort_values([uid, "timestamp"]).copy()
    ev_s["prev_ts"]  = ev_s.groupby(uid)["timestamp"].shift(1)
    ev_s["gap_sec"]  = (ev_s["timestamp"] - ev_s["prev_ts"]).dt.total_seconds()
    ev_s["new_sess"] = (ev_s["gap_sec"] > SESSION_GAP_SECONDS) | ev_s["gap_sec"].isna()

    sessions    = ev_s.groupby(uid)["new_sess"].sum().rename("session_count")
    avg_gap     = (ev_s.groupby(uid)["gap_sec"].mean().fillna(0) / 60).rename("avg_gap_min")
    active_days = ev.groupby(uid)["date"].nunique().rename("active_days")
    total_e     = ev.groupby(uid).size().rename("total_events")
    epd         = (total_e / (active_days + 1)).rename("events_per_active_day")

    return pd.concat([sessions, avg_gap, active_days, epd], axis=1)


def _engagement_features(ev: pd.DataFrame, uid: str) -> pd.DataFrame:
    """Calculate user engagement indicators.
    
    Args:
        ev: Event dataframe with event and funnel columns
        uid: Name of user ID column
        
    Returns:
        DataFrame with columns: return_visitor, max_funnel_depth, repeat_item_views
    """
    active_days    = ev.groupby(uid)["date"].nunique()
    return_visitor = (active_days > 1).astype(int).rename("return_visitor")
    max_funnel     = ev.groupby(uid)["funnel"].max().rename("max_funnel_depth")

    item_views   = ev[ev["event"] == "view"].groupby([uid, "itemid"]).size().reset_index(name="cnt")
    repeat_views = (
        item_views[item_views["cnt"] > 1]
        .groupby(uid).size()
        .rename("repeat_item_views")
    )
    return pd.concat([return_visitor, max_funnel, repeat_views], axis=1)


def _add_derived_ratios(feat: pd.DataFrame) -> pd.DataFrame:
    """Calculate conversion funnel ratios and derived metrics.
    
    Args:
        feat: Feature dataframe with behavioral and engagement counts
        
    Returns:
        DataFrame with additional ratio columns:
        - v2c: view-to-cart conversion
        - c2p: cart-to-purchase conversion
        - events_per_session, items_per_session: efficiency metrics
        - cart_rate, trans_rate: event-level conversion rates
        - cart_abandon_rate: proportion of abandoned carts
    """
    feat["v2c"]                = feat["total_carts"]  / (feat["total_views"]   + 1)
    feat["c2p"]                = feat["total_trans"]  / (feat["total_carts"]   + 1)
    feat["events_per_session"] = feat["total_events"] / (feat["session_count"] + 1)
    feat["items_per_session"]  = feat["unique_items"]  / (feat["session_count"] + 1)
    feat["cart_rate"]          = feat["total_carts"]  / (feat["total_events"]  + 1)
    feat["trans_rate"]         = feat["total_trans"]  / (feat["total_events"]  + 1)
    feat["cart_abandon_rate"]  = (
        (feat["total_carts"] - feat["total_trans"]).clip(lower=0) / (feat["total_carts"] + 1)
    )
    return feat
