"""Data loading and cleaning module for e-commerce events.

This module handles loading raw event data, performing quality audits,
and removing anomalies like bot activity before downstream processing.
"""
import pandas as pd
from pathlib import Path

from config import DATA_DIR, BOT_THRESH


def load_events(data_dir: Path = DATA_DIR, bot_thresh: int = BOT_THRESH) -> pd.DataFrame:
    """Load, clean, and preprocess e-commerce event data.
    
    Pipeline:
    1. Read raw events from CSV
    2. Convert timestamps from milliseconds to datetime
    3. Remove exact duplicates
    4. Audit data quality (nulls, duplicates)
    5. Filter out bot accounts (>thresh events/day)
    
    Args:
        data_dir: Path to directory containing 'events.csv'
        bot_thresh: Maximum events per user per day before flagging as bot
        
    Returns:
        Cleaned DataFrame with columns: visitorid, timestamp, event, itemid
    """
    print("\n" + "=" * 60)
    print("[1] Loading and Cleaning Data")
    print("=" * 60)

    events = pd.read_csv(Path(data_dir) / "events.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events.drop_duplicates()

    _print_raw_stats(events)
    _data_quality_audit(events)
    events = _remove_bots(events, bot_thresh)

    print(f"\n  Clean events : {len(events):,}")
    print(f"  Clean users  : {events['visitorid'].nunique():,}")
    return events


def _print_raw_stats(events: pd.DataFrame) -> None:
    """Print summary statistics of raw event data.
    
    Args:
        events: DataFrame with event-level data
    """
    print(f"  Raw events    : {len(events):,}")
    print(f"  Unique users  : {events['visitorid'].nunique():,}")
    print(f"  Date range    : {events['timestamp'].min().date()} → {events['timestamp'].max().date()}")
    print(f"  Purchase rate : {(events['event'] == 'transaction').mean() * 100:.3f}%")
    print(f"\n  Event breakdown:")
    print(events["event"].value_counts().to_string())


def _data_quality_audit(events: pd.DataFrame) -> None:
    """Audit and report data quality issues.
    
    Reports:
    - Null value counts by column
    - Number of exact duplicates
    
    Args:
        events: DataFrame to audit
    """
    null_counts = events.isnull().sum()
    if null_counts.sum() == 0:
        print("\n  No null values found.")
    else:
        print(f"\n  Null counts:\n{null_counts[null_counts > 0].to_string()}")

    dup_count = events.duplicated().sum()
    print(f"  Duplicates removed: {dup_count:,}")


def _remove_bots(events: pd.DataFrame, bot_thresh: int) -> pd.DataFrame:
    """Filter out likely bot accounts based on daily event volume.
    
    Users with >bot_thresh events on any single day are flagged as bots.
    
    Args:
        events: DataFrame with event data and 'timestamp' column
        bot_thresh: Daily event threshold for bot detection
        
    Returns:
        Filtered DataFrame excluding bot users
    """
    events["date"] = events["timestamp"].dt.date
    daily_counts   = events.groupby(["visitorid", "date"]).size()
    bot_users      = (
        daily_counts[daily_counts > bot_thresh]
        .index.get_level_values(0)
        .unique()
    )
    clean = events[~events["visitorid"].isin(bot_users)]
    print(f"\n  Bots removed (>{bot_thresh} events/day): {len(bot_users):,}")
    return clean
