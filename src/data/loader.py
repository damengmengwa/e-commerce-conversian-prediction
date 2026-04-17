import pandas as pd
from pathlib import Path

from config import DATA_DIR, BOT_THRESH


def load_events(data_dir: Path = DATA_DIR, bot_thresh: int = BOT_THRESH) -> pd.DataFrame:
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
    print(f"  Raw events    : {len(events):,}")
    print(f"  Unique users  : {events['visitorid'].nunique():,}")
    print(f"  Date range    : {events['timestamp'].min().date()} → {events['timestamp'].max().date()}")
    print(f"  Purchase rate : {(events['event'] == 'transaction').mean() * 100:.3f}%")
    print(f"\n  Event breakdown:")
    print(events["event"].value_counts().to_string())


def _data_quality_audit(events: pd.DataFrame) -> None:
    null_counts = events.isnull().sum()
    if null_counts.sum() == 0:
        print("\n  No null values found.")
    else:
        print(f"\n  Null counts:\n{null_counts[null_counts > 0].to_string()}")

    dup_count = events.duplicated().sum()
    print(f"  Duplicates removed: {dup_count:,}")


def _remove_bots(events: pd.DataFrame, bot_thresh: int) -> pd.DataFrame:
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
