import pandas as pd
import os
from events_rag_project.config.paths import CSV_EVENTS_FILE
from datetime import datetime, timedelta, timezone


def test_events_city():
    """
    Test that all records in events.csv have Paris for city
    """
    df = pd.read_csv(CSV_EVENTS_FILE)
    assert df["location_city"].isna().sum() == 0
    assert len(df) > 1
    assert (df["location_city"].str.lower() == "paris").all()

def test_events_date():
    """
    Test that all records in events.csv are recent events
    lastdate_end >= (csv_file modification date - 365 days)
    """
    file_modification_time = datetime.fromtimestamp(os.path.getmtime(CSV_EVENTS_FILE), tz=timezone.utc)
    df = pd.read_csv(CSV_EVENTS_FILE)
    df["lastdate_end"] = pd.to_datetime(df["lastdate_end"], errors="coerce")
    start_date = (file_modification_time-timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    assert df["lastdate_end"].isna().sum() == 0
    assert len(df) > 1
    assert (df["lastdate_end"] >= start_date).all()
