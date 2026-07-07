import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
import json
from events_rag_project.config.paths import CSV_EVENTS_FILE
import re
from unidecode import unidecode

"""
Fields used for indexing 
"""
used_fields = [
    "keywords_fr",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "location_name",
    "firstdate_begin",
    "lastdate_end",
    "status",
    "age_min",
    "age_max",
    "location_address",
    "location_district",
    "location_department",
    "location_postalcode",
    "location_city",
    "location_coordinates",
    "timings",
    "conditions_fr",
    "canonicalurl",
    "daterange_fr"
]

class AgendaImporter:
    """
    Import public events from the OpenDataSoft/OpenAgenda API and stores them as a CSV file
    """

    def __init__(self, city: str = "paris", days_back: int = 365, days_front: int = 365, limit: int = 100, max_limit: int = 10000):
        """
        Initializes Agenda Importer
        Args:
            city (str): City name used for filtering (default: "paris")
            days_back (int): Number of days to look back for events (default: 365)
            days_front (int): Number of days to look front for events (default: 365)
            limit (int): Number of records per API call (default: 100)
            max_limit (int): Maximum number of records to fetch (default: 10000)
        """
        self.url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
        self.city = city.lower()
        self.limit = limit
        self.max_limit = max_limit
        self.start_date  = (datetime.now()-timedelta(days=days_back)).strftime("%Y-%m-%d")
        self.end_date  = (datetime.now()+timedelta(days=days_front)).strftime("%Y-%m-%d")

    def run(self):
        """
        Executes the full pipeline:
        - import events 
        - serialize complex fields (list, dict, set -> str)
        - normalize text fields (remove accents and put everything in lowercase)
        - remove duplicated records
        - save result to csv file
        """
        try:
            df = self.__import_events()
            df = self.__serialize_complex_fields(df)
            df = self.__normalize_texts_fields(df)
            df = self.__remove_duplicates(df)
            self.__save_to_csv(df)
        except Exception as e:
            print(f"Failed to complete importation process: {e}")

    def __import_events(self) -> pd.DataFrame:
        """
        Imports events from OpenAgenda API
        Returns:
            pd.DataFrame: Event records
        """
        all_results = []
        offset = 0

        print(f"\nStart to import")

        while True:
            params = self.__build_params(offset)
            response = requests.get(self.url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            offset += self.limit
            sys.stdout.write(f"\rImported events: {len(all_results)}")
            sys.stdout.flush()

            if offset >= self.max_limit:
                break

        return pd.DataFrame(all_results)
    
    def __build_params(self, offset: int) -> dict:
        """
        Builds query parameters for the API request
        Args:
            offset (int): Pagination offset
        Returns:
            dict: Query parameters for the request
        """
        return {
            "limit": self.limit,
            "offset": offset,
            "where": f"lower(location_city) = '{self.city}' AND lastdate_end >= date'{self.start_date}' AND firstdate_begin <= date'{self.end_date}'"
        }

    def __serialize_complex_fields(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Convert list/dict/set columns into serializable strings
        Args:
            df_in (pd.DataFrame): Input DataFrame
        Returns:
            df (pd.DataFrame): DataFrame with serialized complex fields
        """
        print(f"\nSerialize complex fields...")
        df = df_in.copy()
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
                if isinstance(x, (list, dict, set))
                else x
            )
        return df
    
    def __normalize_texts_fields(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize textual fields (lowercase, remove accents, trim spaces, collapse multiple spaces)
        Args:
            df_in (pd.DataFrame): Input DataFrame
        Returns:
            df (pd.DataFrame): DataFrame with normalized textual fields

        """
        print(f"\nNormalization...")
        df = df_in.copy()
        text_columns = df.select_dtypes(include="object").columns
        for col in text_columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .apply(self.__normalize_one_text)
            )
        return df
    
    def __normalize_one_text(self, text: str) -> str:
        """
        Normalize text (lowercase, remove accents, trim spaces, collapse multiple spaces)
        Args:
            text (str): Text to normalize
        Returns:
            text (str): Normalized text
        """
        text = unidecode(text)          # Remove accents
        text = text.casefold()          # Lowercase
        text = text.strip()             # Trim spaces
        text = re.sub(r"\s+", " ", text)  # Collapse spaces/tabs/newlines
        return text
    
    def __save_to_csv(self, df_in: pd.DataFrame, output_path: str = CSV_EVENTS_FILE) -> None:
        """
        Saves the DataFrame to a CSV file
        Args:
            df (pd.DataFrame): DataFrame to save
            output_path (str): Path to the output CSV file
        """
        df_in.to_csv(output_path, index=False)
        print(f"\nFinished. {len(df_in)} events saved to {output_path}")

    def __remove_duplicates(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicates from dataframe
        Args:
            df_in (pd.DataFrame): Input DataFrame
        Returns:
            df (pd.DataFrame): DataFrame without duplicated
        """
        print(f"\nSearch for duplicates...")
        df = df_in.copy()
        duplicates = df[df.duplicated(subset=used_fields, keep=False)]
        if len(duplicates) > 0:
            df = df.drop_duplicates(subset=used_fields)
            print(f"Found {len(duplicates)} duplicates")
        return df


if __name__ == "__main__":
    """
    Entry point for script execution
    Runs the full event import pipeline
    """
    importer = AgendaImporter()
    importer.run()
