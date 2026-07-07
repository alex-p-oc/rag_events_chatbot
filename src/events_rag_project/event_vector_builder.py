import pandas as pd
import pickle
import ast
import json
from unidecode import unidecode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from events_rag_project.config.paths import CSV_EVENTS_FILE, FAISS_INDEX_DIR, STORE_FILE
from events_rag_project.config.model_options import MODEL_NAME, download_local_model


class EventVectorBuilder:
    """
    Builds vector representations of event data by transforming text and metadata into embeddings and storing them in a searchable index
    """

    ATTENDANCE_MODE_MAP = {
        "online": "en ligne",
        "offline": "en présentiel",
        "mixed": "hybride"
    }

    def __init__(self):
        """Initialization"""
        print("Initialization...")
        self.texts = []
        self.metadatas = []

    def build(self):
        """
        Build and store FAISS index
        """
        try:
            print("Reading source events data file...")
            df = pd.read_csv(CSV_EVENTS_FILE)

            print("Embedding process...")
            self.__process_data(df)

            print("Saving embeddings to store file...")
            self.__save_store_to_file()

            print("Download model locally...")
            db = download_local_model()

            print("Building vector index...")
            db = self.__build_vector_store()

            print("Saving index to local file...")
            db.save_local(FAISS_INDEX_DIR)

            print("Complete")
        except Exception as e:
            print(f"Failed to complete vector index building process: {e}")

    def __process_data(self, df):
        """
        Processes dataFrame into texts and metadata
        Args:
            df (pd.DataFrame): data
        """
        for _, row in df.iterrows():
            text = self.__build_text_resume(row)

            if text.strip():
                self.texts.append(text)
                self.metadatas.append(self.__build_metadata(row))

    def __save_store_to_file(self):
        """
        Saves processed texts and metadata to a pickle file.
        """
        with open(STORE_FILE, "wb") as f:
            pickle.dump({"texts": self.texts,"metadatas": self.metadatas }, f)


    def __build_vector_store(self):
        """
        Creates FAISS vector store from texts and metadata
        Returns:
            FAISS: Vector database instance
        """
        embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"}, 
                                                encode_kwargs={"normalize_embeddings": True})
        return FAISS.from_texts(self.texts, embedding_model, metadatas=self.metadatas)
    
    def __build_text_resume(self, row) -> str:
        """
        Builds a text summary from a data row
        We only keep the 200 first characters of description and longdescription, so the full
        resume is less than 500 characters.
        Args:
            row: Data row.
        Returns:
            str: Text resume of a row
        """
        sections = []
        self.__add_text_list_section(sections, row, "keywords_fr", "evenement concernant")
        self.__add_text_section(sections, row, "title_fr", "titre")
        self.__add_text_section(sections, row, "description_fr", "description", 200)
        self.__add_text_section(sections, row, "longdescription_fr", "description longue", 200)
        self.__add_text_section(sections, row, "location_name", "lieu")
        return "\n".join(sections)

    def __build_metadata(self, row) -> dict:
        """
        Extracts structured metadata from a row
        Args:
            row: Data row
        Returns:
            dict: Metadata dictionary
        """
        metadata = {}
        self.__add_metadata_field(metadata, row, "firstdate_begin", "debut")
        self.__add_metadata_field(metadata, row, "lastdate_end", "fin")
        self.__add_metadata_field_from_json(metadata, row, "status", "status", "label", "fr")
        self.__add_metadata_field(metadata, row, "mode", "mode")
        self.__add_metadata_field(metadata, row, "mode", "mode de participation", self.ATTENDANCE_MODE_MAP)
        self.__add_metadata_field(metadata, row, "age_min", "age minimum")
        self.__add_metadata_field(metadata, row, "age_max", "age maximum")
        self.__add_metadata_field(metadata, row, "location_address", "adresse")
        self.__add_metadata_field(metadata, row, "location_district", "quartier")
        self.__add_metadata_field(metadata, row, "location_department", "departement")
        self.__add_metadata_field(metadata, row, "location_postalcode", "arrondissement")
        self.__add_metadata_field(metadata, row, "location_city", "ville")
        self.__add_metadata_field(metadata, row, "location_coordinates", "coordonnees GPS")
        self.__add_metadata_field(metadata, row, "timings", "horaires d'ouverture")
        self.__add_metadata_field(metadata, row, "conditions_fr", "conditions d’acces, tarif")
        self.__add_metadata_field(metadata, row, "canonicalurl", "lien web evennement")
        self.__add_metadata_field(metadata, row, "daterange_fr", "date")
        return metadata
    
    def __add_metadata_field(self, metadata: dict, row, field_name: str, output_name: str, mapping: dict = None):
        """
        Adds a key-value pair to metadata
        Args:
            metadata (dict): Metadata dictionary
            row: Data row
            field_name (str): Column name
            output_name (str): Output key name
            mapping (dict, optional): Value mapping dictionary
        """
        value = row.get(field_name)
        if self.__is_valid(value):
            if mapping:
                value = mapping.get(value, value)
            metadata[output_name] = value

    def __add_metadata_field_from_json(self, metadata: dict, row, field_name: str, output_name: str, json_key: str, json_subkey: str):
        """
        Adds a key-value pair to metadata
        Args:
            metadata (dict): Metadata dictionary
            row: Data row
            field_name (str): Column name
            output_name (str): Output key name
            json_key (str): json content key
            json_subkey (str): json content sub-key
        """
        try:
            value = row.get(field_name)
            data = json.loads(value)
            metadata[output_name] = data.get("label", {}).get("fr")
        except:
            ...

    def __add_text_section(self, sections: list, row, field_name: str, section_name: str, max_length: int = None):
        """
        Adds a text section to the resume if valid
        Args:
            sections (list): List of text sections.
            row: Data row.
            field_name (str): Column name.
            section_name (str): Label for the section.
            max_length (int): Maximum text length. If text is longuer it will be trunkated. If None(default), not used.
        """
        value = row.get(field_name)
        if self.__is_valid(value):
            if max_length is not None and len(value) > max_length:
                value = value[:max_length]  + "..."
            sections.append(f"{section_name}: {value}")

    def __add_text_list_section(self, sections: list, row, field_name: str, section_name: str):
        """
        Adds a text section (from a list)
        Args:
            sections (list): List of text sections
            row: Data row
            field_name (str): Column name containing a serialized list
            section_name (str): Label for the section
        """
        value = row.get(field_name)
        if self.__is_valid(value):
            parsed = ast.literal_eval(value)
            result = ", ".join(parsed)
            if self.__is_valid(result):
                sections.append(f"{section_name}: {result}")

    def __is_valid(self, value) -> bool:
        """
        Checks if a value is non-null and non-empty
        Args:
            value: Input value
        Returns:
            bool: True if valid, False otherwise
        """
        return pd.notna(value) and str(value).strip() != ""

if __name__ == "__main__":
    """
    Entry point for script execution
    Builds vector representations of agenda events
    """
    event_vector_builder = EventVectorBuilder()
    event_vector_builder.build()
