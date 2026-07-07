import pickle
from events_rag_project.config.paths import STORE_FILE

def test_store_load():
    """
    Test that store file can be load
    """
    with open(STORE_FILE, "rb") as store_file:
        store = pickle.load(store_file)
    assert isinstance(store, dict)

def test_store_qtt_of_texts_vs_metatdatas():
    """
    Test that store file contains same qunatity of texts chunk than metadatas
    """
    with open(STORE_FILE, "rb") as store_file:
        store = pickle.load(store_file)
    assert len(store["texts"]) > 0
    assert len(store["texts"]) == len(store["metadatas"])
    
def test_texts_not_empty():
    """
    Test that texts are not empty
    """
    with open(STORE_FILE, "rb") as store_file:
        store = pickle.load(store_file)
    for txt in store["texts"]:
        assert isinstance(txt, str)
        assert txt.strip() != None

def test_metadatas_not_empty():
    """
    Test that metadatas are not empty
    """
    with open(STORE_FILE, "rb") as store_file:
        store = pickle.load(store_file)
    for metadata in store["metadatas"]:
        assert isinstance(metadata, dict)
        assert len(metadata) > 0