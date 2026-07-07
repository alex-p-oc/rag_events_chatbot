from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from events_rag_project.config.paths import FAISS_INDEX_DIR
from events_rag_project.config.model_options import MODEL_NAME

def test_faiss_load():
    """
    Test load of FAISS vector
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    assert len(db.index_to_docstore_id) > 0

def test_faiss_search():
    """
    Test search on FAISS index
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    results = db.similarity_search("spectacle paris", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)

    results = db.similarity_search("forum emploi", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)

    results = db.similarity_search("evennement familial", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)

    results = db.similarity_search("sortie musicale", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)

    results = db.similarity_search("un film au cinéma ?", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)

    results = db.similarity_search("une exposition dans un musee", k=3)
    assert len(results) > 0
    assert all(r.page_content for r in results)


def test_faiss_docstore_alignment():
    """
    Test that FAISS index corresponds to document store
    """
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    assert len(db.docstore._dict) == len(db.index_to_docstore_id)