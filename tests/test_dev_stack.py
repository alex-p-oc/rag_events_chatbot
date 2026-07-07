"""
Test technical stack compatibility
"""

import requests
import numpy as np
import pickle
import ast
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sentence_transformers import SentenceTransformer
from unidecode import unidecode
import re


def test_pickle():
    """
    Test pickle serialization & de-serialization
    """
    data = {"a": 1}
    dumped = pickle.dumps(data)
    loaded = pickle.loads(dumped)
    assert loaded == data

def test_ast():
    """
    Test ast, used to convert python list to text litteral equivalent
    """
    value = "['a', 'b']"
    parsed = ast.literal_eval(value)
    assert parsed == ['a', 'b']

def test_numpy():
    """
    Test a basic numpy operation 
    """
    a = np.array([1, 2, 3])
    assert a.sum() == 6

def test_request():
    """
    Test requests library
    """
    response = requests.get("https://api.github.com", timeout=5)
    assert response.status_code == 200

def test_faiss():
    """
    Test FAISS
    Creates a vector index from text data and verifies that the similarity search returns relevant results.
    """
    texts = ["This is a test", "LangChain"]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_texts(texts, embeddings)
    results = db.similarity_search("test", k=2)
    assert len(results) == 2
    assert any("test" in r.page_content.lower() for r in results)

def test_mistral_api():
    """
    Test Mistral API call
    Send a request to the model and check that there is an answer
    """
    load_dotenv()
    client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.chat(model="mistral-small", 
                           messages=[ChatMessage(role="user", content="Can you say me hello in top 5 languages")])
    assert "Hello" in response.choices[0].message.content

def test_sentence_transformer_model_load():
    """
    Test SentenceTransformer
    Load an embedding model and generate a vector form text
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode("test")
    assert embedding is not None
    assert len(embedding) > 0

def test_unidecode():
    """
    Test normalization of text with unidecode
    """
    text = "Être à un événnement"
    expected_result = "Etre a un evennement"
    result = unidecode(text)
    assert result == expected_result

def test_regular_expression():
    """
    Test normalization of text with unidecode
    """
    text = "Mon\t\t    event\n\n\nsur plusieurs\r\n\t lignes      !"
    result = re.sub(r"\s+", " ", text)
    assert result == "Mon event sur plusieurs lignes !"