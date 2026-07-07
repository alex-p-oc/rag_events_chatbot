from events_rag_project.event_rag_service import EventRAGService
import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from events_rag_project.config.paths import FAISS_INDEX_DIR
from events_rag_project.config.model_options import MODEL_NAME
from datetime import datetime
import time
from unidecode import unidecode

question_expected_answers_couples = [
    (
        "Un concert rock pour la fête de la musique dans le 5e arrondissement",
        "Concert covers Pop/Rock CHANNY ON THE ROCK"
    ),
    (
        "Un concert tribute à Abba",
        "ABBA MANIA, The Abba Tribute"
    ),
    (
        "Je recherche un atelier de mosaïque à l'institut du monde arabe",
        "Atelier de mosaïque par Joëlle Fayad Geagea"
    ),
    (
        "Un atelier créatif au musée bourdelle sur le sujet de la mythologie accessible aux enfants",
        "Atelier créatif \"Figures de la mythologie\""
    ),
    (
        "Un concert de musique classique au nouvel an",
        "Concert du Nouvel an à Paris"
    ) ,
    (
        "Un spectacle de danses traditionelles du japon",
        "Danses et musiques traditionnelles du Japon"
    ),
    (
        "Une visite libre du musée Clémenceau",
        "Visite libre du Musée Clemenceau"
    ),
    (
        "A l'occasion de la Nuit des musées je souhaite visiter de la Bnf (site Richelieu)",
        "Une nuit au musée de la BnF | Richelieu"
    ),
    (
        "Un forum emploi à bercy village",
        "FORUM MULTISECTEURS A BERCY VILLAGE"
    ),
    (
        "Une sortie le dimanche pour les enfants parc des buttes-chaumont",
        "Dimanche au Vert en famille avec enfants de 3 à 10 ans. Paris 19."
    )
]

def save_result(df, filename):
    """
    Save test results to a log file
    """
    try:
        df.to_csv(filename, index=False, encoding="utf-8-sig", sep=";")
    except:
        pass

def test_answer_quality():
    """
    Test 10 specific questions and compare result to expected result
    """
    df = pd.DataFrame(columns=["question","answer","expected answer","result","faiss_result"])
    chat_service = EventRAGService()
    
    for question, expected in question_expected_answers_couples:

        question = unidecode(question.lower())  

        # Clear chat history between each question
        chat_service.clear_history()

        answer = chat_service.ask(question)
        result = unidecode(expected.lower()) in unidecode(answer.lower())

        # Check if we find expected answeer with Faiss index
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        results = db.similarity_search(question, k=10)
        faiss_result = any(unidecode(expected.lower()) in unidecode(r.page_content.lower()) for r in results)
        assert all(r.page_content for r in results)

        # Result is stored in a dataframe that we will save in a csv file
        df.loc[len(df)] = [question,answer,expected,result,faiss_result]

        time.sleep(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_result(df, f"./logs/answer_quality_{timestamp}.csv")
    assert True
    
if __name__ == "__main__":
    pytest.main([__file__,"-s","-vv"])
    