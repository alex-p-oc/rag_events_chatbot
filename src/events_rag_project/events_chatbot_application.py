from events_rag_project.event_rag_service import EventRAGService
from events_rag_project.config.prompts import Prompts
import pandas as pd


class EventsChatbotApplication:
    """
    Command-line chat application using an EventRAGService backend
    Simple interactive interface for querying
    """

    def __init__(self):
        """
        Initializes chat application
        """
        self.chat_service = self.__initialize_service()

    def __initialize_service(self):
        """une sortie dans un cinéma
        Initializes the EventRAGService
        Returns initialized service instance or None if failed
        """
        try:
            return EventRAGService()
        except Exception as e:
            print(f"L'initialisation du service a échouée: {e}")
            return None

    def run(self):
        """
        Starts the interactive chat loop
        """
        print("Initialisaton du service de chatbot...")
        if not self.chat_service:
            print("Impossible de démarrer le service.")
            return

        print(f"\n{Prompts.get_welcome_message()}\n")
        try:
            while True:
                user_input = input("\nVous: ")
                response = self.chat_service.ask(user_input)
                print(f"\nBot: {response}")

        except KeyboardInterrupt:
            pass
        print("\nBot: Au revoir!")

if __name__ == "__main__":
    """
    Entry point for running the chat application
    """
    app = EventsChatbotApplication()
    app.run()