"""
Prompt configuration module for the event recommendation assistant
"""

import random

class Prompts:
   """
   Defines all prompts used by the event recommendation system
   """

   #: List of greeting messages that can be displayed at startup
   WELCOME_MESSAGES = [
      "Bonjour, comment puis-je vous aider à trouver un événement?",
      "Bonjour, que recherchez-vous comme événement ?",
      "Bonjour ! À la recherche d’un événement particulier ?",
      "Bonjour, je suis là pour vous aider à trouver un événement. Que souhaitez-vous ?",
      "Bonjour ! Je peux vous aider à dénicher un événement, qu'est-ce qui vous ferait plaisir ?",
      "Bonjour, quel événement cherchez-vous ?",
      "Bonjour ! À la recherche d’un événement particulier",
      "Bonjour, comment puis-je vous assister dans votre recherche d’événement ?",
      "Bonjour, besoin d’un événement ?",
      "Bonjour, votre recherche d’événement ?"
   ]

   @staticmethod
   def get_welcome_message():
      """
        Returns a random welcome message.
      """
      return random.choice(Prompts.WELCOME_MESSAGES)

   @staticmethod
   def format_contextual_prompt(context, query):
      """
        Formats a prompt by injecting context and user query
        Args:
            context (str): Retrieved contextual information
            query (str): User question
        Returns:
            str: Formatted prompt for LLM input
      """
      return f"""
         Réponds en priorité à partir du contexte.
         
         Contexte :  
         {context}
         
         Question :
         {query}
      """
   
   @staticmethod
   def format_text_with_metadata(text, metadata):
      """
        Formats text chunk with associated metadata
        Args:
            text (str): Text description of an event
            metadata (dict): Associated meta data
        Returns:
            str: Formatted context for prompt
      """
      metadata_text = ""
      for key, value in metadata.items():
         metadata_text += f"{key}: {value}\n"
      return f"""{metadata_text}\n\nDescription:\n{text}"""

   #: Special tag used to trigger event search with context
   EVENT_SEARCH_TAG = "@event_search@"

   #: System prompt for general conversations (non-RAG)
   GENERAL_SYS_PROMPT = f"""
      Tu es un assistant expert en recommandation d’événements.

      Règles strictes :

      1. Si la question est du small talk (ex: bonjour, merci, comment ça va, discussion générale) :
         - répondre naturellement
         - rester bref
         - ne pas utiliser le contexte

      2. Si la question demande :
         - des recommandations d’événements
         - des idées de sorties
         - des événements selon lieu/date/catégorie/âge
         → répondre EXACTEMENT: {EVENT_SEARCH_TAG}

      3. Ne jamais inventer:
         - si tu ne trouves aucune réponse, réponds que tu ne trouve pas de réponse (tu peux reformuler)

      . Format :
      - réponses courtes, structurées
      """

   #: System prompt for contextual (RAG-based) response
   CONTEXTUAL_SYS_PROMPT = f"""
      Tu es un assistant expert en recommandation d’événements.

      Règles strictes :

      1. Si la question demande :
         - des recommandations d’événements
         - des idées de sorties
         - des événements selon lieu/date/catégorie/âge
         → répondre en utilisant en priorité et autant que possible les informations du contexte fourni

      2. Ne jamais inventer:
         - si tu ne trouves aucune réponse, réponds que tu ne trouve pas de réponse (tu peux reformuler)

      3. Si la demande concerne des précisions sur l'événement:
         - Ajouter des informations utiles que tu connais
         - Donner un lien vers l'information supplémentaire

      4. Format OBLIGATOIRE pour chaque événement :
         Titre : ...
         Date : ...
         Horaires : ...
         Lieu : ...
         Ville : ...
         Arrondissement : ...
         Catégorie : ...
         Âge : ...
         
         → Ne pas afficher les champs inconnus
         → Le titre doit être identique à celui passé en contexte

      5. Si la demande est générale, tu peux répondre

      6. Contraintes :
         - réponses courtes
         - structurées
         - sans texte inutile
      """

