# Exemplo Fonte: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/

from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_ollama import ChatOllama

"""
Passos para replicar este exemplo:
1. Crie uma conta no Firebase
2. Crie um novo projeto Firebase
    - Copie o ID do projeto
3. Crie um banco de dados Firestore no projeto Firebase
4. Instale o Google Cloud CLI no seu computador
    - https://cloud.google.com/sdk/docs/install
    - Autentique o Google Cloud CLI com sua conta Google
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Defina seu projeto padrão para o novo projeto Firebase criado
5. Habilite a API Firestore no Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
""" 

load_dotenv()

# Configurar Firebase Firestore
PROJECT_ID = 'langchain-demo-abf48'
SESSION_ID = "user_session_new"  # Pode ser um nome de usuário ou um ID único
COLLECTION_NAME = "chat_history"

# Inicializar Cliente Firestore
print("Initializing Firestore Client...")

client = firestore.Client(project=PROJECT_ID)

chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client
)

# Inicializar Modelo de Chat
model = ChatOllama(model = "llama3")


print("Start chatting with the AI. Type 'exit' to quit.")


while True:
    
    human_input = input("User: ")
    
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    print(f"AI: {ai_response.content}")
