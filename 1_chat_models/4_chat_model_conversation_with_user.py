from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")


chat_history = []  # Usar uma lista para armazenar as mensagens


# Definir uma mensagem inicial do sistema (opcional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# Loop do chat
while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    # Obter resposta da IA usando o histórico
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")

print("---- Message History ----")
print(chat_history)
