from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")

# SystemMessage:
# Mensagem usada para ajustar o comportamento da IA, geralmente passada como a primeira
# de uma sequência de mensagens de entrada.

# HumanMessage:
# Mensagem enviada por um humano para o modelo de IA.

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# AIMessage:
# Mensagem da IA
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invocar o modelo com as mensagens
result = model.invoke(messages)
print()
print(f"Answer from AI: {result.content}")
