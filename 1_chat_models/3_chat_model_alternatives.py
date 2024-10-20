from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# Configurar variáveis de ambiente e mensagens
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# ---- Exemplo de Modelo de Chat OpenAI com LangChain ----
model = ChatOpenAI(model="gpt-4o")

# Invocar o modelo com as mensagens
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Exemplo de Modelo de Chat Anthropic ----

# Criar um modelo da Anthropic
# Modelos Anthropic: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Exemplo de Modelo de Chat Google ----
# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
