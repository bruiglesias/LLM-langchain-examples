from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")

# PARTE 1: Criar um ChatPromptTemplate usando uma string de template
print("-----Prompt from Template-----")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)


# PARTE 2: Prompt com Múltiplos Placeholders
print("\n----- Prompt with Multiple Placeholders -----\n")
template_multiple =  """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)

prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
result = model.invoke(prompt)
print(result.content)

# PARTE 3: Prompt com Mensagens de Sistema e Humanas (Usando Tuplas)
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)
