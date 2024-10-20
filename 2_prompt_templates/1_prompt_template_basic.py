# Documentação sobre Prompt Template:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# PARTE 1: Criar um ChatPromptTemplate usando uma string de template
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)


# PARTE 2: Prompt com Múltiplos Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""

prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt)


# PARTE 3: Prompt com Mensagens de Sistema e Humanas (Usando Tuplas)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)


# Informação Extra sobre a Parte 3

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)


# Isto NÃO funciona:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
