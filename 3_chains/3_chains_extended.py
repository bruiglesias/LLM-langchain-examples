from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")

# Definir templates de prompt (não há necessidade de cadeias separadas de Runnable)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)

# Definir etapas de processamento adicionais usando RunnableLambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Criar a cadeia combinada usando LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

# Executar a cadeia
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Saída
print(result)
