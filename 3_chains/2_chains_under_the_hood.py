from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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

# Criar runnables individuais (etapas na cadeia)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# Criar a RunnableSequence (equivalente à cadeia LCEL)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Executar a cadeia
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Saída
print(response)
