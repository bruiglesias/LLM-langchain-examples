from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Criar um modelo ChatOllama
model = ChatOllama(model = "llama3")

# Definir templates de prompt (não há necessidade de cadeias separadas de Runnable)
messages = [
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

# Definir etapa de análise dos prós
def analyze_pros(features):
    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the pros of these features."),
    ]
    pros_template = ChatPromptTemplate.from_messages(messages)
    return pros_template.format_prompt(features=features)

# Definir etapa de análise dos contras
def analyze_cons(features):
    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the cons of these features."),
    ]
    cons_template = ChatPromptTemplate.from_messages(messages)
    return cons_template.format_prompt(features=features)

# Combinar prós e contras em uma revisão final
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Simplificar ramificações com LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Criar a cadeia combinada usando LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Executar a cadeia
result = chain.invoke({"product_name": "MacBook Pro"})

# Saída
print(result)
