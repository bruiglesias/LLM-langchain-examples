from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(model = "llama3")

# Define prompt templates (no need for separate Runnable chains)
messages = [
    ("system", "You are an expert product reviewer."),
    ("human", "List the main features of the product {product_name}."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)


# Define pros analysis step
def analyze_pros(features):
    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the pros of these features."),
    ]
    pros_template = ChatPromptTemplate.from_messages(messages)
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    messages = [
        ("system", "You are an expert product reviewer."),
        ("human", "Given these features: {features}, list the cons of these features."),
    ]
    cons_template = ChatPromptTemplate.from_messages(messages)
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"



# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "MacBook Pro"})

# Output
print(result)