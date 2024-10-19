from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(model = "llama3")

# Define prompt templates (no need for separate Runnable chains)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)


# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()

# Run the chain
result = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(result)