from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(model = "llama3")

# SystemMessage:
# Message for priming AI behavior, usually passed in as the first of a sequence
# of input messages.

# HumanMessagse:
#  Message from a human to the AI model.

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")


# AIMessage:
# Message from an AI
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print()
print(f"Answer from AI: {result.content}")