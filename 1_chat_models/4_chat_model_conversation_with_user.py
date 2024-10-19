from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Load environment variables from .env
load_dotenv()

# Create a ChatOllama model
model = ChatOllama(model = "llama3")


chat_history = [] # Use a list to store messages


# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}")

print("---- Message History ----")
print(chat_history)