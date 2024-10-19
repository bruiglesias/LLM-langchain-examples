# Chat Model Documents: https://python.langchain.com/docs/integrations/chat/
# Ollama Chat Model Documents: https://python.langchain.com/docs/integrations/chat/ollama/

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables from .env
load_dotenv()


# Execute before: ollama pull llama3
# Create a ChatOllama model
model = ChatOllama(
    model = "llama3",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)


result = model.invoke("what is 81 divided by 9?")

print("Full result:")
print(result)
print()
print("Content only:")
print(result.content)