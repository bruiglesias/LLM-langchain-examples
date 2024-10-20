# Chat Model Documents: https://python.langchain.com/docs/integrations/chat/
# Ollama Chat Model Documents: https://python.langchain.com/docs/integrations/chat/ollama/

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()


# Execute antes: ollama pull llama3
# Criar um modelo ChatOllama
model = ChatOllama(
    model = "llama3",
    temperature = 0.8,
    num_predict = 256,
    # outros parâmetros ...
)


result = model.invoke("what is 81 divided by 9?")

print("Resultado completo:")
print(result)
print()
print("Somente o conteúdo:")
print(result.content)
