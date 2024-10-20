import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Carrega variáveis de ambiente do .env
load_dotenv()

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir,
    "db",
    "chroma_db_with_metadata"
)

# Define o modelo de embedding
embeddings = OllamaEmbeddings(model="llama3")

# Carrega o vetor de armazenamento existente com a função de embedding
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define a pergunta do usuário
query = "How can I learn more about LangChain?"

# Recupera documentos relevantes com base na consulta
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
relevant_docs = retriever.invoke(query)

# Exibe os resultados relevantes com metadados
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combina a consulta e os conteúdos dos documentos relevantes
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    +  "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# Cria um modelo ChatOllama
model = ChatOllama(model="llama3")

# Define as mensagens para o modelo
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Invoca o modelo com a entrada combinada
result = model.invoke(messages)

# Exibe o resultado completo e o conteúdo apenas
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
