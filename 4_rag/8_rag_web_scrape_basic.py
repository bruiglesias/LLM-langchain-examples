import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Carrega variáveis de ambiente do .env
load_dotenv()

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_apple")

# Etapa 1: Raspagem do conteúdo de apple.com usando WebBaseLoader
# WebBaseLoader carrega páginas da web e extrai seu conteúdo
urls = ["https://www.apple.com/"]

# Cria um loader para o conteúdo da web
loader = WebBaseLoader(urls)
documents = loader.load()

# Etapa 2: Divide o conteúdo raspado em pedaços
# CharacterTextSplitter divide o texto em pedaços menores
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Exibe informações sobre os documentos divididos
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Etapa 3: Cria embeddings para os pedaços de documentos
# OllamaEmbeddings transforma texto em vetores numéricos que capturam significado semântico
embeddings = OllamaEmbeddings(model="llama3")

# Etapa 4: Cria e persiste o armazenamento de vetores com os embeddings
# Chroma armazena os embeddings para busca eficiente
if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store in {persistent_directory} ---")
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Etapa 5: Consulta o armazenamento de vetores
# Cria um retriever para consultar o armazenamento de vetores
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define a pergunta do usuário
query = "What new products are announced on Apple.com?"

# Recupera documentos relevantes com base na pergunta
relevant_docs = retriever.invoke(query)

# Exibe os resultados relevantes com metadados
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
