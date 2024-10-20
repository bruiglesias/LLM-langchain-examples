import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Definir o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")


def create_vector_store():
    """Raspar o site, dividir o conteúdo, criar embeddings e persistir o armazenamento vetorial."""
    # Definir a chave da API do Firecrawl
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    # Passo 1: Raspagem do site usando FireCrawlLoader
    print("Begin crawling the website...")
    loader = FireCrawlLoader(
        api_key=api_key, url="https://apple.com", mode="scrape")
    docs = loader.load()
    print("Finished crawling the website.")

    # Converter valores de metadados em strings se forem listas
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Passo 2: Dividir o conteúdo raspado em partes
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    # Exibir informações sobre os documentos divididos
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")
    print(f"Sample chunk:\n{split_docs[0].page_content}\n")

    # Passo 3: Criar embeddings para as partes dos documentos
    embeddings = OllamaEmbeddings(model="llama3")

    # Passo 4: Criar e persistir o armazenamento vetorial com os embeddings
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(
        split_docs, embeddings, persist_directory=persistent_directory
    )
    print(f"--- Finished creating vector store in {persistent_directory} ---")


# Verificar se o armazenamento vetorial Chroma já existe
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(
        f"Vector store {persistent_directory} already exists. No need to initialize.")

# Carregar o armazenamento vetorial com os embeddings
embeddings = OllamaEmbeddings(model="llama3")
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)


# Passo 5: Consultar o armazenamento vetorial
def query_vector_store(query):
    """Consultar o armazenamento vetorial com a pergunta especificada."""
    # Criar um recuperador para consultar o armazenamento vetorial
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Recuperar documentos relevantes com base na consulta
    relevant_docs = retriever.invoke(query)

    # Exibir os resultados relevantes com metadados
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# Definir a pergunta do usuário
query = "Apple Intelligence?"

# Consultar o armazenamento vetorial com a pergunta do usuário
query_vector_store(query)
