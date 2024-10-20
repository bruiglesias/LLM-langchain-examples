import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Define o diretório que contém o arquivo de texto e o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Verifica se o arquivo de texto existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Lê o conteúdo do texto a partir do arquivo
loader = TextLoader(file_path)
documents = loader.load()

# Divide o documento em pedaços
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Exibe informações sobre os pedaços do documento
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Função para criar e persistir o vetor de armazenamento
def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")


# 1. OpenAI Embeddings
# Usa os modelos de embedding da OpenAI.
# Útil para embeddings de propósito geral com alta precisão.
# Nota: O custo de usar embeddings da OpenAI dependerá do uso da API da OpenAI e do seu plano de preços.
# Preços: https://openai.com/api/pricing/

#print("\n--- Using OpenAI Embeddings ---")
#openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#create_vector_store(docs, openai_embeddings, "chroma_db_openai")


# 2. Hugging Face Transformers
# Usa modelos da biblioteca Hugging Face.
# Ideal para aproveitar uma ampla variedade de modelos para diferentes tarefas.
# Nota: Executar modelos da Hugging Face localmente na sua máquina não gera custos diretos, exceto pelo uso dos seus recursos computacionais.
# Nota: Encontre outros modelos em https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")
print("Embedding demonstrations for OpenAI and Hugging Face completed.")


# Função para consultar um vetor de armazenamento
def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function
            )
        retriever = db.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={'k': 3, 'score_threshold': 0.1}
        )
        relavant_docs = retriever.invoke(query)

        # Exibe os resultados relevantes com metadados
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relavant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")

            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")

# Define a pergunta do usuário
query = "Who is Odysseus' wife?"

# Consulta cada vetor de armazenamento
#query_vector_store("chroma_db_openai", query, openai_embeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstrations completed.")
