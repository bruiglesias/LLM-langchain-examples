import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Carrega variáveis de ambiente do .env
load_dotenv()

# Define o diretório persistente
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define o modelo de embedding
embeddings = OllamaEmbeddings(model='llama3') 

# Carrega o vetor de armazenamento existente com a função de embedding
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Cria um retriever para consultar o vetor de armazenamento
# `search_type` especifica o tipo de busca (por exemplo, similaridade)
# `search_kwargs` contém argumentos adicionais para a busca (por exemplo, número de resultados a retornar)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Cria um modelo ChatOllama
llm = ChatOllama(model='llama3')

# Prompt do sistema para contextualizar a pergunta
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Cria um template de prompt para contextualizar perguntas
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Cria um retriever consciente da história
# Isso usa o LLM para ajudar a reformular a pergunta com base na história do chat
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt do sistema para responder perguntas
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Cria um template de prompt para responder perguntas
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Cria uma cadeia para combinar documentos para question answering
# `create_stuff_documents_chain` alimenta todo o contexto recuperado no LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Cria uma cadeia de recuperação que combina o retriever consciente da história e a cadeia de question answering
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Função para simular um chat contínuo
def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Coleta a história do chat aqui (uma sequência de mensagens)
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        # Processa a consulta do usuário através da cadeia de recuperação
        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Exibe a resposta da IA
        print(f"AI: {result['answer']}")
        # Atualiza a história do chat
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

# Função principal para iniciar o chat contínuo
if __name__ == "__main__":
    continual_chat()
