from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()


# Define ferramentas
def get_current_time(*args, **kwargs):
    """Retorna a hora atual no formato H:MM AM/PM."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Faz uma busca na Wikipedia e retorna o resumo do primeiro resultado."""
    from wikipedia import summary

    try:
        # Limita a duas sentenças para brevidade
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


# Define as ferramentas que o agente pode usar
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Útil para quando você precisa saber a hora atual.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Útil para quando você precisa saber informações sobre um tópico.",
    ),
]

# Carrega o prompt de chat JSON correto do hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Inicializa um modelo ChatOllama
llm = ChatOllama(model="llama3")

# Cria um Agente de Chat estruturado com Memória de Conversa em Buffer
# ConversationBufferMemory armazena o histórico da conversa, permitindo ao agente manter o contexto entre as interações
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent inicializa um agente de chat projetado para interagir usando um prompt estruturado e ferramentas
# Combina o modelo de linguagem (llm), ferramentas e prompt para criar um agente interativo
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor é responsável por gerenciar a interação entre a entrada do usuário, o agente e as ferramentas
# Também lida com a memória para garantir que o contexto seja mantido ao longo da conversa
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Usa a memória de conversa para manter o contexto
    handle_parsing_errors=True,  # Lida com quaisquer erros de parsing de forma graciosa
)

# Mensagem inicial do sistema para definir o contexto do chat
# SystemMessage é usada para definir uma mensagem do sistema para o agente, estabelecendo instruções iniciais ou contexto
initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Loop de chat para interagir com o usuário
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Adiciona a mensagem do usuário à memória de conversa
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoca o agente com a entrada do usuário e o histórico atual do chat
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Adiciona a resposta do agente à memória de conversa
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
