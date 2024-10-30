# Docs: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

# Importa as bibliotecas necessárias
import os
from typing import Type
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI


load_dotenv()

# Modelos Pydantic para argumentos das ferramentas

class SimpleSearchInput(BaseModel):
    query: str = Field(description="deve ser uma consulta de pesquisa")


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="Primeiro número a multiplicar")
    y: float = Field(description="Segundo número a multiplicar")


# Ferramenta personalizada com apenas entrada customizada
class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "útil para responder perguntas sobre eventos atuais"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Usa a ferramenta."""
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return f"Search results for: {query}\n\n\n{results}\n"


# Ferramenta personalizada com entrada e saída customizadas
class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "útil para multiplicar dois números"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, x: float, y: float) -> str:
        """Usa a ferramenta."""
        result = x * y
        return f"The product of {x} and {y} is {result}"
    
# Cria ferramentas usando a abordagem de subclasse do Pydantic
tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

# Inicializa um modelo ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# Puxa o template de prompt do hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Cria o agente ReAct usando a função create_tool_calling_agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Cria o executor do agente
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# Testa o agente com consultas de exemplo
response = agent_executor.invoke({"input": "Search for Apple Intelligence"})
print("Response for 'Search for LangChain updates':", response)

response = agent_executor.invoke({"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", response)
