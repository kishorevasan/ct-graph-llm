from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id

from tools.vector import get_ct_description
from tools.cypher import cypher_ct_chain

# Create a movie chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clinical trials expert providing information about clinical trials. Use the graph database and vector search to ground your responses in real world data."),
        ("human", "{input}"),
    ]
)

ct_chat = chat_prompt | llm | StrOutputParser()


# Create a set of tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general clinical trials chat not covered by other tools",
        func=ct_chat.invoke,
    ),
    Tool.from_function(
        name = "Document embedding search",
        description = "For when you need to search for clinical trials based on a given query",
        func=get_ct_description,
    ),
    Tool.from_function(
        name = 'Clinical trials knowledge graph',
        description = 'Answer clinical trials related questions using real-world accurate data. Done using Knowledge graphs and Cypher queries',
        func = cypher_ct_chain
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']
