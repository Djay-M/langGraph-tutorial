import os

from typing import TypedDict, Annotated, Sequence

from langchain_ollama import ChatOllama, OllamaEmbeddings

from operator import add as add_messages

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# setup the LLM model and the embedding
LLM = ChatOllama(model="llama3.1", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Loading PDF files
pdf_file_path = "./logs/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"No PDF file found on the path, please check the path")

pdf_loader = PyPDFLoader(pdf_file_path)


try:
    pages = pdf_loader.load()
except Exception as pdl_loader_error:
    print("Error while loading the PDF file", pdl_loader_error)


# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

pages_split = text_splitter.split_documents(pages)


persist_directory = r"/Users/dj/Documents/Agents/langGraph-tutorial/logs/DB"
collection_name = "pdf_details"

if not os.path.exists(persist_directory):
    os.mkdir(persist_directory)

# Creating the Chroma DB
try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print("Created DB Sucessfully!")
except Exception as db_creation_error:
    print("Error while creating the DB for the PDF file", db_creation_error)

# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # K is the amount of chunks to return
)


@tool
def retriever_tool(query: str) -> str:
    """Tool searches and returns the information from the PDF file."""

    docs = retriever.invoke(query)

    if not docs:
        return f"No data for the query in the docs."

    results = []
    for index, doc in enumerate(docs):
        results.append(f"Document {index + 1}: \n  {doc.page_content}")

    return "\n\n".join(results)


# Binding tools to LLM model
tools = [retriever_tool]
LLM.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> bool:
    """Node to check if the last message has any remaininng tool calls"""

    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about infomation in the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the PDF file. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {tool.name: tool for tool in tools}


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = LLM.invoke(messages)
    return {"messages": [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            HumanMessage(content=user_input)
        ]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


running_agent()
