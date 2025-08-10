from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from IPython.display import display, Image

"This is global varible to store the document content"

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """update the LLM content to the global variable 'document_content'"""
    global document_content
    document_content = content
    return f"Document has been updated sucessfully! The current content is: \n {document_content}"


@tool
def save(filename: str) -> str:
    """save the current document_content into a text file.
    Agrs:
        filename: Name of the text file
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n 💾 Document has been saved successfully to {filename}")
        return f"\nDoucment has been saved successfully to {filename}"

    except Exception as error:
        return f"Error while saving the content to text file f{str(error)}"


tools = [update, save]

LLM = ChatOllama(model="llama3.1").bind_tools(tools=tools)


def our_agent(state: AgentState) -> AgentState:
    """Node to get the user input to whether to update the document_content or save it to file"""

    system_prompt = SystemMessage(
        content=f"""
        You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
        
        - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
        - If the user wants to save and finish, you need to use the 'save' tool.
        - Make sure to always show the current document state after modifications.
        
        The current document content is:{document_content}
    """
    )

    if not state["messages"]:
        user_input = input(
            "\n I am ready to help you with the document, What would you like to create ? \n"
        )
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n What would you like to do with the document ? \n")
        print(f"\n 👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_message = [system_prompt] + list(state["messages"]) + [user_message]

    response = LLM.invoke(all_message)

    print(f"\n🤖 AI {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Node to determine if the workflow should continue or end"""

    messages = state["messages"]

    if not messages:
        return "continue"

    # Check if the most recent tool was 'save'
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "save":
            return "end"

    return "continue"


def print_message(messages: list) -> None:
    """Function to print humnan readable messages"""

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n 🛠️ Tool Result: {message.content}")

    return


graph = StateGraph(AgentState)
tool_node = ToolNode(tools)

graph.add_node("agent", our_agent)
graph.add_node("tools", tool_node)


graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app_agent = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER ====== \n")

    state = {"messages": []}

    for step in app_agent.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("\n ===== DRAFTER FINISHED ===== \n")


if __name__ == "__main__":
    run_document_agent()
