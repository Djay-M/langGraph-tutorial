from typing import TypedDict
from langgraph.graph import StateGraph


# Agent State
class AgentState(TypedDict):
    userName: str
    greetingMsg: str


# Node
def greeter_node(state: AgentState) -> AgentState:
    """Simple Node that greets the user"""

    state["greetingMsg"] = "Hey Hi " + state["userName"] + " How are you doing ?"

    return state


# Creating a graph
graph = StateGraph(AgentState)

# Adding a node to graph
graph.add_node("greeter", greeter_node)


# Setting entry point and end point
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

# Compiling the graph
app = graph.compile()

# Creating a image of the graph
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"userName": "DJ"})
print(result["greetingMsg"])
