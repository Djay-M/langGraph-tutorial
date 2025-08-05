"""Dict
Basic a object or a JSON
movie = { name: "KGF", year: 2020}
"""

""" TypeDict
It is type of JSON, but the key's data is defined.
"""
from typing import TypedDict


class Movie(TypedDict):
    name: str
    year: int


kannadaMovie = Movie(name="KGF", year=2020)
print(kannadaMovie)

""" Type Annotations

    1. Union: meaning a function parameter can be of different data types.
    2. Optional: paramter can be of any of the given types.
    3. Any: parameter can be any type.
    4. Lambda functions.
"""

""" Elements

    1. State:
        It stores all the variables and the program data,
        basically like a white board in a meeting room, every descussion is rewritten to borad.
    
    2. Nodes:
        It is individual function that perfrom a specific task.
    
    3. Graph:
        It is a overaching structure that maps out how different nodes are connected and excuted.
        It is Visual representation of the workflow, like a roadmap connecting the nodes.
    
    4. Edges:
        It is connection between 2 nodes, it just which node to excute after the current one.
        Like a train track, where it connects to one node to another node.
    
    5. Condidtional Edges:
        It is specialized connections that decide the next node based on the condition of the STATE,
        It is like a traffic light, green -> go, yellow -> wait and red -> stop.
        Like a If-Else condition on the edge.

    6. Start Point:
        It is the virtual entry point, Where the train begins.
    
    7.End Point:
        It is the virtual end point, Where the train stops.
    
    8. Tools:
        Tools are specialized functions or utilities that Node can utilize to make a API call.
        They enhance Node's capabilities by providing additional functionalities.
    
    9. ToolNode:
        It is special kind of Node which runs a tool.
    
    10. StateGraph:
        It is class in LangGraph used to build and compile the graph structure.
        It manages all the node, edges and state, ensuring that the workflow operates in a unified way and
        the data flows correctly between components. Like Blue Print of a building.
    
    11. Runnable:
        It is standrdized excutable component that perform a specific task within a AI workflow.
        Diff: Node vs Runnable
            Node gets a state performs a task and updates the state
            Runnable is like lego brick, combined to perform complex AI workflows.
        
    12. Messages:
        1. Human Message: Given by the user.
        2. AI Message: Response genrated by AI models.
        3. System Message: Used to provide instruction or context to AI model.
        4. Function Message: Respresents the result of a function.
        5. Tool Message: Similar to function message, but for the tool.

(Basically, a train track is the EDGES and station are the NODES and STATE is a train, and overall map this is GRAPH)
"""
