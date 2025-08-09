from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
import os

LLM = ChatOllama(model="llama3.1", temperature=0.8)

class AgentState(TypedDict):
    messages: List[Union[AIMessage, HumanMessage]]

def process_message(state: AgentState) -> AgentState:
    """Node to process the messages in the state"""

    ai_response = LLM.invoke(state["messages"])

    state["messages"].append(AIMessage(content=ai_response.content))

    print("\n AI: ", ai_response.content)

    return state

graph = StateGraph(AgentState)

graph.add_node("process_message", process_message)

graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

agent = graph.compile()

conversation = []

user_input = input("\n You: ")

while user_input.lower() not in ["bye", "exit", "done", "close"]:
    conversation.append(HumanMessage(content=user_input))
    agentResponse = agent.invoke({ "messages": conversation })
    conversation = agentResponse["messages"]
    user_input = input("\n You: ")

with open("./logs/conversation.txt", "w") as file:
    file.write("Conversation logs: \n")

    for message in conversation:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content} \n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content} \n")
    
    file.write("End of conversation \n")

print("conversation saved to logs file")

