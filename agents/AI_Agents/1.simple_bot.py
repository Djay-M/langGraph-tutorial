from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os


class AgentState(TypedDict):
    messages: list[HumanMessage]


llm = ChatOllama(model="llama3.1")


def process_message(state: AgentState) -> AgentState:
    """Node to process the human message to LLM"""

    if state["messages"]:
        ai_response = llm.invoke(state["messages"])
        print("\n AI :: ", ai_response.content)

    return state


graph = StateGraph(AgentState)

graph.add_node("process_message", process_message)

graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

agent = graph.compile()

user_input = input("\n Enter :: ")

while user_input not in ["exit", "EXIT", "STOP", "stop", "end", "END"]:
    agent.invoke({"messages": user_input})
    user_input = input("\n Enter :: ")
