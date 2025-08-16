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
pdf_file_path = ""

if not os.path.exists(pdf_file_path):
    raise FileNotFoundError(f"No PDF file found on the path, please check the path")

pdf_loader = PyPDFLoader(pdf_file_path)
