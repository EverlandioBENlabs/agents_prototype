# streamlit run self_rag.py args --server.fileWatcherType none --server.address localhost

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.output_parsers.openai_tools import (
    PydanticToolsParser,
)
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langsmith import traceable
from langchain_core.tools import tool
from typing import List
import json
from langgraph.graph import END, MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt.tool_executor import ToolInvocation, ToolExecutor
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from collections import defaultdict
from langchain_core.tools import tool
from typing import List
import os

OPENAI_MODEL = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4-turbo-preview"]
)

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

embedding_function = OpenAIEmbeddings()

loader = CSVLoader("./raw_data.csv", encoding="windows-1252")
documents = loader.load()

db = Chroma.from_documents(documents, embedding_function)
retriever = db.as_retriever()

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# LLM with function call 
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)