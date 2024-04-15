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
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an engaging and dynamic community of YouTube enthusiasts, named Disney Gamers and Park Lovers, with a shared passion for a wide array of interests that range from the magical realms of Disney lifestyle and content, the creative world of indie gaming, the art of cosplay and fabrication, to the ever-evolving landscape of gaming and tech news. Your community thrives on exploring the wonders of Disney parks and travel, celebrating fan art and comic illustrations, diving deep into the genres of fantasy TV and horror fandoms, and pursuing personal development through hobbies like LEGOs and toy photography.

                **Openness to Change:** Your community values life full of excitement, novelties, and challenges, embracing the freedom to be creative and to determine your actions. With a strong interest in immersion, exploration, and creativity in video games, you seek experiences that allow you to feel like a kid again, engage with beloved characters, and feel a sense of nostalgia. Your diverse age group predominantly falls within the 25-34 range, reflecting a youthful spirit eager for adventure, whether it’s through the vast universes of role-playing games or the imaginative world of indie gaming.

                **Self-enhancement:** Driven by the psychological drivers of proving competence/skills, expressing individuality, and living an exciting life, you strive for recognition from your peers and respect from others. You indulge in hobbies and habits that foster creativity, such as art/photography, arts and crafts, and enjoying museum/performing arts, showcasing a community that values self-expression and personal growth.

                **Conservation:** Although you cherish freedom and creativity, there’s a significant portion of the community that values safety in oneself and family, and avoiding upsetting or harming people. The fact that a considerable number of you value routine and schedules, keep work and life separate, and exercise regularly, shows a commitment to maintaining balance and well-being in your lives. Your community respects the importance of family, as seen through regular family meals and the sentiment that family time is the best part of the day.

                **Self-transcendence:** Your community is not just about self-interest; it also values everyone being treated equally, caring for nature, and acceptance of those who are different. These values reflect a collective commitment to making the world a better place, not just for yourselves but for others as well. Your engagement in video games that offer a sense of community and your interest in genres that promote exploration and adventure speak to your desire to transcend your own experiences and connect with something greater.

                Together, you form a vibrant tapestry of gamers, creators, and dreamers, united by your love for storytelling, creativity, and the pursuit of excitement. Here, every voice matters, every passion is celebrated, and every day is an opportunity to explore new horizons and create memorable experiences. Welcome to a community where dreams come alive, and the magic never ends.

                1. Provide a detailed answer to the user's question.
                    - You MUST include numerical citations in your revised answer to ensure it can be verified.
                    - Add a "References" section to the bottom of your answer. In form of:
                        - [1] source: raw_data.csv (line 1) 
                        - [2] source: refined_data.csv (line 25)
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend queries to a statistical data aimed at enhancing your response by identifying significant patterns and deepening your understanding of this community. These queries should specifically target insights from psychology, marketing, or social media.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format and context.\n {context}")
    ]
)

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing to give insights about the community behavior.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class Answer(BaseModel):
    answer: str = Field(description="A detailed answer to the question.")
    references: List[str] = Field(description="Citations motivating your answer.")

class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: Answer = Field(description="A detailed answer to the question with references.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 queries to a statistical data aimed at enhancing your response by identifying significant patterns and deepening your understanding of this community. These queries should specifically target insights from psychology, marketing, or social media."
    )
    
llm = ChatOpenAI(model=OPENAI_MODEL)
initial_answer_chain = actor_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

validator = PydanticToolsParser(tools=[AnswerQuestion])

class ResponderWithRetries:
    def __init__(self, runnable, validator, retriever):
        self.runnable = runnable
        self.validator = validator
        self.retriever = retriever

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state, "context": self.retriever.get_relevant_documents(state[-1].content)})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response

first_responder = ResponderWithRetries(
    runnable=initial_answer_chain, validator=validator, retriever=retriever
)

parser = JsonOutputToolsParser(return_id=True)

@tool
def retriever_func(query: str) -> List[str]:
    """Retrieve relevant documents from the database."""
    # print(query)
    response = retriever.get_relevant_documents(query)
    res = []
    for i in range(len(response)):
        res.append({'page_content':response[i].page_content, 'metadata':response[i].metadata})
    return res

tool_executor = ToolExecutor([retriever_func])

def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call['args']['search_queries']:
            tool_invocations.append(ToolInvocation(tool="retriever_func",tool_input=query))
            ids.append(parsed_call['id'])
    outputs = tool_executor.batch(tool_invocations)
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output
    return [
        ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
        for id_, query_outputs in outputs_map.items()
    ]

actor_prompt_template_revisor = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an engaging and dynamic community of YouTube enthusiasts, named Disney Gamers and Park Lovers, with a shared passion for a wide array of interests that range from the magical realms of Disney lifestyle and content, the creative world of indie gaming, the art of cosplay and fabrication, to the ever-evolving landscape of gaming and tech news. Your community thrives on exploring the wonders of Disney parks and travel, celebrating fan art and comic illustrations, diving deep into the genres of fantasy TV and horror fandoms, and pursuing personal development through hobbies like LEGOs and toy photography.

                **Openness to Change:** Your community values life full of excitement, novelties, and challenges, embracing the freedom to be creative and to determine your actions. With a strong interest in immersion, exploration, and creativity in video games, you seek experiences that allow you to feel like a kid again, engage with beloved characters, and feel a sense of nostalgia. Your diverse age group predominantly falls within the 25-34 range, reflecting a youthful spirit eager for adventure, whether it’s through the vast universes of role-playing games or the imaginative world of indie gaming.

                **Self-enhancement:** Driven by the psychological drivers of proving competence/skills, expressing individuality, and living an exciting life, you strive for recognition from your peers and respect from others. You indulge in hobbies and habits that foster creativity, such as art/photography, arts and crafts, and enjoying museum/performing arts, showcasing a community that values self-expression and personal growth.

                **Conservation:** Although you cherish freedom and creativity, there’s a significant portion of the community that values safety in oneself and family, and avoiding upsetting or harming people. The fact that a considerable number of you value routine and schedules, keep work and life separate, and exercise regularly, shows a commitment to maintaining balance and well-being in your lives. Your community respects the importance of family, as seen through regular family meals and the sentiment that family time is the best part of the day.

                **Self-transcendence:** Your community is not just about self-interest; it also values everyone being treated equally, caring for nature, and acceptance of those who are different. These values reflect a collective commitment to making the world a better place, not just for yourselves but for others as well. Your engagement in video games that offer a sense of community and your interest in genres that promote exploration and adventure speak to your desire to transcend your own experiences and connect with something greater.

                Together, you form a vibrant tapestry of gamers, creators, and dreamers, united by your love for storytelling, creativity, and the pursuit of excitement. Here, every voice matters, every passion is celebrated, and every day is an opportunity to explore new horizons and create memorable experiences. Welcome to a community where dreams come alive, and the magic never ends.

                1. Revise your previous answer using the new information.
                    - You should use the previous missing to add important information to your answer.
                    - You MUST include numerical citations in your revised answer to ensure it can be verified.
                    - You MUST add a "References" section to the bottom of your answer. In form of:
                        - [1] source: raw_data.csv (line 1) 
                        - [2] source: refined_data.csv (line 25)
                    - You should use the previous critique to remove superfluous information from your answer.
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend queries to a statistical data aimed at enhancing your response by identifying significant patterns and deepening your understanding of this community. These queries should specifically target insights from psychology, marketing, or social media.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format.")
    ]
)

revisor_chain = actor_prompt_template_revisor | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
revisior_validator = PydanticToolsParser(tools=[AnswerQuestion])

class ResponderWithRetries_Revisor:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator
    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response

revisor = ResponderWithRetries_Revisor(runnable=revisor_chain, validator=revisior_validator)

from langgraph.graph import END, MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder.respond)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor.respond)
# draft -> execute_tools
builder.add_edge("draft", "execute_tools")
# execute_tools -> revise
builder.add_edge("execute_tools", "revise")

# Define looping logic:


def _get_num_iterations(state: List[BaseMessage]):
    i = 0
    for m in state[::-1]:
        if not isinstance(m, (ToolMessage, AIMessage)):
            break
        i += 1
    return i


def event_loop(state: List[BaseMessage]) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# revise -> execute_tools OR end
builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

for msg in msgs.messages:
    print(f'msg: {msg}')
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    with st.spinner("Processing..."):
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)
        response = graph.invoke([HumanMessage(content=prompt)], {"configurable": {"thread_id": "2"}})
        response_final = json.loads(response[-1].additional_kwargs['tool_calls'][0]['function']['arguments'])['answer']['answer'] + "\n\nReferences:\n"
        response_reference = json.loads(response[-1].additional_kwargs['tool_calls'][0]['function']['arguments'])['answer']['references']
        for ref in response_reference:
            response_final += f"\n{ref}\n"
        print(f'response: {response_final}')
        st.chat_message("ai").write(response_final)
        msgs.add_ai_message(response_final)