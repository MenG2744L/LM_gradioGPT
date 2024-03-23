# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import SpacyTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader, CSVLoader
import json
import re
import gradio as gr
from langchain_core.tools import tool

ORDER_1 = "20230101ABC"
ORDER_2 = "20230101EFG"
load_dotenv()
os.environ.get("OPENAI_API_KEY")
llm = OpenAI(
    temperature=0
)
ORDER_1_DETAIL = {
    "order_number": ORDER_1,
    "status": "已发货",
    "shipping_date": "2023-01-03",
    "estimated_delivered_date": "2023-01-05",
}

ORDER_2_DETAIL = {
    "order_number": ORDER_2,
    "status": "未发货",
    "shipping_date": None,
    "estimated_delivered_date": None,
}

# --VectorDBQA 让 Tool 支持问答--
loader = TextLoader('E:\python-prj\gradioGPT-main\data.txt', encoding="utf-8")
documents = loader.load()
text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

data_chain = VectorDBQA.from_chain_type(
    llm=llm,
    vectorstore=docsearch,
    verbose=True
)

from langchain.agents import Tool, initialize_agent, AgentType


def search(input: str) -> str:
    return "未找到该订单"


def recommend(input: str) -> str:
    return "红色连衣裙"


@tool("Data")
def data_tool(input: str) -> str:
    """当你要回答问题时你需要在语料库中进行查询，看是否能查询到结果"""
    return data_chain.run(input)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


tools = (
    Tool(
        name="search", func=search,
        description="当你需要回答有关客户订单的问题时非常有用"
    ),
    Tool(
        name="recommend", func=recommend,
        description="在你需要回答有关产品推荐的问题时非常有用"
    ),
    # data_tool,
)


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,
    max_iteration=1,
    verbose=True
)


def chat_call(question: str) -> str:
    result = agent.run(input=question)
    return result


with gr.Blocks() as demo:
    title = gr.HTML("<h1>いらっしゃいます</h1>")
    input = gr.Textbox(label="Input:")
    output = gr.Textbox(label="Answer:")
    btn = gr.Button("Submit")
    btn.click(fn=chat_call, inputs=input, outputs=output)

demo.launch()
