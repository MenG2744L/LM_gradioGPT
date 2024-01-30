# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.tools import Tool
import gradio as gr



load_dotenv()

os.environ.get("OPENAI_API_KEY")
llm = OpenAI(
    temperature=0.7
)

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

from langchain.agents import tool, initialize_agent, AgentType


def search(input: str) -> str:
    return "未找到该订单"


def recommend(input: str) -> str:
    return "红色连衣裙"


@tool("Data")
def data_tool(input: str) -> str:
    """当你需要回答一些电商的问题时非常有用"""
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
    data_tool
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True,
    # max_iteration=2,
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
