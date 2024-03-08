# -*- coding: utf-8 -*-
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader
from langchain_core.tools import tool
from wikipedia import wikipedia


def init_llm():
    llm = OpenAI(
        temperature=0
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

    @tool("Data")
    def data_tool(input: str) -> str:
        """当你要回答问题时你需要在语料库中进行查询，看是否能查询到结果"""
        return data_chain.run(input)

    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    tools.append(

        Tool(data_tool)
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        max_iteration=1,
        verbose=True
    )

    return agent
