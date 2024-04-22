# -*- coding: utf-8 -*-
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import OpenAI, VectorDBQA
from langchain_community.document_loaders import TextLoader

from langchain_core.tools import tool, BaseTool
from wikipedia import wikipedia


# 定义一个初始化智能体的函数
def init_agent():
    # 初始化一个OpenAI对象，使用gpt-3.5-turbo模型，并设置temperature为0，确保输出更加确定
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )
    # 创建一个TextLoader对象来加载文档
    loader = TextLoader('E:\python-prj\gradioGPT-main\\assets\data.txt', encoding="utf-8")
    documents = loader.load()  # 使用TextLoader对象加载文档内容
    # 创建一个SpacyTextSplitter对象，使用中文模型将文档拆分，这里设置每行上限为256
    text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()  # 初始化一个OpenAIEmbeddings对象，用于将文本转换为向量
    docsearch = FAISS.from_documents(texts, embeddings)  # 使用FAISS库，根据文本和嵌入向量创建一个文档搜索引擎
    # 使用向量数据库QA工具链，从链类型创建一个VectorDBQA对象，用于从语料库中查询答案
    data_chain = VectorDBQA.from_chain_type(
        llm=llm,
        vectorstore=docsearch,
        verbose=True  # 显示处理信息流
    )

    # 定义一个函数data_tool，它接受一个字符串输入，并使用data_chain运行查询，返回结果
    def data_tool(input: str) -> str:
        return data_chain.run(input)

    tools = load_tools(['wikipedia'], llm=llm)   # 加载内置工具wikipedia
    # 向tools列表中添加一个自定义工具，名为"data"，执行方法为data_tool，并设置描述
    tools.append(
        Tool(name="data",
             func=data_tool,
             description="""当你要回答问题时你需要在语料库中进行查询，看是否能查询到结果""")
    )
    # 初始化一个ConversationBufferMemory对象，用于存储对话历史，并设置返回消息
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # 初始化一个智能体对象
    agent = initialize_agent(
        tools=tools,    # 传入工具tools
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,    # 指定agent类型
        memory=memory,
        handle_parsing_errors=True,   #处理解析错误
        max_iteration=3,    # 设置agent最多执行3次思考
        verbose=True
    )

    return agent
