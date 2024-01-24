import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType
)
import langchain
from dotenv import load_dotenv
import gradio as gr

langchain.debug = True
# 加载环境变量
load_dotenv()

os.environ.get("OPENAI_API_KEY")

chat = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",
    verbose=True
)
# 连接维基百科
tools = load_tools(['wikipedia'], llm=chat)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 连接各部分
agent = initialize_agent(
    tools,
    chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms",
    memory=memory
)

def call_agent(user_question):
    response = agent.run(input=user_question)
    return response

with gr.Blocks() as demo:
    title = gr.HTML("<h1>欢迎来到Langchain框架的gpt</h1>")
    input = gr.Textbox(label="你想知道什么？")
    output = gr.Textbox(label="这里是答案：")
    btn = gr.Button("获取答案")
    btn.click(fn=call_agent, inputs=input, outputs=output)

demo.launch(debug=True)