import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
import gradio as gr
from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool
)

load_dotenv()

os.environ.get("OPENAI_API_KEY")
llm = OpenAIChat(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.7
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

default_system_prompt = Path("E:\python-prj\gradioGPT-main\src\prompts\system.prompt").read_text(encoding="utf-8")

def search(input: str) -> str:
    return "调用了search方法"


def recommend(input: str) -> str:
    return "红色连衣裙"


def faq(input: str) -> str:
    return "7天无理由退货"


tools = [
    Tool(
        name="search", func=search,
        description="当你需要回答有关客户订单的问题时非常有用"
    ),
    Tool(
        name="recommend", func=recommend,
        description="在你需要回答有关产品推荐的问题时非常有用"
    ),
    Tool(
        name="faq", func=faq,
        description="当你需要回答有关购物的问题时非常有用"
    ),
]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iteration=2,
    memory=memory
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
