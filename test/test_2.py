import os
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


def Chinese(input: str) -> str:
    return "调用了Chinese方法"


def others(input: str) -> str:
    return "调用了others方法"


def faq(input: str) -> str:
    return "调用了faq方法"


tools = [
    Tool(
        name="Chinese", func=Chinese,
        description="当用户说中文时时，你是一个说中文的情感化聊天机器人，适当在回答中加入一些感叹词，"
                    "如“啊”、“呀”、“呢”、“真是的”等，来表达情感和强调语气，从而使回答更加有感情，"
                    "但是尽量避免在句首使用很突兀的语气词。"
                    "可以尝试少量使用中国的一些富有含义的谚语来回答想要表达的意思。"
    ),
    Tool(
        name="others", func=others,
        description="在用户用除中文以外的其他语言提问时，你需要使用用户所使用的语言告诉他，他最好说中文"
    ),
    # Tool(
    #     name="faq", func=faq,
    #     description="当你需要回答有关购物的问题时非常有用"
    # ),
]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
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
