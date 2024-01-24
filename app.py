# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from queue import Empty, Queue
from threading import Thread
import gradio as gr
from langchain.agents import (
    load_tools,
    initialize_agent,
    AgentType
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from callback import QueueCallback

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# 加载环境变量
load_dotenv()
# 设置可选择的模型
MODELS_NAMES = ["gpt-3.5-turbo", "gpt-4"]
DEFAULT_TEMPERATURE = 0.7
ChatHistory = List[str]
logging.basicConfig(
    format="[%(asctime)s %(levelname)s]: %(message)s", level=logging.INFO
)
human_message_prompt_template = HumanMessagePromptTemplate.from_template("{text}")
# 加载系统提示
default_system_prompt = Path("E:\python-prj\gradioGPT-main\src\prompts\system.prompt").read_text(encoding="utf-8")


# 创建聊天机器人界面，并保存聊天记录
def on_message_button_click(
        chat: Optional[ChatOpenAI],
        message: str,
        # 用户输入信息
        chatbot_messages: ChatHistory,
        # 包含AIMessage、HumanMessage和SystemMessage
        messages: List[BaseMessage],
) -> Tuple[ChatOpenAI, str, ChatHistory, List[BaseMessage]]:
    if chat is None:
        # 设置环境变量
        os.environ.get("OPENAI_API_KEY")
        queue = Queue()
        # 创建聊天机器人
        chat = ChatOpenAI(
            model_name=MODELS_NAMES[0],
            temperature=DEFAULT_TEMPERATURE,
            streaming=True,
            callbacks=([QueueCallback(queue)]),
        )

    else:
        # 取出列表中第一个
        queue = chat.callbacks[0].queue

    job_done = object()

    logging.info(f"正在进行提问, 问题-->{message}")
    # 将用户输入的信息也一并加入到记录中
    messages.append(HumanMessage(content=message))
    chatbot_messages.append((message, ""))


    # 连接维基百科
    tools = load_tools(['wikipedia'], llm=chat)

    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms",
    )
    # print(messages)

    agent.run(input=[message, chatbot_messages])


    # 执行聊天机器人的生成任务，并将结果放入队列
    # def task():
    #     chat(messages)
    #     queue.put(job_done)
    #
    # # 创建一个线程并在其中运行生成任务
    # t = Thread(target=task)
    # t.start()
    # 用于保存生成的结果
    content = ""
    # 读取队列中最后一个生成结果
    while True:
        try:
            next_token = queue.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            chatbot_messages[-1] = (message, content)
            yield chat, "", chatbot_messages, messages
        except Empty:
            continue
    # 将结果加入到记录中
    messages.append(AIMessage(content=content))
    logging.debug(f"reply = {content}")
    logging.info(f"Done!")

    print(chatbot_messages, messages)

    return chat, "", chatbot_messages, messages


# 处理系统提示的更改
def system_prompt_handler(value: str) -> str:
    return value


# 清除聊天记录
def on_clear_button_click(system_prompt: str) -> Tuple[str, List, List]:
    return "", [], [SystemMessage(content=system_prompt)]


# 应用设置，包括模型model_name和temperature
def on_apply_settings_button_click(
        system_prompt: str, model_name: str, temperature: float
):
    logging.info(
        f"当前设置: model_name={model_name}, temperature={temperature}"
    )
    chat = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        streaming=True,
        callbacks=[QueueCallback(Queue())],
    )
    # 清除队列queue
    chat.callbacks[0].queue.empty()
    return chat, *on_clear_button_click(system_prompt)


# 前端界面，对方法进行封装
with gr.Blocks(
        css="""
        #col_container {
        width: 700px;
        margin-left: auto;
        margin-right: auto;
        }
        #chatbot {
        height: 400px;
        overflow: auto;
        background-color: white;
        }"""
) as demo:
    # 设置初始状态
    system_prompt = gr.State(default_system_prompt)
    messages = gr.State([SystemMessage(content=default_system_prompt)])
    # 设置chat初始状态为空
    chat = gr.State(None)

    with gr.Column(elem_id="col_container"):
        # 标题
        gr.Markdown("# 欢迎进入刘濛のGPT! 🌟🚀")
        gr.Markdown(
            "——可自定义情感的智能聊天机器人"
        )
        with gr.Column():
            system_prompt_area = gr.TextArea(
                default_system_prompt, lines=4, label="机器人情感设置（可根据以下内容自行修改机器人情感）",
                interactive=True
            )
            # 获取系统提示词，并可输入
            system_prompt_area.input(
                system_prompt_handler,
                inputs=[system_prompt_area],
                outputs=[system_prompt],
            )
            system_prompt_button = gr.Button("写入")

        chatbot = gr.Chatbot(label="记录")
        with gr.Column():
            message = gr.Textbox(label="输入")
            message.submit(
                on_message_button_click,
                inputs=[chat, message, chatbot, messages],
                outputs=[chat, message, chatbot, messages],
                queue=True,
            )
            message_button = gr.Button("提交", variant="primary")
            message_button.click(
                on_message_button_click,
                inputs=[chat, message, chatbot, messages],
                outputs=[chat, message, chatbot, messages],
            )
        with gr.Row():
            with gr.Column():
                clear_button = gr.Button("清空")
                clear_button.click(
                    on_clear_button_click,
                    inputs=[system_prompt],
                    outputs=[message, chatbot, messages],
                    queue=False,
                )
            with gr.Accordion("自定义", open=False):
                model_name = gr.Dropdown(
                    choices=MODELS_NAMES, value=MODELS_NAMES[0], label="model"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="temperature",
                    interactive=True,
                )
                apply_settings_button = gr.Button("应用")
                apply_settings_button.click(
                    on_apply_settings_button_click,
                    [system_prompt, model_name, temperature],
                    [chat, message, chatbot, messages],
                )

        system_prompt_button.click(
            on_apply_settings_button_click,
            [system_prompt, model_name, temperature],
            [chat, message, chatbot, messages],
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
