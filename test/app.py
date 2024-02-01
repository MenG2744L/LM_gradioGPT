import os
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool, load_tools
)
import whisper

load_dotenv()
os.environ.get("OPENAI_API_KEY")
os.environ.get("SERPAPI_API_KEY")
llm = OpenAIChat(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.7
)

memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)

default_system_prompt = Path("E:\python-prj\gradioGPT-main\src\prompts\system.prompt").read_text(encoding="utf-8")

# tools = (
#     Tool(
#         name="search", func=search,
#         description="当你需要回答有关客户订单的问题时非常有用"
#     ),
#     Tool(
#         name="recommend", func=recommend,
#         description="在你需要回答有关产品推荐的问题时非常有用"
#     ),
#     # data_tool,
# )

tools = load_tools(["llm-math", "serpapi"], llm=llm)

agent = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iteration=2,
    memory=memory
)


def chatbot_agent(input, history=[]):
    history.append(input)
    output = agent.run(input=input)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history


with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        # 录音功能
        with gr.Row():
            # 得到音频文件地址
            audio = gr.Audio(sources="microphone", type="filepath")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", elem_id="textbox")
        audio.change(whisper.process_audio, [audio, state], [txt])
    txt.submit(chatbot_agent, [txt, state], [chatbot, state])

if __name__ == "__main__":
    demo.launch()
