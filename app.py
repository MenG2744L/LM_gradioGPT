import os
import gradio as gr
import openai
from dotenv import load_dotenv
from src import whisper
from src import llm_run

with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
    gr.Markdown("# Welcome to GradioGPT! 🌟🚀")
    gr.Markdown(
        "An easy to use template. It comes with state and settings managment"
    )
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        # 录音功能
        with gr.Row():
            # 得到音频文件地址
            audio = gr.Audio(sources="microphone", type="filepath")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", elem_id="textbox")
    txt.submit(llm_run.conversation_run, [txt, state], [chatbot, state])
    audio.change(whisper.process_audio, [audio, state], [chatbot, state])

if __name__ == "__main__":
    load_dotenv(".env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    demo.launch()
