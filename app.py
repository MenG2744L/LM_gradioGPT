import os
import gradio as gr
import openai
from dotenv import load_dotenv
from src import whisper, login, llm_run, img_to_text

with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo_Emotion:
    gr.Markdown("# Welcome to Emotion-GPT! 🌟🚀")
    gr.Markdown(
        "An easy to use template. It comes with state and settings managment"
    )
    role = gr.Radio(label="角色设定", choices=["无", "默认", "营养师", "游戏人物"], value="默认")
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", elem_id="textbox")
        # 得到音频文件地址
        audio = gr.Audio(sources="microphone", type="filepath", label="Input Audio")
    img = gr.Image(sources="upload", type="filepath")
    audio_output = gr.Audio(label="Output Audio", autoplay=True)

    img.upload(img_to_text.img2text, [img, state], [chatbot, state, audio_output])
    txt.submit(llm_run.conversation_run, [txt, role, state], [chatbot, state, audio_output])
    audio.change(whisper.process_audio, [audio, role, state], [chatbot, state, audio_output])

with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo_QA:
    gr.Markdown("# Welcome to QA-GPT! 🌟🚀")
    gr.Markdown(
        "An easy to use template. It comes with state and settings managment"
    )
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", elem_id="textbox")
        # 得到音频文件地址
        audio = gr.Audio(sources="microphone", type="filepath", label="Input Audio")
    audio_output = gr.Audio(label="Output Audio", autoplay=True)
    txt.submit(llm_run.conversation_run, [txt, state], [chatbot, state, audio_output])

if __name__ == "__main__":
    # 运行bat文件，输入 %PYTHON% inference_webui.py --model_dir ./logs\OUTPUT_MODEL\G_5900.pth
    load_dotenv(".env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    demo = gr.TabbedInterface(
        [demo_Emotion, demo_QA],
        tab_names=["情感化聊天机器人", "知识问答"],
        title="demo"
    )
    demo.launch(share=False, inbrowser=True)  # , auth=login.open_login_window
