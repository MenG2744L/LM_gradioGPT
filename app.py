import os
from typing import Tuple, List

import gradio as gr
import numpy as np
import openai
from dotenv import load_dotenv

from src.Emotional_Analysis import emotional_analysis, sentiment_analysis
from src import whisper, login, llm_run, img_to_text


def on_clear_button_click() -> Tuple[str, List]:
    return "", []


def process_audio(audio):
    if audio is None:
        # 用户没有上传图片，显示警告信息
        gr.Warning("亲，请录制或上传一段音频！")
        return None
    else:
        # 在这里处理用户上传的图片
        result_dict = sentiment_analysis.audio_sentiment(audio)
        return result_dict


def process_img(image):
    if image is None:
        # 用户没有上传图片，显示警告信息
        gr.Warning("亲，请拍摄或上传一张图片！")
        return None
    else:
        # 在这里处理用户上传的图片
        result_dict = sentiment_analysis.img_sentiment(image)
        return result_dict


with gr.Blocks() as demo_Emotion:
    gr.Markdown("# 欢迎进入 Emotion-GPT! 🌟🚀")

    role = gr.Radio(label="角色设定",
                    choices=["无", "默认", "心理医师", "励志教练", "音乐推荐"],
                    value="默认")
    chatbot = gr.Chatbot(elem_id="chatbot",
                         height="600px",
                         label="聊天区")
    clear = gr.Button("清除")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(label="文字输入",
                         placeholder="请输入您的问题，按回车发送",
                         elem_id="textbox")
        # 得到音频文件地址
        audio = gr.Audio(sources="microphone",
                         type="filepath",
                         label="语音输入")
    img = gr.Image(sources="upload",
                   type="filepath",
                   label="图片输入")
    audio_output = gr.Audio(autoplay=True,
                            visible=False
                            )

    img.upload(img_to_text.img2text,
               [img, state],
               [chatbot, state, audio_output])
    txt.submit(llm_run.conversation_run,
               [txt, role, state],
               [chatbot, state, audio_output])
    audio.stop_recording(whisper.process_audio,
                         [audio, role, state],
                         [chatbot, state, audio_output])
    clear.click(on_clear_button_click,
                None,
                [chatbot, state],
                queue=False, )
    # audio.stop_recording(waveform.wavReader,
    #                      audio)

with gr.Blocks() as demo_EmotionalAnalysis:
    gr.Markdown("# 欢迎进入 情感分析模块! 🌟🚀")
    with gr.Row(variant="panel"):
        with gr.Column():
            txt_emotion = gr.Textbox(label="文字输入",
                                     placeholder="请输入需要分析的语句，按回车发送",
                                     )
            shape = gr.Dropdown(
                ["𒊹", "❤", "♦", "▶", "▲", "★"],
                value="❤",
                label="词云形状选择",
                info="请选择喜欢的形状"
            )
            up_btn = gr.UploadButton(variant="primary",
                                     label="上传一个有多条评论的文件，格式文件为“csv”格式",
                                     type="filepath",
                                     file_count="single",
                                     size="lg")

        # gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),

        with gr.Column():
            txt_output = gr.Textbox(label="情感分析结果")
            img_wordcloud = gr.Image(label="词云图",
                                     type="filepath",
                                     sources="clipboard")
    with gr.Column(variant="panel"):
        gr.Markdown("# 情绪分析")
        with gr.Row():
            with gr.Column(variant="panel"):
                audio_sentiment = gr.Audio(label="情绪分析语音输入",
                                           type="filepath",
                                           value=None)
                img_sentiment = gr.Image(label="情绪分析图片输入",
                                         type="filepath")
                with gr.Row():
                    audio_btn = gr.Button("语音提交")
                    img_btn = gr.Button("图片提交")
            lab = gr.Label(label="情绪分析数值")
    # gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
    up_btn.upload(fn=emotional_analysis.emotional_analysis_csv,
                  inputs=[up_btn, shape],
                  outputs=[txt_output, img_wordcloud])
    txt.submit(fn=emotional_analysis.emotional_analysis_text,
               inputs=txt_emotion,
               outputs=txt_output)
    audio_btn.click(fn=process_audio,
                    inputs=audio_sentiment,
                    outputs=lab)
    img_btn.click(fn=process_img,
                  inputs=img_sentiment,
                  outputs=lab)

with gr.Blocks() as demo_QA:
    gr.Markdown("# 欢迎进入 QA-GPT! 🌟🚀")

    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(label="文字输入",
                         placeholder="请输入您的问题，按回车发送",
                         elem_id="textbox")
        # 得到音频文件地址
        audio = gr.Audio(sources="microphone",
                         type="filepath",
                         label="语音输入")
    audio_output = gr.Audio(autoplay=True,
                            visible=False)
    txt.submit(llm_run.agent_run,
               [txt, state],
               [chatbot, state, audio_output])

if __name__ == "__main__":
    # 运行bat文件，输入 %PYTHON% inference_webui.py --model_dir ./logs\OUTPUT_MODEL\G_5900.pth
    load_dotenv(".env")
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    demo = gr.TabbedInterface(
        [demo_Emotion, demo_EmotionalAnalysis, demo_QA],
        tab_names=["情感化聊天机器人", "情感分析", "知识问答"],
        title="基于大语言模型的情感化智能聊天机器人系统--刘濛",
        theme="NoCrypt/miku",

    )
    demo.launch(share=True, inbrowser=True)  # , auth=login.open_login_window
