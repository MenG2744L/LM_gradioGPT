import gradio as gr
import llm_agent
import whisper


with gr.Blocks(css="#chatbot{height:800px} .overflow-y-auto{height:800px}") as demo:
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])
    with gr.Row():
        # 录音功能
        with gr.Row():
            # 得到音频文件地址
            audio = gr.Audio(sources="microphone", type="filepath")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", elem_id="textbox")
    txt.submit(llm_agent.agent_run, [txt, state], [chatbot, state])
    audio.change(whisper.process_audio, [audio, state], [chatbot, state])

if __name__ == "__main__":
    demo.launch()
