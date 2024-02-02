import openai
import llm_agent


def transcribe(audio):
    # os.rename(audio, audio + '.wav')
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']


# 录音文件转文本的过程
def process_audio(audio, history=[]):
    input = transcribe(audio)
    print(input)
    if input is None:
        input = "你好"
    return llm_agent.agent_run(input, history)