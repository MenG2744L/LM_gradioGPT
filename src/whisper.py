import openai
from src import llm_run


def transcribe(audio):
    # os.rename(audio, audio + '.wav')
    while True:
        try:
            audio_file = open(audio, "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            break
        except:
            pass
    return transcript['text']


# 录音文件转文本的过程
def process_audio(audio, role, history=[]):
    input = transcribe(audio)
    print(input)
    if input is None:
        input = "你好,你是谁？"
    return llm_run.conversation_run(input, role, history)
