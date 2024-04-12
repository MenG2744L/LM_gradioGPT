import openai
from src import llm_run

# 使用whisper-1模型进行语音转文字
def transcribe(audio):
    # os.rename(audio, audio + '.wav')
    audio_file = open(audio, "rb")     # 读取语音组件中的用户语音输入
    transcript = openai.Audio.transcribe("whisper-1", audio_file)    # 语音转文字
    return transcript['text']


# 将语音转文字后的文本、角色选择、历史记录输入给大模型
def process_audio(audio, role, history=[]):
    input = transcribe(audio)    # 调用语音转文字方法
    print(input)    # 将转换后的信息显示在控制台
    if input is None:    # 如果未识别到语音中有文字，则自动发送"你好,你是谁？"
        input = "你好,你是谁？"
    return llm_run.conversation_run(input, role, history)   # 调用大模型
