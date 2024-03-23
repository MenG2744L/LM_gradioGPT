from src import llm_setting, agent_setting
import requests
from src import TTS


def conversation_run(input: str, role: str, history=[]):
    conversation = llm_setting.init_llm(role)
    history.append(input)
    output = conversation.predict(input=input)
    audio_output = TTS.run(output)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history, audio_output


def story_run(input: str, history=[]):
    story = llm_setting.story_init_llm(input)
    history.append("根据图片生成故事")
    output = story.predict(input=input)
    audio_output = TTS.run(output)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history, audio_output


def agent_run(input: str, history=[]):
    agent = agent_setting.init_agent()
    history.append(input)
    output = agent.run(input=input)
    audio_output = TTS.run(output)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history, audio_output
