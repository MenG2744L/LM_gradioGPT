from src import llm_set
import requests
from src import TTS


def conversation_run(input: str, history=[]):
    conversation = llm_set.init_llm()
    history.append(input)
    output = conversation.predict(input=input)
    audio_output = TTS.run(output)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history, audio_output
