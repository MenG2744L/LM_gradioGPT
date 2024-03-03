from src import llm_set


def conversation_run(input: str, history=[]):
    conversation = llm_set.init_llm()
    history.append(input)
    output = conversation.predict(input=input)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history
