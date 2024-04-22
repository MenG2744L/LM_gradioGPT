from src import llm_setting, agent_setting, mongo, TTS


def conversation_run(input: str, role: str, history=[]):
    conversation = llm_setting.init_llm(role)

    history.append(input)
    output = conversation.predict(input=input)
    mongo.insert_mongo(role, input, output)
    audio_output = TTS.run(output)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history, audio_output


def story_run(input: str, history=[]):
    # 调用llm_setting模块的story_init_llm函数，传入用户输入的字符串input，用于初始化一个讲故事机器人，并返回该机器人对象
    story = llm_setting.story_init_llm(input)
    history.append("根据图片生成故事")
    output = story.predict(input=input)   # 调用story对象的predict方法，传入用户输入的字符串input，获取机器人根据输入生成的故事
    audio_output = TTS.run(output)    # 文字转语音
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]    # 将历史记录history中的元素两两配对，生成一个包含用户输入和机器人响应的元组列表
    return responses, history, audio_output   # 返回结果、历史记录和音频


def agent_run(input: str, history=[]):
    agent = agent_setting.init_agent()
    history.append(input)
    output = agent.run(input=input)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history
