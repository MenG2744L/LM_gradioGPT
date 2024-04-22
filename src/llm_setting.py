from pathlib import Path
from langchain import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory


# LLM的初始化
def init_llm(role):
    SUMMARIZER_TEMPLATE = """请将以下内容逐步概括所提供的对话内容，并将新的概括添加到之前的概括中，形成新的概括。
    EXAMPLE
    Current summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量。
    New lines of conversation:
    Human：为什么你认为人工智能是一种积极的力量？
    AI：因为人工智能将帮助人类发挥他们的潜能。
    New summary:
    Human询问AI对人工智能的看法。AI认为人工智能是一种积极的力量，因为它将帮助人类发挥他们的潜能。
    END OF EXAMPLE
    Current summary:
    {summary}
    New lines of conversation:
    {new_lines}
    New summary:
    """

    SUMMARY_PROMPT = PromptTemplate(
        input_variables=["summary", "new_lines"],
        template=SUMMARIZER_TEMPLATE
    )
    # chain的记忆部分，实现多轮对话
    memory = ConversationSummaryBufferMemory(llm=OpenAI(model_name="gpt-3.5-turbo"), prompt=SUMMARY_PROMPT,
                                             max_token_limit=256)

    file_name = ""  # 定义一个空str，接收用户选择的角色
    match role:
        case '基础GPT':
            file_name = "None"  # 无角色时file_name = "None"
        case '情感机器人':
            file_name = "default"  # 选择默认时file_name = "default"
        case '心理医师':
            file_name = "psychiatrist"  # 选择心理医师时file_name = "psychiatrist"
        case '励志教练':
            file_name = "coach"  # 选择励志教练时file_name = "coach"
        case '音乐推荐':
            file_name = "Music_Recommendation"  # 选择音乐推荐时file_name = "Music_Recommendation"
    # 读取不同prompt
    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\\{file_name}.prompt").read_text(encoding="utf-8")
    # prompt链部分，是conversation_with_summar组成的一部分
    CHEF_PROMPT = PromptTemplate(
        input_variables=["input", "history"],
        template=TEMPLATE
    )
    # ConversationChain将gpt-3.5、prompt、memory连接起来
    conversation_with_summar = ConversationChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", stop="\n\n", max_tokens=2048, temperature=0.5),
        prompt=CHEF_PROMPT,
        memory=memory,
        verbose=True
    )

    return conversation_with_summar  # 返回一个已经初始化并经过prompt微调的gpt-3.5


# 讲故事的语言模型
def story_init_llm(result):
    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\img_story.prompt").read_text(
        encoding="utf-8")  # 读取指定路径下的故事模板文件，并将其内容赋值给TEMPLATE变量
    propmt = PromptTemplate(template=TEMPLATE, input_variables=["input"])  # 创建一个PromptTemplate对象，传入模板内容和需要替换的输入变量名input
    # 创建一个LLMChain对象，用于构建一个包含语言模型的链
    story_llm = LLMChain(
        # 初始化OpenAI语言模型对象，使用"gpt-3.5-turbo"模型，并设置temperature值为0.8
        llm=OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8),

        prompt=propmt,  # 将前面创建的PromptTemplate对象作为prompt传入
        verbose=True  # 设置verbose为True，以便输出详细的调试信息
    )

    return story_llm  # 返回初始化好的LLMChain对象
