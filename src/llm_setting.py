from pathlib import Path
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationSummaryBufferMemory


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

    memory = ConversationSummaryBufferMemory(llm=OpenAI(), prompt=SUMMARY_PROMPT, max_token_limit=256)

    file_name = ""
    match role:
        case '无':
            file_name = "None"
        case '默认':
            file_name = "default"
        case '营养师':
            file_name = "dietitian"
        case '游戏人物':
            file_name = "game_characters"

    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\{file_name}.prompt").read_text(encoding="utf-8")

    CHEF_PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=TEMPLATE
    )
    conversation_with_summar = ConversationChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", stop="\n\n", max_tokens=2048, temperature=0.5),
        prompt=CHEF_PROMPT,
        memory=memory,
        verbose=True
    )
    # memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)

    # default_system_prompt = Path("E:\python-prj\gradioGPT-main\src\prompts\system.prompt").read_text(encoding="utf-8")

    # tools = (
    #     Tool(
    #         name="search", func=search,
    #         description="当你需要回答有关客户订单的问题时非常有用"
    #     ),
    #     Tool(
    #         name="recommend", func=recommend,
    #         description="在你需要回答有关产品推荐的问题时非常有用"
    #     ),
    #     # data_tool,
    # )

    # tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    #
    # agent = initialize_agent(
    #     tools,
    #     llm=llm,
    #     agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True,
    #     max_iteration=2,
    #     memory=memory
    # )

    return conversation_with_summar


def story_init_llm(result):

    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\story.prompt").read_text(encoding="utf-8")
    propmt = PromptTemplate(template=TEMPLATE, input_variables=["input"])

    story_llm = LLMChain(
        llm=OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=1),
        prompt=propmt,
        verbose=True
    )

    return story_llm