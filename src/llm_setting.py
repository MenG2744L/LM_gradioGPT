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
        case '心理医师':
            file_name = "psychiatrist"
        case '励志教练':
            file_name = "coach"
        case '音乐推荐':
            file_name = "Music_Recommendation"

    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\{file_name}.prompt").read_text(encoding="utf-8")

    CHEF_PROMPT = PromptTemplate(
        input_variables=["input", "history"],
        template=TEMPLATE
    )

    conversation_with_summar = ConversationChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", stop="\n\n", max_tokens=2048, temperature=0.5),
        prompt=CHEF_PROMPT,
        memory=memory,
        verbose=True
    )

    return conversation_with_summar


def story_init_llm(result):

    TEMPLATE = Path(f"E:\python-prj\gradioGPT-main\src\prompts\img_story.prompt").read_text(encoding="utf-8")
    propmt = PromptTemplate(template=TEMPLATE, input_variables=["input"])

    story_llm = LLMChain(
        llm=OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.8),
        prompt=propmt,
        verbose=True
    )

    return story_llm