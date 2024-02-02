import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool, load_tools
)

load_dotenv()
os.environ.get("OPENAI_API_KEY")
os.environ.get("SERPAPI_API_KEY")

llm = OpenAI(
    model="gpt-3.5-turbo",
    max_tokens=2048,
    temperature=0.7
)

memory = ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=2048)

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

tools = load_tools(["llm-math", "serpapi"], llm=llm)

agent = initialize_agent(
    tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iteration=2,
    memory=memory
)


def agent_run(input, history=[]):
    history.append(input)
    output = agent.run(input=input)
    history.append(output)
    responses = [(u, b) for u, b in zip(history[::2], history[1::2])]
    return responses, history
