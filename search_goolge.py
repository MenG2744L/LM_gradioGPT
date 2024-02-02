import os
from dotenv import load_dotenv
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ.get("OPENAI_API_KEY")


template = """
在>>>和<<<之间的，是直接来自Google的搜索结果。
请把对于问题'{query}'的答案从里面直接提取出来，如果里面灭有相关信息的话就回答'找不到'
请使用如下格式：
Extracted：<answer or "找不到">
>>>{requests_result}<<<
Extracted：
"""

sys_prompt = PromptTemplate(
    template=template,
    input_variables=["query", "requests_result"]
)

request_chain = LLMRequestsChain(llm_chain=LLMChain(llm=OpenAI(temperature=0.5), prompt=sys_prompt))

question = "今天河北省廊坊市天气怎么样？"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question
}
result = request_chain.run(inputs)

print(result)
