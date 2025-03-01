from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

from dotenv import load_dotenv

# LOAD ENV 
load_dotenv()

# DEFINE TOOL
# WEB SEARCH TOOL
tool = TavilySearchResults(max_results=2)
# tools = [tool]
# msg = tool.invoke("What's a 'node' in LangGraph?")
# print(msg)

# LLM
llm = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
  # base_url="https://models.inference.ai.azure.com",
  # api_key="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",
)


today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

# BIND TOOLS
# specifying tool_choice will force the model to call this tool.
llm_with_tools = llm.bind_tools([tool])

# DEFINE CHAIN
llm_chain = prompt | llm_with_tools

@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)


response = tool_chain.invoke("What is a node in Langraph")
print(response)