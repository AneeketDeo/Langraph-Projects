from langchain_community.tools.tavily_search import TavilySearchResults
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

from dotenv import load_dotenv
import os

# LOAD ENV 
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# DEFINE TOOL
# WEB SEARCH TOOL
tool = TavilySearchResults(max_results=2)
# tools = [tool]
# msg = tool.invoke("What's a 'node' in LangGraph?")
# print(msg)

# LLM
llm = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= api_key,

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