from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

# INSTANTIATED MEMMORY SAVER TO SAVE CONVERSATIONAL MEMORY
memory = MemorySaver()

# LOAD ENV 
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# GRAPH
# 1. STATE
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. INIT STATEGRAPH
graph_builder = StateGraph(State)

# 3. CREATE TOOL
tool = TavilySearchResults(max_results=2)
tools = [tool]

# 4. DEFINE LLM MODEL
llm = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= api_key,
)

# 5. BIND TOOLS
llm_with_tools = llm.bind_tools(tools)


# 6. CREATE & ADD NODES
    # a. chatbot
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

    # b. tool node
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# 7. ADD CONDITIONAL EDGES FOR ROUTING
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# 8. ADD EDGES TO THE GRAPH
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# 9. COMPILE THE CREATED GRAPH 
# Add our created checkpointer while compiling
graph = graph_builder.compile(checkpointer=memory)


# 9. [OPTIONAL] VISUALIZE GRAPHH

try:
    output_path = 'Memory_graph.png'
    graph_img = graph.get_graph().draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(graph_img)
except Exception:
    # This requires some extra dependencies and is optional
    pass


# MAIN LOOP
# Settin config to maintain parallel threads
config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        # stream_mode="values" #Setting stream mode will make it easier to loop through the event values. 
        # Now we do not need to loop using event.values
    )
    for event in events:
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break