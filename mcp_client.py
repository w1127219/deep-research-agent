import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict

# 引入 MCP 客户端连接组件
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 注意：因为 MCP 本质是异步的，所以这里要用 async
async def main():
    print("🔌 正在连接 MCP 服务端并加载工具...")
    
    # 1. 告诉客户端如何启动刚才写的 mcp_server.py
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"],
        env=os.environ.copy()
    )

    # 2. 建立连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 3. 把 MCP 里的工具，瞬间转换成大模型能用的 LangChain Tool 格式！
            tools = await load_mcp_tools(session)
            print(f"✅ 成功加载 {len(tools)} 个 MCP 工具！")
            for t in tools:
                print(f"  🔧 发现工具: {t.name}")

            # 4. 构建模型和图
            llm = ChatOpenAI(model="deepseek-chat", temperature=0)
            llm_with_tools = llm.bind_tools(tools)

            def chatbot(state: AgentState):
                messages = state["messages"]
                if not any(isinstance(m, SystemMessage) for m in messages):
                    sys_msg = SystemMessage(content="你是一个有读取本地文件权限的智能助手。当用户让你读取文件时，请调用相关工具。")
                    messages = [sys_msg] + messages
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}

            def route_after_chatbot(state: AgentState):
                if state["messages"][-1].tool_calls:
                    return "tools"
                return END

            builder = StateGraph(AgentState)
            builder.add_node("chatbot", chatbot)
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "chatbot")
            builder.add_conditional_edges("chatbot", route_after_chatbot, {"tools": "tools", END: END})
            builder.add_edge("tools", "chatbot")
            
            app = builder.compile()

            # 5. 测试运行
            print("\n🚀 MCP 智能体已启动！")
            # 为了测试，先在本地生成一个测试文件
            with open("mcp_secret.txt", "w", encoding="utf-8") as f:
                f.write("恭喜你！如果你看到了这段话，说明大模型真的通过 MCP 协议读取了你本地硬盘上的文件！")
            
            # 提示词：让大模型去读取刚刚生成的文件
            query = "帮我看看 E:\\dev\\my-agent\\mcp_secret.txt 这个文件里写了什么？"
            print(f"\n👤 你: {query}")
            
            inputs = {"messages": [HumanMessage(content=query)]}
            
            async for event in app.astream(inputs):
                for key, value in event.items():
                    last_msg = value["messages"][-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"  [🤖 决策] 准备调用工具: {last_msg.tool_calls[0]['name']}")
                        print(f"  [参数] {last_msg.tool_calls[0]['args']}")
                    elif key == "tools":
                        print(f"  [🔧 工具返回] {last_msg.content}")
                    elif hasattr(last_msg, 'content') and last_msg.content:
                        print(f"🤖 AI: {last_msg.content}")

if __name__ == "__main__":
    asyncio.run(main())