import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# 引入 LangGraph 的核心组件
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 1. 定义工具（使用 @tool 装饰器，模型原生识别）
# ==========================================
@tool
def get_current_time(query: str = "") -> str:
    """获取当前的日期和时间。"""
    import datetime
    return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入必须是一个纯数学公式，例如 '3 * 5 + 2'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

tools = [get_current_time, calculator]
tools_map = {t.name: t for t in tools}

# ==========================================
# 2. 初始化模型并绑定工具
# ==========================================
# 这里不要加 temperature=0，部分模型调用工具时设为 0 可能会报错
llm = ChatOpenAI(model="deepseek-chat")
# 告诉模型：你现在有了这两个工具，你可以自己决定什么时候用
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 定义 Agent 的状态（State）
# ==========================================
class AgentState(TypedDict):
    # 消息列表，add_messages 会自动把新消息追加到历史里，不用手动 append
    messages: Annotated[list, add_messages]

# ==========================================
# 4. 定义节点（Node）
# ==========================================
def chatbot(state: AgentState):
    """调用模型进行思考的节点"""
    messages = state["messages"]
    # 如果这是第一轮，强制注入系统提示词
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content="你是一个严谨的智能助手。请根据用户的问题，决定直接回答，还是调用工具。")
        messages = [sys_msg] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# 使用内置的 ToolNode，它会自动解析模型返回的工具调用，并执行对应函数
tool_node = ToolNode(tools)

# ==========================================
# 5. 定义条件路由（Edge）
# ==========================================
def route_after_chatbot(state: AgentState):
    """决定下一步去哪里：如果模型要调工具，就去 tools；否则结束"""
    last_message = state["messages"][-1]
    # 如果模型返回了 tool_calls，说明要调工具
    if last_message.tool_calls:
        return "tools"
    # 否则直接结束
    return END

# ==========================================
# 6. 组装图（Graph）
# ==========================================
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# 设置入口
graph_builder.add_edge(START, "chatbot")

# 条件边：从 chatbot 出来后，走 route_after_chatbot 逻辑
graph_builder.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
    {
        "tools": "tools", # 如果返回 "tools"，跳转到 tools 节点
        END: END          # 如果返回 END，直接结束
    }
)

# 普通边：工具执行完后，必须再回到 chatbot 把结果喂给模型
graph_builder.add_edge("tools", "chatbot")

# ==========================================
# 6.5 赋予记忆能力（保存状态）
# ==========================================
memory = MemorySaver()
# 告诉图：在执行 tools 节点之前，先暂停！
app = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"] 
)
# ==========================================
# 7. 测试运行（带记忆的交互式对话）
# ==========================================
if __name__ == "__main__":
    print("🚀 LangGraph 带审批功能的 Agent 已启动！")
    
    config = {"configurable": {"thread_id": "user_session_1"}}
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # 运行图
        for event in app.stream(inputs, config=config):
            for key, value in event.items():
                # 🌟 关键修复：过滤掉 LangGraph 内部的中断系统信号
                if key == "__interrupt__":
                    continue
                    
                last_msg = value["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"  [🤖 决策] 模型想要调用工具: {last_msg.tool_calls[0]['name']}")
                elif key == "tools":
                    print(f"  [🔧 执行] 工具返回结果: {last_msg.content}")
                elif key == "chatbot" and not hasattr(last_msg, 'tool_calls'):
                    print(f"🤖 AI: {last_msg.content}")        # ==========================================
        # 关键：检查图是否暂停了（停在工具节点之前）
        # ==========================================
        state = app.get_state(config)
        if state.next: # 如果 next 不是空的，说明卡在某个节点前面了
            print(f"\n⚠️ [安全拦截] 模型准备执行工具: {state.next}")
            
            # 打印一下模型到底传了什么参数
            last_ai_msg = state.values["messages"][-1]
            print(f"待执行的操作: {last_ai_msg.tool_calls[0]['args']}")
            
            approve = input("👉 是否批准执行？(y/n): ")
            if approve.lower() == 'y':
                print("✅ 已批准，继续执行...")
                for event in app.stream(None, config=config):
                    for key, value in event.items():
                        # 🌟 这里也加上过滤
                        if key == "__interrupt__":
                            continue
                        if key == "tools":
                            print(f"  [🔧 执行] 工具返回结果: {value['messages'][-1].content}")
                        elif key == "chatbot":
                            print(f"🤖 AI: {value['messages'][-1].content}")