import os
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# ==========================================
# 1. 定义工具
# ==========================================
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入纯数字算式，如 '3 * 5 + 2'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

# ==========================================
# 2. 定义状态（包含下一个由谁执行）
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str # 记录主管决定下一步派谁去

llm = ChatOpenAI(model="deepseek-chat", temperature=0)

# ==========================================
# 3. 定义专家节点（Worker）
# ==========================================
def math_expert(state: AgentState):
    """数学专家：负责调用计算器工具"""
    print("  [👨‍🔬 数学专家] 正在处理...")
    # 告诉模型它的角色，并给它绑定工具
    sys_msg = SystemMessage(content="你是数学专家，只能使用 calculator 工具解答数学题。")
    messages = [sys_msg] + state["messages"]
    
    # 专家可以使用工具
    llm_with_tools = llm.bind_tools([calculator])
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def chat_expert(state: AgentState):
    """闲聊专家：只负责日常对话，不使用工具"""
    print("  [💬 闲聊专家] 正在处理...")
    sys_msg = SystemMessage(content="你是闲聊助手，请用简短有趣的话语回答常识问题，不要用工具。")
    messages = [sys_msg] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# ==========================================
# 4. 定义主管节点（Supervisor）
# ==========================================
def supervisor(state: AgentState):
    """主管：不看历史，只看最后一句，决定派谁去"""
    print("  [👔 主管] 正在分配任务...")
    
    sys_prompt = """你是一个任务分配主管。请根据用户的最新问题，决定由谁来处理：
    - 如果是数学计算题，回复：math_expert
    - 如果是其他问题，回复：chat_expert
    请只回复专家名称（math_expert 或 chat_expert），不要输出任何多余的字。"""
    
    # 我们只把用户的最后一句话给主管看，免得主管被历史记录绕晕
    last_user_msg = state["messages"][-1].content
    
    response = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"用户问题：{last_user_msg}")
    ])
    
    # 提取主管的决定
    decision = response.content.strip()
    if "math_expert" in decision:
        decision = "math_expert"
    else:
        decision = "chat_expert"
        
    print(f"  [👔 主管] 决定指派给: {decision}")
    return {"next_agent": decision}

# ==========================================
# 5. 定义条件路由（主管的分配逻辑）
# ==========================================
def route_to_expert(state: AgentState):
    """根据主管的决定，把状态路由到对应节点"""
    return state["next_agent"]

# 定义工具执行节点
tool_node = ToolNode([calculator])

# ==========================================
# 6. 组装多智能体图（Graph）
# ==========================================
builder = StateGraph(AgentState)

# 添加节点
builder.add_node("supervisor", supervisor)
builder.add_node("math_expert", math_expert)
builder.add_node("chat_expert", chat_expert)
builder.add_node("tools", tool_node) # 专门给数学专家用的工具

# 入口永远是主管
builder.add_edge(START, "supervisor")

# 主管的条件路由
builder.add_conditional_edges(
    "supervisor",
    route_to_expert,
    {
        "math_expert": "math_expert",
        "chat_expert": "chat_expert"
    }
)

# 数学专家如果调用了工具，去 tools；否则结束
def route_math(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return END

builder.add_conditional_edges(
    "math_expert",
    route_math,
    {"tools": "tools", END: END}
)

# 闲聊专家干完活直接结束
builder.add_edge("chat_expert", END)

# 工具执行完后，回到数学专家，给出最终结论
builder.add_edge("tools", "math_expert")

app = builder.compile()

# ==========================================
# 7. 测试运行（修复了输出显示）
# ==========================================
if __name__ == "__main__":
    print("🚀 多智能体协作系统已启动！\n")
    
    def run_test(query):
        print(f"\n--- 用户提问: {query} ---")
        inputs = {"messages": [HumanMessage(content=query)]}
        
        for event in app.stream(inputs):
            for node_name, state in event.items():
                # 🌟 关键修复：检查 state 里有没有 messages 字段
                if "messages" not in state:
                    continue  # 如果没有（比如主管的分配环节），直接跳过
                    
                last_msg = state["messages"][-1]
                
                # 如果模型决定调用工具，打印决策
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"  [🤖 决策] {node_name} 准备调用工具: {last_msg.tool_calls[0]['name']}")
                # 如果是工具执行完毕，打印工具结果
                elif node_name == "tools":
                    print(f"  [🔧 执行] 工具返回: {last_msg.content}")
                # 如果是专家节点给出的最终回复（非工具调用内容）
                elif hasattr(last_msg, 'content') and last_msg.content:
                    print(f"  [💡 最终回答] {last_msg.content}")

    # 测试 1：需要数学专家
    run_test("帮我算一下 888 除以 2 等于多少？")
    
    # 测试 2：需要闲聊专家
    run_test("请给我讲一个关于猫的笑话。")