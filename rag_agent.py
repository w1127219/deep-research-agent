import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

# 引入 RAG 相关组件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 引入 LangGraph 组件
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ==========================================
# 1. 构建本地知识库（只跑一次，存入 Chroma）
# ==========================================
print("⏳ 正在加载并索引本地文档...")
# 读取刚才的 txt
loader = TextLoader("company_rules.txt", encoding="utf-8")
docs = loader.load()

# 文本切块（长文档必须切碎，模型才能精准定位）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

# 使用免费的本地中文向量模型 BAAI/bge-small-zh-v1.5
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 存入本地 Chroma 数据库
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
# 转换成检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 每次找最相关的 2 块

# ==========================================
# 2. 把“检索器”包装成 Agent 的一个工具
# ==========================================
@tool
def search_company_rules(query: str) -> str:
    """当用户询问关于公司制度、福利、产品、工作时间等问题时，使用此工具查阅内部文档。"""
    print(f"  [🔍 检索] 正在搜索知识库: {query}")
    docs = retriever.invoke(query)
    # 把找到的文档片段拼接成一个长字符串返回
    result = "\n\n".join([doc.page_content for doc in docs])
    return result if result else "文档中没有找到相关信息。"

tools = [search_company_rules]
llm = ChatOpenAI(model="deepseek-chat", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. LangGraph 节点与状态
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(content="你是星火科技公司的智能客服。回答用户问题时，如果涉及公司内部信息，请务必使用 search_company_rules 工具去查证。如果查不到，请如实回答不知道。")
        messages = [sys_msg] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def route_after_chatbot(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# ==========================================
# 4. 组装图
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", route_after_chatbot, {"tools": "tools", END: END})
builder.add_edge("tools", "chatbot")

memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# ==========================================
# 5. 交互式测试
# ==========================================
if __name__ == "__main__":
    print("\n🚀 RAG 智能客服已启动！ (输入 exit 退出)")
    print("提示：你可以问问它关于‘星火科技’的事情。\n")
    
    config = {"configurable": {"thread_id": "rag_session"}}
    
    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        for event in app.stream(inputs, config=config):
            for node_name, state in event.items():
                if node_name == "__interrupt__" or "messages" not in state:
                    continue
                    
                last_msg = state["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"  [🤖 决策] 准备查阅文档...")
                elif node_name == "tools":
                    print(f"  [📄 查到的资料] \n{last_msg.content}\n")
                elif hasattr(last_msg, 'content') and last_msg.content:
                    print(f"🤖 AI: {last_msg.content}")