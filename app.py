import streamlit as st
import os
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# 引入 LangChain 和 LangGraph 组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langgraph.graph import StateGraph, START, END

# 1. 设置全局 User-Agent，伪装成正常浏览器，防止爬虫被拦截
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# 加载环境变量
load_dotenv()

# ==========================================
# 页面基础设置
# ==========================================
st.set_page_config(page_title="AI 深度研究员", page_icon="🔬", layout="wide")
st.title("🔬 企业级 Deep Research 智能体")
st.markdown("输入你的研究课题，AI 将自动进行**任务拆解、全网搜索、撰写报告、自我审查**。")

# ==========================================
# 定义状态与节点逻辑
# ==========================================
class ResearchState(TypedDict):
    topic: str
    search_queries: list
    search_results: str
    report: str
    feedback: str
    loop_count: int

# 初始化大模型：连接本地 Ollama
llm = ChatOpenAI(
    model="deepseek-r1:7b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 任意非空字符串
    temperature=0.3
)

# --- 节点 1：规划者 (Planner) ---
def planner_node(state: ResearchState):
    feedback_prompt = ""
    if state.get("feedback"):
        feedback_prompt = f"\n注意：上一版报告被打回了，意见是：{state['feedback']}。请针对意见补充新的查询词。"

    prompt = f"""你是一个资深的行业分析师。请针对课题：“{state['topic']}”生成 3 个搜索查询词。
    {feedback_prompt}
    要求：只输出3个搜索词，用英文逗号分隔。不要包含序号、引号、换行符或任何其他废话。
    输出格式示例：固态电池 现状, 固态电池 技术突破, 固态电池 市场预测"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # 🌟 终极清洗逻辑：解决本地模型乱输出格式的问题
    # 1. 去除 R1 模型的思考过程标签
    content = content.replace("", "") 
    
    # 2. 暴力清洗：把换行符全部变成逗号
    content = content.replace("\n", ",")
    
    # 3. 正则去除所有的序号（比如 "1. " "2. "）
    content = re.sub(r'\d+\.', '', content)
    
    # 4. 去除各种奇怪的符号（引号、中括号等）
    content = content.replace('"', '').replace("'", "").replace("[", "").replace("]", "")
    
    # 5. 把所有的中文逗号、分号、顿号，全部统一替换成英文逗号
    content = re.sub(r'[，；、]', ',', content)
    
    # 6. 按英文逗号切分，并去掉空字符串
    queries = [q.strip() for q in content.split(",") if q.strip()]
    
    # 7. 强行只保留前 3 个词
    queries = queries[:3]
    
    print(f"  🧹 终极清洗后的搜索词: {queries}")
    
    return {"search_queries": queries, "loop_count": state.get("loop_count", 0) + 1}

# --- 节点 2：搜索者 (Researcher) ---
def researcher_node(state: ResearchState):
    search_tool = TavilySearchResults(max_results=2)
    all_content = state.get("search_results", "")
    
    for query in state["search_queries"]:
        try:
            search_docs = search_tool.invoke(query)
            urls = [doc["url"] for doc in search_docs]
            if urls:
                # 使用伪装头加载网页
                loader = WebBaseLoader(urls)
                page_docs = loader.load()
                for doc in page_docs:
                    all_content += f"\n\n--- {doc.metadata.get('source', '未知')} ---\n"
                    all_content += doc.page_content[:2000]
        except Exception as e:
            print(f"  ⚠️ 搜索或读取失败: {e}")
            
    return {"search_results": all_content}

# --- 节点 3：撰写者 (Writer) ---
def writer_node(state: ResearchState):
    prompt = f"""顶级行业研究员，请根据资料写一份关于“{state['topic']}”的深度报告。
    要求：结构清晰（摘要、现状、趋势、结论），Markdown 格式。
    
    【资料】：{state['search_results']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # 清洗 R1 模型的思考过程
    content = content.replace("", "") 
        
    return {"report": content}

# --- 节点 4：主编审查 (Reviewer) ---
def reviewer_node(state: ResearchState):
    prompt = f"""严格的行业主编，评估关于“{state['topic']}”的报告。
    报告：{state['report']}
    如果你认为完美，回复：PASS
    如果认为需补充，回复：FAIL。换行写明需要补充什么。"""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "PASS" in response:
        return {"feedback": "PASS"}
    else:
        return {"feedback": response.replace("FAIL", "").strip()}

# --- 路由条件 ---
def route_review(state: ResearchState):
    if state["feedback"] == "PASS":
        return END
    if state.get("loop_count", 0) >= 2:
        return END
    return "planner"

# ==========================================
# 构建 Graph
# ==========================================
@st.cache_resource
def get_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("reviewer", reviewer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "reviewer")
    builder.add_conditional_edges("reviewer", route_review, {END: END, "planner": "planner"})
    return builder.compile()

app = get_graph()

# ==========================================
# 前端交互界面
# ==========================================
topic = st.text_input("请输入研究课题：", "2024年固态电池产业链的发展现状与突破")

if st.button("🚀 启动深度研究", type="primary"):
    if not topic.strip():
        st.warning("请输入课题！")
    else:
        status_box = st.status("🔍 智能体正在工作...", expanded=True)
        initial_state = {"topic": topic, "loop_count": 0, "search_results": ""}
        
        final_report = ""
        
        for event in app.stream(initial_state):
            for node_name, state in event.items():
                if node_name == "planner":
                    status_box.write(f"🧠 **规划者**：生成了搜索词 `{state['search_queries']}`")
                elif node_name == "researcher":
                    length = len(state.get("search_results", ""))
                    status_box.write(f"🌐 **搜索者**：资料收集完毕，共 {length} 字符。")
                elif node_name == "writer":
                    status_box.write(f"✍️ **撰写者**：初稿完成。")
                    # 🌟 在这里把生成的报告存下来
                    final_report = state["report"]
                elif node_name == "reviewer":
                    if state.get("feedback") == "PASS":
                        status_box.write(f"🧐 **主编**：报告完美，予以通过！✅")
                    else:
                        status_box.write(f"🧐 **主编**：报告不达标，打回重写！意见：{state['feedback']} ❌")
                        
        status_box.update(label="🎉 研究完成！", state="complete", expanded=False)
        
        st.divider()
        if final_report:
            st.markdown(final_report)
            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=final_report,
                file_name=f"{topic}_研究报告.md",
                mime="text/markdown"
            )
        else:
            st.error("生成报告失败，请检查终端日志。")