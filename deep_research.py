import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader

from langgraph.graph import StateGraph, START, END

load_dotenv()

# ==========================================
# 1. 定义状态（新增：反馈、循环次数）
# ==========================================
class ResearchState(TypedDict):
    topic: str
    search_queries: list
    search_results: str
    report: str
    feedback: str    # 主编的修改意见
    loop_count: int  # 记录循环了多少次

llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)

# ==========================================
# 2. 节点一：规划者 (Planner)
# ==========================================
def planner_node(state: ResearchState):
    print("\n🧠 [规划者] 正在拆解研究任务...")
    
    # 如果是第一次规划，或者带上了主编的反馈，生成不同的查询词
    feedback_prompt = ""
    if state.get("feedback"):
        feedback_prompt = f"\n注意：上一版报告被主编打回了，主编的意见是：{state['feedback']}。请针对这个意见补充新的查询词。"

    prompt = f"""你是一个资深的行业分析师。
请针对研究课题：“{state['topic']}”，生成 3 个用于搜索引擎的查询词。
{feedback_prompt}
要求：直接用逗号分隔输出这三个查询词，不要包含任何其他废话或序号。"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    queries = [q.strip() for q in response.content.split(",")]
    print(f"  👉 拟定搜索词: {queries}")
    
    # 循环次数 + 1
    return {"search_queries": queries, "loop_count": state.get("loop_count", 0) + 1}

# ==========================================
# 3. 节点二：搜索者 (Researcher)
# ==========================================
def researcher_node(state: ResearchState):
    print("\n🌐 [搜索者] 正在全网搜集资料...")
    search_tool = TavilySearchResults(max_results=2)
    
    # 注意：要把之前搜过的资料保留下来，拼上新的资料
    all_content = state.get("search_results", "")
    
    for query in state["search_queries"]:
        print(f"  🔍 正在搜索: {query}")
        try:
            search_docs = search_tool.invoke(query)
            urls = [doc["url"] for doc in search_docs]
            if urls:
                loader = WebBaseLoader(urls)
                page_docs = loader.load()
                for doc in page_docs:
                    all_content += f"\n\n--- 来自 {doc.metadata.get('source', '未知')} 的资料 ---\n"
                    all_content += doc.page_content[:2000] # 每篇截取2000字
        except Exception as e:
            print(f"  ⚠️ 搜索失败: {e}")
            
    print(f"  ✅ 资料搜集完毕，目前共有 {len(all_content)} 个字符的素材。")
    return {"search_results": all_content}

# ==========================================
# 4. 节点三：撰写者 (Writer)
# ==========================================
def writer_node(state: ResearchState):
    print("\n✍️ [撰写者] 正在研读资料并撰写报告...")
    prompt = f"""你是一个顶级的行业研究员。请根据以下资料，写一份关于“{state['topic']}”的深度研究报告。
要求：
1. 结构清晰（包含：摘要、核心现状、未来趋势、结论）。
2. 必须使用 Markdown 格式。
3. 如果资料里有很多废话，请自行提炼。

【参考资料】：
{state['search_results']}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content}

# ==========================================
# 5. 节点四：主编审查 (Reviewer) - 新增！
# ==========================================
def reviewer_node(state: ResearchState):
    print("\n🧐 [主编] 正在审阅报告质量...")
    
    # 强制主编用结构化格式回答
    prompt = f"""你是一个极其严格的行业研究主编。请评估这份关于“{state['topic']}”的报告。
报告内容：
{state['report']}

如果你认为报告内容详实、结构完整，请只回复：PASS
如果你认为报告缺乏深度、数据不足，请回复：FAIL。随后换行写明你需要补充什么信息（这将被发给搜索员）。"""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    if "PASS" in response:
        print("  ✅ 主编：报告完美，予以通过！")
        return {"feedback": "PASS"}
    else:
        # 提取反馈意见
        feedback = response.replace("FAIL", "").strip()
        print(f"  ❌ 主编：报告不达标。打回修改！意见：{feedback}")
        return {"feedback": feedback}

# ==========================================
# 6. 条件路由（核心！决定是结束还是循环）
# ==========================================
def route_review(state: ResearchState):
    # 如果通过，结束
    if state["feedback"] == "PASS":
        return END
    # 如果没通过，但已经循环了 2 次（防止无限死循环），也强制结束
    if state.get("loop_count", 0) >= 2:
        print("\n⚠️ 达到最大循环次数，强制结束流程。")
        return END
    # 否则，打回重做
    return "planner"

# ==========================================
# 7. 组装带“环”的图
# ==========================================
builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "reviewer")

# 从 reviewer 出来后，走条件路由（要么去 END，要么回去找 planner）
builder.add_conditional_edges(
    "reviewer",
    route_review,
    {
        END: END,
        "planner": "planner"
    }
)

app = builder.compile()

# ==========================================
# 8. 运行测试
# ==========================================
if __name__ == "__main__":
    print("🚀 带“反思循环”的 Deep Research 系统启动！")
    
    # 故意给一个比较宽泛的题目，逼主编挑刺
    topic = "2024年开源大模型与闭源大模型的性能差距分析"
    print(f"👤 研究课题: {topic}")
    
    # 执行图（初始 loop_count 为 0）
    final_state = app.invoke({"topic": topic, "loop_count": 0, "search_results": ""})
    
    print("\n" + "="*50)
    print("🎉 最终研究报告生成完毕：")
    print("="*50 + "\n")
    print(final_state["report"])