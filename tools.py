import datetime

def get_current_time(query: str) -> str:
    """获取当前具体的日期和时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")

def calculator(expression: str) -> str:
    """用于计算数学表达式。输入必须是一个纯数学公式，例如 '3 * 5 + 2'"""
    try:
        # 注意：实际生产中不要直接用 eval，这里为了演示方便
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"