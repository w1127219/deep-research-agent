import sys
import os
from mcp.server.fastmcp import FastMCP

# 创建一个名为 "MyLocalTools" 的 MCP 服务端
mcp = FastMCP("MyLocalTools")

@mcp.tool()
def get_local_file_content(file_path: str) -> str:
    """读取本地电脑上的文本文件内容。请输入完整的绝对路径，例如 E:\\dev\\my-agent\\test.txt"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"

@mcp.tool()
def get_system_info() -> str:
    """获取当前电脑的操作系统和用户信息"""
    return f"当前操作系统: {os.name}, 当前工作目录: {os.getcwd()}"

if __name__ == "__main__":
    # 使用标准输入输出（stdio）作为通信协议运行
    mcp.run(transport="stdio")