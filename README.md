# Novel-RAG-Agent
基于大模型和RAG架构的长文本智能问答系统

本项目旨在解决通用大模型在面对特定长文本（如私有文档、长篇报告）时易产生“幻觉”且难以溯源的问题，提供了一套轻量级的 RAG（检索增强生成）解决方案。

##  技术栈
后端：Python, FastAPI, ChromaDB
模型：DeepSeek API (文本生成), 通义千问 API (Embedding 向量化)
前端：HTML5, CSS3, 原生 JavaScript, marked.js

## 核心特性
私有知识库构建：使用正则与 TextSplitter 处理长文本，调用千问 Embedding 存入本地 ChromaDB。
精准溯源防幻觉：通过精细化 System Prompt 工程设定边界，确保回答严格基于检索片段并标注出处。
流式交互体验：后端基于 `StreamingResponse` 实现 SSE 流式输出，前端实现打字机效果与多轮对话记忆。

## 快速启动
1. 安装依赖：`pip install -r requirements.txt`
2. 配置环境变量：在根目录新建 `.env` 文件，填入你的 `DASHSCOPE_API_KEY` 和 `DEEPSEEK_API_KEY`。
3. 构建知识库：运行 `python xiaoShuo.py`。
4. 启动后端：运行 `python server.py`。
5. 运行前端：直接用浏览器打开 `index.html` 即可。
