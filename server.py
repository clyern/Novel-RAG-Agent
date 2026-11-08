import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from openai import OpenAI
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 1. 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
embedding_model = os.getenv("QWEN_EMBEDDING_MODEL")
ds_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))

# 2. 连数据库
client = chromadb.PersistentClient(path="./novel_db_v2")
collection = client.get_collection(name="novel_collection_v2")

# 3. 开启 Web 服务
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    history: list = []


# 4. 核心问答逻辑（流式生成器）
def generate_response(request: ChatRequest):
    # 向量检索
    resp = TextEmbedding.call(model=embedding_model, input=request.question)
    question_embedding = resp.output['embeddings'][0]['embedding']
    results = collection.query(query_embeddings=[question_embedding], n_results=10)

    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    context = ""
    for i in range(len(retrieved_docs)):
        context += f"\n【出处：{retrieved_metadatas[i]['chapter']}】\n{retrieved_docs[i]}\n"

    system_prompt = f"""你是一个热情且严谨的小说阅读助手。
根据提供的【参考资料】和【历史对话】解答问题。
要求：详细还原剧情，结合上下文。末尾必须标注【出处：第x章】。
【参考资料】：{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in request.history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": request.question})

    # 调用 DeepSeek，开启 stream=True
    response = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    # 迭代生成数据块并 yield 给前端
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(generate_response(request), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("🚀 后端服务已启动！(流式输出版)")
    uvicorn.run(app, host="127.0.0.1", port=8000)