import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from openai import OpenAI
import chromadb

# 1. 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
embedding_model = os.getenv("QWEN_EMBEDDING_MODEL")

ds_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

client = chromadb.PersistentClient(path="./novel_db_v2")
collection = client.get_collection(name="novel_collection_v2")


def ask_agent(question, history):
    resp = TextEmbedding.call(model=embedding_model, input=question)
    if resp.status_code != 200:
        return "❌ 问题向量化失败。"

    question_embedding = resp.output['embeddings'][0]['embedding']

    # 【修改】扩大到 15 块，尽可能多地抓取上下文
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=15
    )
    retrieved_docs = results['documents'][0]
    retrieved_metadatas = results['metadatas'][0]

    context = ""
    for i in range(len(retrieved_docs)):
        chapter_name = retrieved_metadatas[i]['chapter']
        context += f"\n【出处：{chapter_name}】\n{retrieved_docs[i]}\n"

    # 【修改】重写提示词，要求它像个热情的书友一样详细回答
    system_prompt = f"""你是一个热情且严谨的小说阅读助手。
你的任务是根据提供的【参考资料】和【历史对话】来解答读者的问题。

【回答要求】：
1. 请根据检索到的片段，尽量**详细、丰满**地还原剧情。不要只干巴巴地扔出一句台词，请结合上下文说明当时的情境、人物的情绪和动作。
2. 如果用户问及章节，必须在回答末尾标注【出处：第x章】。
3. 结合【历史对话】理解用户的代词（如“他”、“那时”）。
4. 资料中未提及的，如实回答“未提及”，绝对不要自己编造。

【参考资料】：
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    response = ds_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7  # 【新增】加一点点创造力，让它说话更自然，不呆板
    )

    return response.choices[0].message.content


# --- 开启终端交互 ---
if __name__ == "__main__":
    print("=" * 50)
    print("📘 欢迎使用专属小说阅读 Agent！(记忆版)")
    print("💡 输入 'clear' 可以清空记忆，输入 'quit' 退出。")
    print("=" * 50)

    # 核心：用一个列表来存对话历史
    chat_history = []

    while True:
        user_question = input("\n👤 你: ")

        if user_question.strip().lower() in ['quit', 'exit', 'q']:
            break
        if user_question.strip().lower() == 'clear':
            chat_history = []
            print("🧹 记忆已清空！")
            continue
        if not user_question.strip():
            continue

        print("-" * 30)

        # 获取答案
        answer = ask_agent(user_question, chat_history)

        # 将这一轮的问答存入记忆
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": answer})

        print("\n💡 Agent 回答：")
        print(answer)
        print("=" * 50)
