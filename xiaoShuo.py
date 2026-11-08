import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import re

# 1. 加载环境变量
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
embedding_model = os.getenv("QWEN_EMBEDDING_MODEL")

# 2. 读取 txt
novel_path = "sample_document.txt"

# 如果文件不存在，自动生成一个测试文件，防止面试官拉取代码后直接报错
if not os.path.exists(novel_path):
    with open(novel_path, "w", encoding="utf-8") as f:
        f.write("这是一个测试文档。\n\n第1章 测试章节\n\n这是用来测试 RAG 系统的示例文本。你可以将其替换为你自己的长文本。")
    print("⚠️ 未找到示例文件，已自动生成 sample_document.txt")

try:
    with open(novel_path, "r", encoding="utf-8") as f:
        novel_text = f.read()
    print(f"✅ 读取成功，总字数约: {len(novel_text)} 字")
except Exception as e:
    print(f"❌ 读取文件失败: {e}")
    exit()



# 3. 文本切割（升级版：带上章节元数据）
print("\n正在切割文本并提取章节信息...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)

# 按章节切分 (假设你的txt里章节名格式类似“第x章”或“第x节”)
# 用正则表达式匹配章节，保留章节名
chapters = re.split(r'(第[0-9一二三四五六七八九十百千万]+[章回节].*?\n)', novel_text)

chunks = []
metadatas = []

current_chapter = "未知章节"
for part in chapters:
    if not part.strip(): continue

    # 如果这一小段是章节标题
    if re.match(r'第[0-9一二三四五六七八九十百千万]+[章回节]', part):
        current_chapter = part.strip()
    else:
        # 这是正文，进行切分
        split_chunks = text_splitter.split_text(part)
        for chunk in split_chunks:
            chunks.append(chunk)
            # 核心：把章节名和文本块绑定在一起存入数据库
            metadatas.append({"chapter": current_chapter})

print(f"✅ 切割完成，共切分为 {len(chunks)} 个文本块")

# 4 & 5. 向量化并存入 ChromaDB
print("\n开始计算向量并存入数据库 (请稍候)...")

# 注意：这里换了一个新的数据库文件夹叫 novel_db_v2，防止和老数据冲突
client = chromadb.PersistentClient(path="./novel_db_v2")
collection = client.get_or_create_collection(name="novel_collection_v2")

batch_size = 25
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i: i + batch_size]
    batch_metadatas = metadatas[i: i + batch_size]  # 获取对应批次的章节信息

    resp = TextEmbedding.call(model=embedding_model, input=batch_chunks)

    if resp.status_code == 200:
        embeddings = [item['embedding'] for item in resp.output['embeddings']]
        ids = [f"id_{i + j}" for j in range(len(batch_chunks))]

        # 将 metadatas 一起存入
        collection.add(
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=ids
        )
        print(f"已处理并存入: {i + len(batch_chunks)} / {len(chunks)} 块")
    else:
        print(f"❌ 向量化失败: {resp.code} - {resp.message}")

print("\n🎉 知识库 v2 构建完成！")