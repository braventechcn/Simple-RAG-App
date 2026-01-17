import chromadb
from chromadb.config import Settings
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time
import uuid
from dotenv import load_dotenv
from datetime import datetime


# Step 1: Text Embedding Service
class EmbeddingService:
    """Text Embedding Service supporting both OpenAI API and local embedding models."""

    def __init__(self, use_local_model=True, local_model_name="bge-base-zh-v1.5"):
        """
        Initialize the embedding service.
        Args:
            use_local_model: Whether to use a local embedding model (default: True).
            local_model_name: The name of the local embedding model to use (default: "bge-base-zh-v1.5").
        """
        self.use_local_model = use_local_model

        if self.use_local_model:
            # Initialize the local embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_name)
            self.model = AutoModel.from_pretrained(local_model_name)
            self.model.eval()  # Set the model to evaluation mode
        else:
            # Initialize the OpenAI API client
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.model = "text-embedding-ada-002"

    def get_embedding(self, text: str) -> list:
        """
        Get the embedding vector for the given text.
        Args:
            text: Input text to be embedded.
        Returns:
            List of floats representing the embedding vector.
        """
        if self.use_local_model:
            # Use the local embedding model
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.model(**inputs)
                # Use the mean pooling of the last hidden state as the embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            return embedding
        else:
            # Use the OpenAI API
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding


# Step 2: Chroma Memory Manager
class ChromaMemoryManager:
    """Chroma Memory Manager for storing and retrieving user chat memories."""

    def __init__(self, persist_directory="./chroma_db/user_memory"):
        """
        Initialize a Chroma database client and create a collection for storing user chat memories.
        初始化一个 Chroma 数据库客户端, 并创建或获取一个用于存储用户聊天记忆的集合
        Args:
            persist_directory: Directory for data persistence
        """
        self.persist_directory = persist_directory

        # Create persistent client
        # - chromadb.Client: 是 Chroma 的核心类，用于与向量数据库交互; client 需要一个 Settings 对象来配置数据库的实现方式和存储位置
        # - chromadb.config.Settings: 是 Chroma 的配置类, 用于定义数据库的存储后端(数据库实现方式)和数据持久化目录等其他参数设置
        #   =============
        #   - 从 1.4 开始，ChromaDB 移除了旧的 Settings 配置结构，不再暴露 chroma_db_impl
        #	- 统一用 client = chromadb.Client()  # 默认 In-Memory
        #   - 统一用 client = chromadb.PersistentClient(path="path/to/dir") # 管理持久化
        #	- 默认后端引擎就是 SQlite（无需配置）
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Get or create collection (Collection) for user memories
        # - get_or_create_collection: 是 Chroma 提供的一个方法，用于获取一个已存在的集合（collection），或者在集合不存在时创建一个新的集合
        self.collection = self.client.get_or_create_collection(
            name="emotion_chat_memory",     # 集合的名称，用于标识存储的内容
            metadata={
                "description": "User memory system for the Emotion Chatbot"
            } # metadata 是一个字典，用于存储集合的描述信息或其他元数据等附加信息
        )

        print(f"=====Chroma initialized successfully=====")
        print(f"  - Storage Path: {persist_directory}")
        print(f"  - Collection Name: emotion_chat_memory")
        
        self.embedding_service = EmbeddingService()

    def add_memory(self, user_id: str, text: str, metadata: dict = None):
        """
        Add a new memory entry for a user.
        Args:
            user_id: User ID
            text: Memory content
            metadata: Metadata (emotion, timestamp, etc.)
        """
        # Generate vector
        vector = self.embedding_service.get_embedding(text)

        # Construct metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            "user_id": user_id,                     # 关联用户ID
            "text": text,                           # 原始文本内容
            "timestamp": time.time(),               # 时间戳
            "datetime": datetime.now().isoformat()  # 可读时间格式
        })

        # Generate unique memory ID
        memory_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

        # Store in Chroma
        self.collection.add(
            ids=[memory_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[text]  # Chroma supports storing the original text as well
        )

        print(f"Memory saved: {memory_id}")
        return memory_id
    
    def retrieve_memories(self, user_id: str, query_text: str, top_k: int = 5):
        """
        Retrieve top_k relevant memories for a user based on a query text.
        Args:
            user_id: User ID
            query_text: Query text for similarity search
            top_k: Number of top relevant memories to retrieve
        Returns:
            List of retrieved memories with metadata
        """
        # Generate query vector
        query_vector = self.embedding_service.get_embedding(query_text)

        # Perform similarity search in Chroma
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"user_id": user_id}  # Filter by user_id to get user-specific memories
        )

        # Extract memory content
        memories = []
        if results["documents"]:
            for ids, doc, meta, distance in zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0]
            ):
                memories.append({
                    "id": ids,
                    "text": doc,
                    "metadata": meta,
                    "score": distance if distance is not None else 0.0
            })

        print(f"Retrieved {len(memories)} memories for user: {user_id} ")
        return memories

    def get_user_all_memories(self, user_id: str):
        """获取用户的所有记忆"""
        # Chroma 的 get() 方法可以通过 where 过滤
        results = self.collection.get(
            where={"user_id": user_id}
        )

        memories = []
        for i, doc in enumerate(results['documents']):
            memory = {
                'id': results['ids'][i],
                'text': doc,
                'metadata': results['metadatas'][i]
            }
            memories.append(memory)

        return memories

    def delete_memory(self, memory_id: str):
        """删除指定记忆"""
        self.collection.delete(ids=[memory_id])
        print(f"✓ 记忆已删除: {memory_id}")
        
    def clear_user_memories(self, user_id: str):
        """清空用户的所有记忆"""
        # 先获取该用户的所有记忆ID
        results = self.collection.get(
            where={"user_id": user_id}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"✓ 已清空用户 {user_id} 的 {len(results['ids'])} 条记忆")
        else:
            print(f"用户 {user_id} 没有记忆")

    def get_memory_stats(self):
        """获取记忆统计信息"""
        # 获取所有记忆
        all_data = self.collection.get()

        total_count = len(all_data['ids'])

        # 按用户统计
        user_counts = {}
        for metadata in all_data['metadatas']:
            user_id = metadata.get('user_id', 'unknown')
            user_counts[user_id] = user_counts.get(user_id, 0) + 1

        return {
            "total_memories": total_count,
            "user_count": len(user_counts),
            "memories_by_user": user_counts
        }
    
    def retrieve_memories_with_time_decay(self, user_id: str, query: str, k: int = 3):
        """检索记忆时考虑时间衰减"""
        # 检索更多结果
        results = self.collection.query(
            query_embeddings=[self.get_embedding(query)],
            n_results=k * 2,
            where={"user_id": user_id}
        )

        # 计算时间加权分数
        import time
        current_time = time.time()

        scored_memories = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            timestamp = results['metadatas'][0][i].get('timestamp', current_time)

            # 时间衰减：30天内权重为1，之后每天衰减1%
            days_old = (current_time - timestamp) / 86400
            time_weight = max(0.5, 1 - (max(0, days_old - 30) * 0.01))

            # 综合分数（距离越小越好，权重越大越好）
            score = (1 - distance) * time_weight

            scored_memories.append({
                'doc': doc,
                'metadata': results['metadatas'][0][i],
                'score': score
            })

        # 按分数排序，返回top-k
        scored_memories.sort(key=lambda x: x['score'], reverse=True)
        return scored_memories[:k]
    
# 记忆存储策略
# -存储记忆也是有成本的，因此需要设计合理的存储策略。可以有选择地保存重要内容。
 # ✓ 好的做法：有选择地存储重要对话
def should_save_memory(message: str, emotion: str) -> bool:
    """判断是否应该保存为长期记忆"""
    # 重要情绪
    important_emotions = ["焦虑", "抑郁", "压力大", "痛苦"]

    # 重要话题
    important_keywords = ["工作", "家庭", "健康", "梦想", "目标"]

    return (
        emotion in important_emotions or
        any(keyword in message for keyword in important_keywords)
    )

# ✗ 避免：保存所有对话（会导致存储膨胀和检索噪音）   
    

# Step 3: Example Usage
def chat_with_memory(user_id: str, user_input: str, memory_manager: ChromaMemoryManager):
    """
    Chat with memory integration.
    Args:
        user_id: User ID
        user_input: User input text
        memory_manager: ChromaMemoryManager instance
    """
    # 1. Retrieve relevant memories
    print("\nRetrieving user memories...")
    relevant_memories = memory_manager.retrieve_memories(
        user_id=user_id,
        query_text=user_input,
        top_k=5
    )

    # 2. Construct memory context
    context = ""
    if relevant_memories:
        context = "以下是用户过去提到的相关内容：\n"
        for i, mem in enumerate(relevant_memories, 1):
            emotion = mem['metadata'].get('emotion', '未知')
            datetime_str = mem['metadata'].get('datetime', '未知时间')
            context += f"{i}. [{datetime_str}] [{emotion}] {mem['text']}\n"

    # 3. Construct prompt with memory context
    prompt = f"""\n历史记忆：{context if context else "（这是用户第一次对话，暂无历史记忆）"}
                 \n当前输入：\n{user_input}请以共情、支持的语气回应，避免机械重复。如果历史记忆中有相关内容，请自然地关联起来，展现你对用户情况的了解和关心。
             """

    # 4. Call the large model
    print("\nGenerating response...")
    try:
        client = OpenAI(
            # api_key=os.getenv("OPENAI_API_KEY")
            api_key=os.getenv("DASHSCOPE_API_KEY"), # 使用 DashScope API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",  
            messages=[
                {"role": "system", "content":"你是一位温暖的心理陪伴者聊天机器人。请结合用户当前输入和历史记忆进行回应。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content
        # 如需查看完整响应，请取消下列注释
        # print(completion.model_dump_json())
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

    # 5. Save new memories
    print("\nSaving new memories...")
    # Save user input (optional: perform sentiment analysis first)
    emotion = predict_emotion(user_input)  # Needs implementation of sentiment analysis
    memory_manager.add_memory(
        user_id=user_id,
        text=user_input,
        metadata={
            "role": "user",
            "emotion": emotion,
            "type": "user_message"
        }
    )
    # Save model response
    memory_manager.add_memory(
        user_id=user_id,
        text=answer,
        metadata={
            "role": "assistant",
            "type": "ai_response"
        }
    )

    return answer, relevant_memories


def predict_emotion(text: str) -> str:
    """
    Simple sentiment analysis to predict emotion from text.
    """
    # Here you can integrate a sentiment analysis model
    # ToDo:
    # For demonstration, use simple keyword matching
    keywords = {
        "焦虑": ["焦虑", "担心", "紧张", "害怕"],
        "压力大": ["压力", "累", "疲惫", "撑不住"],
        "悲伤": ["难过", "伤心", "悲伤", "痛苦"],
        "愤怒": ["生气", "愤怒", "气愤", "恼火"],
        "快乐": ["开心", "高兴", "快乐", "愉快"]
    }

    for emotion, kws in keywords.items():
        if any(kw in text for kw in kws):
            return emotion

    return "平静"

# test_memory_system.py

# 加载环境变量
load_dotenv()

def test_memory_system():
    """测试记忆系统"""
    print("=" * 70)
    print(" 心语机器人 - Chroma 记忆系统测试")
    print("=" * 70)

    # 初始化记忆管理器
    memory_manager = ChromaMemoryManager(persist_directory="./chroma_db/test_memory")

    user_id = "test_user_001"

    # 第一轮对话
    print("\n【第1轮对话】")
    print("-" * 70)
    user_input_1 = "最近工作压力好大，每天都加班到很晚。"
    print(f"用户: {user_input_1}")

    response_1, memories_1 = chat_with_memory(user_id, user_input_1, memory_manager)
    print(f"\n心语: {response_1}")

    # 第二轮对话
    print("\n\n【第2轮对话】")
    print("-" * 70)
    user_input_2 = "我觉得自己快撑不住了，感觉身体也吃不消了。"
    print(f"用户: {user_input_2}")

    response_2, memories_2 = chat_with_memory(user_id, user_input_2, memory_manager)
    print(f"\n心语: {response_2}")

    # 第三轮对话（测试记忆检索）
    print("\n\n【第3轮对话 - 测试记忆检索】")
    print("-" * 70)
    user_input_3 = "项目快上线了，这周又要天天熬夜了。"
    print(f"用户: {user_input_3}")

    response_3, memories_3 = chat_with_memory(user_id, user_input_3, memory_manager)
    print(f"\n心语: {response_3}")

    # 显示检索到的记忆
    print("\n\n【检索到的历史记忆】")
    print("-" * 70)
    for i, mem in enumerate(memories_3, 1):
        print(f"{i}. {mem['text']}")
        print(f"   时间: {mem['metadata']['datetime']}")
        print(f"   情绪: {mem['metadata'].get('emotion', '未知')}")
        print()

    print("=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_memory_system()