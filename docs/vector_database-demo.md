# 文件功能描述 - `vector_database-demo.py`

## 功能
- 提供一个基于 Chroma 向量数据库的记忆管理系统。
- 支持用户聊天记忆的存储、检索、删除和统计。
- 集成了本地嵌入模型（如 `bge-base-zh-v1.5`）和 OpenAI 的嵌入 API，用于生成文本嵌入向量。

## 主要模块
- **`EmbeddingService`**: 提供文本嵌入服务，支持本地模型和远程 API。
- **`ChromaMemoryManager`**: 管理用户聊天记忆，包括存储、检索、删除和统计功能。
- **`chat_with_memory`**: 模拟聊天流程，结合历史记忆生成个性化回复。

## 技术栈
- `chromadb`: 用于存储和检索嵌入向量。
- `transformers`: 加载和使用本地嵌入模型。
- `torch`: 支持本地模型的推理。
- `dotenv`: 加载环境变量。

## 运行说明

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
- **功能**: 测试 Chroma 记忆管理系统。
- **命令**:
```bash
python3 src/vector_database-demo.py
```