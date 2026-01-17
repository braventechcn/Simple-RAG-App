# 文件功能描述 - `chat_memory_bot.py`

## 功能
- 实现一个带有长期记忆功能的聊天机器人。
- 使用 Chroma 存储聊天历史，并基于用户输入检索相关记忆。
- 将检索到的记忆注入到大模型的上下文中，生成个性化回复。

## 主要模块
- **`ChatMemoryBot`**: 聊天机器人类，管理记忆存储、检索和生成回复。
- **`_load_embedding_model`**: 加载本地嵌入模型。
- **`add_message`**: 存储用户或机器人的消息到 Chroma。
- **`retrieve_memories`**: 检索与用户输入相关的历史记忆。
- **`generate`**: 调用 Qwen 大模型生成回复。

## 技术栈
- `chromadb`: 存储和检索聊天记忆。
- `sentence-transformers`: 嵌入生成。
- `openai`: 调用 Qwen API。
- `httpx`: HTTP 客户端，用于 API 调用。

## 运行说明

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
- **功能**: 测试带记忆功能的聊天机器人。
- **命令**:
```bash
python3 src/chat_memory_bot.py
```