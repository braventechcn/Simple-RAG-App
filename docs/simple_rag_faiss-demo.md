# 文件功能描述 - `simple_rag_faiss-demo.py`

## 功能
- 实现一个简单的基于检索增强生成（RAG）的流程。
- 加载 PDF 文档，分块后生成嵌入向量并存储到 FAISS 向量索引中。
- 支持用户查询的向量化检索，并调用大模型生成最终回答。

## 主要模块
- **`load_embedding_model`**: 加载本地嵌入模型（`bge-base-zh-v1.5`）。
- **`indexing_process`**: 将 PDF 文档分块并生成嵌入向量，存储到 FAISS 索引中。
- **`retrieval_process`**: 基于用户查询从 FAISS 索引中检索最相似的文本块。
- **`generate_process`**: 调用 Qwen 大模型生成最终回答。

## 技术栈
- `faiss`: 用于高效的向量检索。
- `sentence-transformers`: 加载嵌入模型并生成向量。
- `dashscope`: 调用 Qwen 大模型 API。
- `PyPDFLoader`: 加载 PDF 文档内容。

## 运行说明

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行程序
- **功能**: 测试基于 FAISS 的 RAG 流程。
- **命令**:
```bash
python3 src/simple_rag_faiss-demo.py
```