# Simple-RAG-App

## 01 开发环境与技术库

采用 Python 编程语言，Python 版本 3.8 及以上(基于 Python 3.10 构建)，运行于 Linux 操作系统下

结合上述的技术框架与选型，我们的开发环境需要按照以下步骤进行准备：

1. **创建并激活虚拟环境：**
    
    虚拟环境隔离项目依赖，避免项目之间冲突
    
    ```bash
    # 创建 RAG 项目文件夹并进入文件夹
    mkdir Simple-RAG-App
    cd Simple-RAG-App/
    # 在 RAG 项目文件夹中创建名为 simple_rag_env 的虚拟环境
    python3.10 -m venv simple_rag_env
    # 激活虚拟环境
    source simple_rag_env/bin/activate
    ```
    
2. **安装技术依赖库**：
    
    首先，升级 pip 版本，以确保兼容性，在命令行中执行以下指令：
    
    ```bash
    # 升级pip版本以确保兼容性
    pip install --upgrade pip
    ```
    
    安装相关技术依赖库：
    
    ```bash
    # GPU 版本
    pip install langchain langchain_community pypdf sentence-transformers faiss-gpu dashscope
    # CPU 版本
    pip install langchain langchain_community pypdf sentence-transformers faiss-cpu dashscope
    # Faiss 目前只支持 NumPy 1.x，如果环境中安装的是 NumPy 2.x，需要 pip install "numpy<2"
    pip install "numpy<2"
    ```
    
    其中：
    
    - **langchain：**用于构建基于大语言模型（LLM）的应用，支持链式调用、工具集成、对话管理等，是 AI 应用开发的主流框架之一。
    - **langchain_community：**LangChain 社区扩展包，包含更多第三方集成、数据连接器和工具，丰富 LangChain 的生态能力。
    - **pypdf：**用于读取、解析和操作 PDF 文件，常用于文档处理、内容抽取等场景。
    - **sentence-transformers：**提供多种预训练的句子/文本向量模型，支持文本语义检索、聚类、相似度计算等 NLP 任务。
    - **faiss-gpu：**Facebook AI Similarity Search，支持高效的向量检索和聚类，GPU 版本可加速大规模向量计算，常用于语义搜索、推荐系统。
    - **dashscope：**阿里云达摩院 DashScope 官方 SDK，支持调用 Qwen 等大模型 API，适用于文本生成、对话、推理等 AI 服务。
    
    如果无法连接，可以使用国内镜像站点，在命令行中执行以下指令：
    
    ```bash
    # GPU 版本
    pip install langchain langchain_community pypdf sentence-transformers faiss-gpu 
    dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    # CPU 版本
    pip install langchain langchain_community pypdf sentence-transformers faiss-cpu 
    dashscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```