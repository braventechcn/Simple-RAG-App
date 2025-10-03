
from langchain_community.document_loaders import PyPDFLoader # PDF文档提取
from langchain_text_splitters import RecursiveCharacterTextSplitter # 文档拆分chunk
from sentence_transformers import SentenceTransformer # 加载和使用Embedding模型
import faiss # Faiss向量库
# 注意: Faiss 目前只支持 NumPy 1.x，如果环境中安装的是 NumPy 2.x，需要 pip install "numpy<2"
import numpy as np # 处理嵌入向量数据，用于Faiss向量检索
import dashscope #调用Qwen大模型
from dashscope import Generation
from http import HTTPStatus #检查与Qwen模型HTTP请求状态
import os # 引入操作系统库，后续配置环境变量与获得当前文件路径使用
import logging # 日志库

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 不使用分词并行化操作, 避免多线程或多进程环境中运行多个模型引发冲突或死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_embedding_model():
    """
    Load the embedding model 'bge-base-zh-v1.5' from a local path.
    Returns: Returns the loaded embedding model.
    """
    logging.info("Loading Embedding Model 'bge-base-zh-v1.5'...")
    # Sentence-Transformers（又称 SBERT）是一个 Python 库 / 框架，用于将句子 / 段落 / 
    #   文本映射为固定维度的向量表示（即 embeddings），使得语义相近的句子在向量空间中更接近。
    # SentenceTransformer 读取绝对路径下的 bge-base-zh-v1.5 模型，非下载
    # os.path.abspath() 会根据当前工作目录，把这个相对路径变成完整的绝对路径
    embedding_model = SentenceTransformer(os.path.abspath('bge-base-zh-v1.5'))
    logging.info(f"The Model 'bge-base-zh-v1.5' Max Input Length: {embedding_model.max_seq_length}")
    logging.info("For the usage details of Embedding Model 'bge-base-zh-v1.5', please refer to 'https://huggingface.co/BAAI/bge-base-zh-v1.5#using-sentence-transformers'")
    return embedding_model


def indexing_process(pdf_file, embedding_model):
    """
    Indexing process: Load the PDF file, split it into chunks, compute embedding vectors, 
    and store them in a FAISS index (in-memory).
    Args:
        pdf_file: Path to the PDF file
        embedding_model: Pre-loaded embedding model
    Returns:
        Returns the FAISS embedding vector index and the list of original text chunks.
    """
    logging.info("Starting the indexing process...")
    
    # Load the PDF file using PyPDFLoader, ignoring image extraction.
    pdf_loader = PyPDFLoader(pdf_file, extract_images=False)
    # Configure the parameters for RecursiveCharacterTextSplitter.
    #  - Each text chunk has a size of 512 characters (not tokens), 
    #  - and adjacent chunks have an overlap of 128 characters (not tokens).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128
    )
    # Load the PDF document and extract text content from all pages.
    # - pdf_loader.load() returns a list, where each element corresponds to one page of the PDF.
    # - Each element is usually an object (such as Document) containing the text content of that page.
    pdf_content_list = pdf_loader.load()
    # Join the text content of each page with newline characters '\n' to form the complete text of the PDF document
    pdf_text = "\n".join([page.page_content for page in pdf_content_list])
    logging.info(f"The total character count of the PDF document: {len(pdf_text)}")

    # Split the PDF document text into chunks.
    chunks = text_splitter.split_text(pdf_text)
    logging.info(f"The number of split text chunks: {len(chunks)}")

    # Convert text chunks to embedding vectors.
    # - The 'normalize_embeddings' parameter indicates whether to normalize the embedding vectors 
    # - for accurate similarity calculation.
    embeddings = []
    for chunk in chunks:
        embedding = embedding_model.encode(chunk, normalize_embeddings=True)
        embeddings.append(embedding)

    logging.info("Completed converting text chunks to embedding vectors.")

    # Convert the list of embedding vectors to a numpy array.
    # - 将多个 chunk 的嵌入向量（Python list，每个元素是一个向量）转换为一个二维 numpy.ndarray。
    # - FAISS 要求批量向量以 numpy 数组形式传入。
    embeddings_np = np.array(embeddings)

    # Get the dimension of the embedding vectors (length of each vector)
    # - 获取嵌入向量的维度 D（每个向量的长度），用于指定 FAISS 索引的向量空间维度。
    dimension = embeddings_np.shape[1]

    # Create a FAISS index using cosine similarity
    # - 创建一个 FAISS 平索引（flat index），以内积（Inner Product）作为相似度度量。索引初始为空，还没有存储任何向量。
    # - 余弦相似度与内积相似度在向量归一化后是等价的，因此这里使用 IndexFlatIP 来实现基于余弦相似度的检索。
    index = faiss.IndexFlatIP(dimension)

    # Add all embedding vectors to the FAISS index for later similarity search
    # - 将所有嵌入向量批量加入到索引中。索引内部会保存这些向量，后续可用于高效的相似性检索。
    index.add(embeddings_np)

    logging.info("Indexing process completed.")

    return index, chunks

def retrieval_process(query, index, chunks, embedding_model, top_k=3):
    """
    Retrieval process: Convert the user query to an embedding vector,
    and retrieve the top k most similar text chunks from the Faiss index.
    Args:
        query: User query string
        index: Established Faiss vector index
        chunks: List of original text chunk contents
        embedding_model: Pre-loaded embedding model
        top_k: Number of top similar results to return
    Returns:
        Returns the most similar text chunks.
    """
    logging.info("Starting the retrieval process...")    
    
    # Convert the user query to an embedding vector.
    # - The 'normalize_embeddings' parameter indicates whether to normalize the embedding vector
    # - for accurate similarity calculation.
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    # Convert the query embedding vector to a numpy array.
    query_embedding = np.array([query_embedding])

    # （1）可以注意到，上面查询语句向量化与文档块向量化是一样的流程，都是调用 embedding_model.encode() 方法；
    #       且都是使用 normalize_embeddings 参数进行归一化处理。
    # （2）这里使用 query_embedding = np.array([query_embedding])，是因为 Faiss 的 search() 方法要求输入是二维 numpy 数组，
    #       即使只有一个查询向量，也需要将其包装成二维数组的形式

    # Use the query_embedding to search the Faiss index for the top_k most similar results.
    # This returns a list of distances (similarity scores) and indices of the most similar text chunks.
    distances, indices = index.search(query_embedding, top_k)
    # distances：二维数组（numpy.ndarray），形状为 (查询数量, top_k)
    # - 每个元素表示查询向量与索引库中前 top_k 个最相似向量的相似度分数（如内积或余弦相似度）。
    # indices：二维数组（numpy.ndarray），形状为 (查询数量, top_k)
    # - 每个元素是对应最相似向量在索引库中的下标（即向量的编号）。

    logging.info(f"Query: {query}")
    logging.info(f"Top {top_k} most similar text chunks:")

    # Store the most similar text chunks in a results list.
    results = []
    for i in range(top_k):
        # Get the original content of the similar text chunk
        # - The indices[0][i] gives the index of the i-th most similar chunk in the original chunks list.
        result_chunk = chunks[indices[0][i]]
        logging.info(f"Chunk {i}:\n{result_chunk}")
        # Get the similarity score between the query and this chunk
        result_distance = distances[0][i]
        logging.info(f"Similarity score: {result_distance}")
        # Store the similar text chunk in the results list
        results.append(result_chunk)

    logging.info("Retrieval process completed.")
    return results

def generate_process(query, chunks):
    """
    Generation process: Call the Qwen large model cloud API to generate the final response 
    based on the query and text chunks.
    Args:
        query: User query string
        chunks: Relevant text chunks obtained from the retrieval process
    Returns:
        Returns the generated response content.
    """
    # Set the specific Qwen model and corresponding API key for calling, obtained from the BAILIAN platform of Alibaba Cloud.
    # For API usage examples, please refer to:
    # - https://bailian.console.aliyun.com/?spm=5176.21213303.nav-v2-dropdown-menu-0.d_main_0_0_3.3ac72f3dBE8Pbb&tab=model&scm=20140722.M_10852063._.V_1#/model-market/detail/qwen-turbo
    
    # Create the prompt for the generation model, including the user query and retrieved context
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Reference context: {i+1}: \n{chunk}\n\n"

    # Create the prompt for the generation model, including the user query and retrieved context
    prompt = f"Answer questions according to reference documents: {query}\n\n{context}"
    
    # Prepare the request messages, using the prompt as input
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, # Add System role to define assistant identity, but it's not mandatory
        {'role': 'user', 'content': prompt}
    ]
    logging.info(f"Prompt for generation model: {prompt}")
    # Call the large-model API cloud service to generate a response
    try:
        # # 版本一：直接一次性获得完整输出（含思考过程）
        # response = Generation.call(
        #     api_key=os.getenv("DASHSCOPE_API_KEY"), # Get your API key from environment variable, if do not have, please set it in your environment
        #     model="qwen-turbo",                     # Replace with your actual model name
        #     messages=messages,                      # The input messages for the model
        #     result_format='message',                # Set the return format to "message"
        #     enable_thinking=True,                   # Enable thinking
        # )
        # # Initialize a variable to store the generated response content
        # generated_response = ""
        # logging.info("Generation process started:")
        # if response.status_code == HTTPStatus.OK:
        #     # Print the thinking process
        #     logging.info("=" * 20 + "Thinking Process" + "=" * 20)
        #     logging.info(response.output.choices[0].message.reasoning_content)
        #     # Print the complete response
        #     logging.info("=" * 20 + "Complete Response" + "=" * 20)
        #     content = response.output.choices[0].message.content
        #     generated_response += content
        #     logging.info(content)
        # else:
        #     logging.error(f"HTTP Return Code: {response.status_code}")
        #     logging.error(f"Error Code: {response.code}")
        #     logging.error(f"Error Message: {response.message}")
        #     return None  # 请求失败时返回 None
        # logging.info("Generation process completed.")
        # return generated_response
        # 版本二：流式获取增量输出
        responses = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"), # Get your API key from environment variable, if do not have, please set it in your environment
            model="qwen-turbo",                     # Replace with your actual model name
            messages=messages,                      # The input messages for the model
            result_format='message',                # Set the return format to "message"
            stream=True,                            # Enable streaming output
            incremental_output=True,                # Get streaming incremental output
        )
        # Initialize a variable to store the generated response content
        generated_response = ""
        logging.info("Generation process started:")
        # Gradually obtain and process the model's incremental output
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                # Print the complete response
                content = response.output.choices[0].message.content
                generated_response += content
                print(content, end='')  # 实时输出模型生成的内容
            else:
                logging.error(f"HTTP Return Code: {response.status_code}")
                logging.error(f"Error Code: {response.code}")
                logging.error(f"Error Message: {response.message}")
                return None  # 请求失败时返回 None
        logging.info("Generation process completed.")
        return generated_response
    except Exception as e:
        logging.error(f"Error occurred during model generation: {e}")
        return None

def main():
    logging.info("The RAG Process Start.")
    
    # Load the embedding model 'bge-base-zh-v1.5'.
    embedding_model = load_embedding_model()

    # Indexing process: 
    # - Load the PDF file, split it into chunks, compute embedding vectors, 
    #   and store them in a FAISS index (in-memory).
    index, chunks = indexing_process(
        'source_file/test_file__digital_transformation.pdf', 
        embedding_model
    )
    
    # User query input
    query="报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    
    # Retrieval process: 
    # - Convert the user query to an embedding vector,
    #   and retrieve the top k most similar text chunks from the Faiss index.
    #   - Here, top_k is set to 3 by default.
    #   - You can modify the top_k parameter in the retrieval_process function call if needed.
    retrieval_chunks = retrieval_process(query, index, chunks, embedding_model)

    # Generation process: 
    # - Call the Qwen large model cloud API to generate the final response based on the query and text chunks.
    generate_process(query, retrieval_chunks)

    logging.info("The RAG Process End.")

if __name__ == "__main__":
    main()
    


