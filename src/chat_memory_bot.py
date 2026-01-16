"""
Chat bot with simple long-term memory using Chroma as vector store and Qwen via OpenAI-compatible API.
- Stores chat history with timestamps and simple emotion tags.
- Retrieves semantically relevant past messages for new turns.
- Injects retrieved memories as context into the LLM for personalized replies.

Requirements:
- chromadb
- sentence-transformers (loads local bge-base-zh-v1.5 by default)
- openai (>=1.0)

Environment variables:
- OPENAI_API_KEY: API key for Qwen OpenAI-compatible endpoint.
- OPENAI_BASE_URL: Base URL for the Qwen endpoint (e.g. https://dashscope.aliyuncs.com/compatible-mode/v1 for Qwen).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Dict, Any

import chromadb
import numpy as np
import httpx
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class ChatMemoryBot:
    """A simple chat bot with vectorized long-term memory stored in Chroma."""

    def __init__(
        self,
        model_name: str = "qwen-plus",
        embedding_model_path: str = "bge-base-zh-v1.5",
        chroma_path: str = ".chroma_memory",
        collection_name: str = "chat_memories",
        top_k: int = 5,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.embedding_model = self._load_embedding_model(embedding_model_path)
        self.embedding_fn = self._build_embedding_function(embedding_model_path)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self.top_k = top_k
        self.temperature = temperature
        # http_client with trust_env=False bypasses system proxy env vars that would require socks extras.
        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(trust_env=False),
        )

    def _load_embedding_model(self, path: str) -> SentenceTransformer:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            logging.info("Loading local embedding model: %s", abs_path)
            model = SentenceTransformer(abs_path)
        else:
            logging.info("Local model not found, downloading: %s", path)
            model = SentenceTransformer("BAAI/bge-base-zh-v1.5", cache_folder=abs_path)
        logging.info("Embedding model max length: %s", model.max_seq_length)
        return model

    def _build_embedding_function(self, model_id: str):
        # Wrap the SentenceTransformer with a name() method as expected by Chroma.
        model = self.embedding_model

        class _ChromaEmbeddingWrapper:
            def __init__(self, name: str):
                self._name = name

            def __call__(self, input: Any) -> List[List[float]]:  # Chroma expects param named "input"
                texts = input if isinstance(input, list) else [input]
                vectors = model.encode(texts, normalize_embeddings=True)
                return np.atleast_2d(vectors).tolist()

            def embed_documents(self, input: Any) -> List[List[float]]:
                return self(input)

            def embed_query(self, input: Any) -> List[List[float]]:
                return self(input)

            def name(self) -> str:
                return self._name

        return _ChromaEmbeddingWrapper(name=os.path.abspath(model_id))

    @staticmethod
    def _tag_emotion(text: str) -> str:
        """Lightweight heuristic emotion tagger to avoid extra dependencies."""
        positive_cues = ["great", "good", "happy", "love", "喜欢", "满意"]
        negative_cues = ["bad", "sad", "angry", "hate", "讨厌", "生气", "失望"]
        lower = text.lower()
        if any(token in lower for token in positive_cues):
            return "positive"
        if any(token in lower for token in negative_cues):
            return "negative"
        return "neutral"

    def add_message(self, role: str, content: str) -> None:
        """Store a message with timestamp and emotion tag into Chroma."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        emotion = self._tag_emotion(content)
        doc_id = f"{timestamp}-{role}-{len(content)}"
        metadata = {"role": role, "timestamp": timestamp, "emotion": emotion}
        self.collection.add(documents=[content], metadatas=[metadata], ids=[doc_id])
        logging.info("Saved %s message with emotion %s", role, emotion)

    def retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant past messages from memory."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=self.top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        memories = []
        for doc, meta, distance in zip(docs, metas, distances):
            memories.append({"content": doc, "metadata": meta, "score": distance})
        return memories

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "(no prior memories)"
        lines = []
        for idx, mem in enumerate(memories, 1):
            meta = mem["metadata"]
            ts = meta.get("timestamp", "?")
            emo = meta.get("emotion", "?")
            role = meta.get("role", "?")
            lines.append(
                f"Memory {idx} | {ts} | {role} | {emo}\n{mem['content']}\n"
            )
        return "\n".join(lines)

    def build_messages(self, user_input: str, memories: List[Dict[str, Any]]):
        memory_block = self._format_memories(memories)
        system_prompt = (
            "You are a helpful, concise assistant. Use the provided chat memories to personalize your reply. "
            "If memories are not relevant, rely on the latest user input. Respond in Chinese when user speaks Chinese."
        )
        memory_prompt = f"Relevant memories:\n{memory_block}\n"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": memory_prompt},
            {"role": "user", "content": user_input},
        ]

    def generate(self, user_input: str) -> str:
        memories = self.retrieve_memories(user_input)
        messages = self.build_messages(user_input, memories)
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        # Save conversation turn into memory
        self.add_message("user", user_input)
        self.add_message("assistant", content)
        return content


def demo_conversation():
    """Demonstrate multi-turn memory retrieval in a single run."""
    bot = ChatMemoryBot(collection_name="demo_conversation", top_k=3)

    turns = [
        "最近工作压力好大，每天都加班到很晚。",
        "我觉得自己快撑不住了，感觉身体也吃不消了。",
        "项目快上线了，这周又要天天熬夜了。",
    ]

    for idx, user_input in enumerate(turns, 1):
        reply = bot.generate(user_input)
        print("-" * 70)
        print(f"第{idx}轮 用户: {user_input}")
        print(f"第{idx}轮 心语: {reply}\n")

    # Show memories retrieved for the last query
    memories = bot.retrieve_memories(turns[-1])
    print("检索到的历史记忆:")
    print("-" * 70)
    for i, mem in enumerate(memories, 1):
        meta = mem.get("metadata", {})
        print(f"{i}. {meta.get('timestamp', '?')} | {meta.get('role', '?')} | {meta.get('emotion', '?')}")
        print(mem.get("content", ""))
        print(f"score={mem.get('score')}")
        print()


if __name__ == "__main__":
    demo_conversation()
