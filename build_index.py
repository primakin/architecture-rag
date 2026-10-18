import os
import json
import uuid
from pathlib import Path
from typing import List

import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
INPUT_DIR = Path("./knowledge_base/modified")
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

tokenizer = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# Загрузка документов и чанкинг
# ---------------------------------------------------------------------------
def load_documents(input_dir: Path) -> List[Document]:
    documents = []
    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"В {input_dir} нет .md файлов")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path.resolve()),
                "title": file_path.stem,
            },
        )
        documents.append(doc)
        print(f"Загружен: {file_path.name} ({len(text)} символов)")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    all_chunks: List[Document] = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        all_chunks.extend(chunks)
        print(f"Разбит на чанки: {doc.metadata['source']} ({len(chunks)} чанков)")
    return all_chunks


# ---------------------------------------------------------------------------
# Эмбеддинги и загрузка в Chroma
# ---------------------------------------------------------------------------
def build_chroma_index(chunks: List[Document], model: SentenceTransformer) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []
    embeddings: List[List[float]] = []

    texts = [c.page_content for c in chunks]
    print(f"Генерация эмбеддингов для {len(texts)} чанков...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, vector in zip(chunks, vectors):
        chunk_id = str(uuid.uuid4())[:12]
        ids.append(chunk_id)
        documents.append(chunk.page_content)
        embeddings.append(vector.tolist())

        meta = {
            "chunk_id": chunk_id,
            "source": chunk.metadata.get("source"),
            "file_path": chunk.metadata.get("file_path"),
            "title": chunk.metadata.get("title"),
            "start_index": chunk.metadata.get("start_index"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "total_chunks": chunk.metadata.get("total_chunks"),
            "char_count": len(chunk.page_content),
        }
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Загружено в Chroma: {collection.count()} записей")
    print(f"    Коллекция: {COLLECTION_NAME}")
    print(f"    Путь: {CHROMA_DIR.resolve()}")
    return collection


# ---------------------------------------------------------------------------
# Поиск по запросу
# ---------------------------------------------------------------------------
def search_query(query: str, model: SentenceTransformer, top_k: int = 5) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_vector = model.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nЗапрос: «{query}»")
    print("=" * 70)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        similarity = 1 - dist
        print(f"\n[{rank}] Релевантность: {similarity:.4f} (distance={dist:.4f})")
        print(f"    Источник: {meta.get('source')}")
        print(f"    Заголовок: {meta.get('title')}")
        print(f"    Позиция: start_index={meta.get('start_index')}, "
              f"чанк {meta.get('chunk_index')+1}/{meta.get('total_chunks')}")
        print(f"    ID чанка: {meta.get('chunk_id')}")
        print(f"    Путь: {meta.get('file_path')}")
        print(f"    Текст: {doc[:250]}...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print(f"Модель: {EMBEDDING_MODEL_NAME}")
    print("=" * 70)

    # 1. Загрузка и чанкинг
    documents = load_documents(INPUT_DIR)
    chunks = chunk_documents(documents)

    # 2. Модель эмбеддингов
    print(f"\nЗагрузка модели {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3. Индексация в Chroma
    build_chroma_index(chunks, model)

    # 4. Пример поиска
    search_query(
        query="Carolyn Timothy",
        model=model,
        top_k=3,
    )


if __name__ == "__main__":
    main()