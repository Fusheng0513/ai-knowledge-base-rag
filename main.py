import os
import pandas as pd
from pathlib import Path
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredExcelLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import time

load_dotenv()

# ========== 1. 多格式文档加载器 ==========
def load_documents(source: str) -> List[Document]:
    """支持 .txt, .pdf, .xlsx, .docx, .csv 的文档加载"""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"文件或目录不存在: {source}")

    all_docs = []
    
    if source_path.is_file():
        suffix = source_path.suffix.lower()
        print(f"正在加载: {source_path}")
        
        try:
            if suffix == '.txt':
                loader = TextLoader(source_path, encoding='utf-8')
                docs = loader.load()
            elif suffix == '.pdf':
                loader = PyPDFLoader(source_path)
                docs = loader.load()
            elif suffix == '.xlsx':
                loader = UnstructuredExcelLoader(source_path, mode="elements")
                docs = loader.load()
            elif suffix == '.docx':
                loader = Docx2txtLoader(source_path)
                docs = loader.load()
            elif suffix == '.csv':
                df = pd.read_csv(source_path, encoding='utf-8')
                docs = []
                for idx, row in df.iterrows():
                    row_text = "，".join([f"{col}: {row[col]}" for col in df.columns])
                    doc = Document(
                        page_content=row_text,
                        metadata={"source": str(source_path), "row": idx}
                    )
                    docs.append(doc)
            else:
                print(f"跳过不支持的类型: {suffix}")
                return []
            
            # 统一添加来源元数据
            for doc in docs:
                if "source" not in doc.metadata:
                    doc.metadata["source"] = str(source_path)
            all_docs.extend(docs)
            print(f"成功加载 {len(docs)} 个文档块")
            
        except Exception as e:
            print(f"加载失败: {e}")
    
    elif source_path.is_dir():
        print(f"扫描目录: {source_path}")
        for file_path in source_path.rglob("*"):
            if file_path.suffix.lower() in ['.txt', '.pdf', '.xlsx', '.docx', '.csv']:
                all_docs.extend(load_documents(str(file_path)))
    
    return all_docs

# ========== 2. 文本分块 ==========
def split_documents(docs: List[Document], chunk_size=500, chunk_overlap=100) -> List[Document]:
    """将长文档切分成适合向量化的文本块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # 每个文本块的最大字符数
        chunk_overlap=chunk_overlap, # 文本块之间的重叠字符数，保留上下文
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"已将 {len(docs)} 个文档切分为 {len(chunks)} 个文本块")
    return chunks

# ========== 3. 清洗文本块 ==========
def clean_chunks(chunks: List[Document]) -> List[Document]:
    """
    清洗文本块：
    1. 过滤掉空内容
    2. 过滤掉过短的内容（少于10个字符）
    3. 截断过长的内容（智谱API限制）
    """
    cleaned = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        
        # 过滤空内容
        if not text:
            continue
        
        # 过滤过短的内容（可能是无意义的碎片）
        if len(text) < 10:
            continue
        
        # 截断过长内容（智谱嵌入模型最大输入 token 数有限，这里保守设为 2000 字符）
        if len(text) > 2000:
            text = text[:2000]
            chunk.page_content = text
        
        cleaned.append(chunk)
    
    print(f"清洗后保留 {len(cleaned)}/{len(chunks)} 个有效文本块")
    return cleaned

# ========== 4. 分批构建 FAISS 向量库 ==========
def build_vector_store(chunks: List[Document], index_name="knowledge_base", batch_size=50):
    """
    分批构建向量库，避免 API 限制
    """
    print(f"正在初始化嵌入模型...")
    embeddings = ZhipuAIEmbeddings()
    
    print(f"正在分批构建向量索引（每批 {batch_size} 条，共 {len(chunks)} 条）...")
    
    vector_store = None
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        batch_texts = []
        batch_metadatas = []
        
        # 提取文本和元数据，同时做基本校验
        for doc in batch:
            text = doc.page_content.strip()
            if text:  # 确保非空
                batch_texts.append(text)
                batch_metadatas.append(doc.metadata)
        
        if not batch_texts:
            print(f"批次 {i//batch_size + 1} 无有效文本，跳过")
            continue
        
        print(f"处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}（{len(batch_texts)} 条）...")
        
        try:
            if vector_store is None:
                # 第一批：创建索引
                vector_store = FAISS.from_texts(
                    batch_texts, 
                    embeddings, 
                    metadatas=batch_metadatas
                )
            else:
                # 后续批次：追加到现有索引
                vector_store.add_texts(batch_texts, metadatas=batch_metadatas)
            
            # 避免请求过快，加个小延迟
            time.sleep(0.1)
            
        except Exception as e:
            print(f"批次 {i//batch_size + 1} 失败: {e}")
            # 尝试逐个添加失败批次中的文本
            print(f"尝试逐个添加该批次文本...")
            for idx, (text, metadata) in enumerate(zip(batch_texts, batch_metadatas)):
                try:
                    vector_store.add_texts([text], metadatas=[metadata])
                    print(f"第 {idx+1} 条添加成功")
                except Exception as inner_e:
                    print(f"第 {idx+1} 条添加失败: {inner_e}")
    
    if vector_store is None:
        raise ValueError("没有成功添加任何文本到向量库")
    
    # 保存到本地
    vector_store.save_local(f"./{index_name}")
    print(f"向量库已保存到 ./{index_name}，共 {vector_store.index.ntotal} 条向量")
    return vector_store

# ========== 5. 加载已有向量库 ==========
def load_vector_store(index_name="knowledge_base"):
    """加载本地向量库"""
    embeddings = ZhipuAIEmbeddings()
    return FAISS.load_local(
        f"./{index_name}", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

# ========== 6. 语义搜索 ==========
def search(vector_store, query: str, k=3):
    """搜索最相似的 k 条结果"""
    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\n搜索: '{query}'")
    print("-" * 50)
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [相似度: {score:.4f}]")
        print(f"   内容: {doc.page_content[:150]}...")
        print(f"   来源: {doc.metadata.get('source', '未知')}")
        print()
    return results

# ========== 7. 主程序 ==========
if __name__ == "__main__":
    DATA_DIR = "test_market_data.csv"
    INDEX_NAME = "my_knowledge_base"
    
    print("=" * 60)
    print("智能知识库构建系统")
    print("=" * 60)
    
    print("\n步骤1: 加载文档...")
    docs = load_documents(DATA_DIR)
    print(f"\n共加载 {len(docs)} 个文档块")
    
    if not docs:
        print("没有找到可加载的文档")
        exit()
    
    print("\n步骤2: 文本分块...")
    chunks = split_documents(docs)
    
    print("\n步骤2.5: 清洗文本块...")
    chunks = clean_chunks(chunks)
    
    if not chunks:
        print("清洗后没有有效文本块")
        exit()
    
    print("\n步骤3: 构建向量库...")
    vector_store = build_vector_store(chunks, INDEX_NAME, batch_size=50)
    
    print("\n步骤4: 测试语义搜索...")
    search(vector_store, "金融投资")
    search(vector_store, "数据")
    
    print("\n" + "=" * 60)
    print("知识库构建完成！")
    print("=" * 60)