import pytest
import tempfile
from pathlib import Path
from langchain_core.documents import Document
from smart_document_loader import load_documents, split_documents, clean_chunks

# ========== 测试用例1：测试文档加载 ==========
def test_load_single_csv():
    """测试：加载一个有效的CSV文件应该返回非空文档列表"""
    # 准备：创建一个临时CSV文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("name,age\n张三,25\n李四,30")
        temp_path = f.name
    
    # 执行：调用被测试的函数
    docs = load_documents(temp_path)
    
    # 断言：验证结果
    assert len(docs) > 0, "应该至少加载到一个文档"
    assert all(isinstance(doc, Document) for doc in docs)
    assert "张三" in docs[0].page_content or "李四" in docs[0].page_content
    
    # 清理
    Path(temp_path).unlink()

def test_load_nonexistent_file():
    """测试：加载不存在的文件应该抛出异常或返回空列表"""
    with pytest.raises(FileNotFoundError):  # 期望函数抛出异常
        load_documents("i_dont_exist.csv")

def test_load_empty_csv():
    """测试：加载空的CSV文件应该返回空列表"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("")  # 空内容
        temp_path = f.name
    
    docs = load_documents(temp_path)
    assert docs == []  # 或者 len(docs) == 0
    
    Path(temp_path).unlink()

# ========== 测试用例2：测试文本分块 ==========
def test_split_documents_basic():
    """测试：正常的文档分块"""
    
    # 准备：创建一个长文档
    long_text = "这是第一句话。" * 100  # 500+ 字符
    doc = Document(page_content=long_text, metadata={"source": "test"})
    
    # 执行：分块
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    
    # 断言：
    assert len(chunks) > 1, "长文档应该被切成多个块"
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)

def test_split_documents_empty():
    """测试：空文档列表应该返回空列表"""
    chunks = split_documents([])
    assert chunks == []

# ========== 测试用例3：测试清洗函数 ==========
def test_clean_chunks_remove_empty():
    """测试：清洗应该移除空内容的块"""
    
    chunks = [
        Document(page_content="   ", metadata={}),  # 只有空格
        Document(page_content="这真的是一个有效内容啊", metadata={}),
    ]
    
    cleaned = clean_chunks(chunks)
    assert len(cleaned) == 1
    assert cleaned[0].page_content == "这真的是一个有效内容啊"

def test_clean_chunks_truncate_long():
    """测试：过长的块应该被截断"""
    
    long_text = "a" * 3000
    chunks = [Document(page_content=long_text, metadata={})]
    
    cleaned = clean_chunks(chunks)
    assert len(cleaned[0].page_content) <= 2000

# ========== 运行所有测试 ==========
if __name__ == "__main__":
    pytest.main([__file__, "-v"])