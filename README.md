# RAG 智能知识库

基于 LangChain + FAISS + 智谱 API 的文档检索系统。

## 功能
- 支持 .txt / .pdf / .xlsx / .docx / .csv 格式
- 自动分块、清洗、向量化
- 语义搜索

## 技术栈
- Python
- LangChain
- FAISS
- 智谱 Embedding API
- pytest

## 运行方式

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（在 .env 文件中）
ZHIPU_API_KEY=你的智谱AI密钥

# 3. 运行
python main.py
