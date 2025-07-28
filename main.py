from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from utils.file_loader import load_file
from utils.chunking import recursive_chunking
from utils.embedding import embed_chunks
from utils.embedding import embed_text
from utils.chunking import smart_pdf_chunking
from qdrant.qdrant_client import init_qdrant
from qdrant.uploader import upload_to_qdrant
from utils.qdrant_utils import search_similar_chunks
from utils.reranker import rerank_with_cross_encoder
from utils.summarizer import summarize_context
from utils.rewriter import rewrite_query
import os
import tempfile

app = FastAPI()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # 1. Lưu file tạm
    if not file.filename.endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ .pdf, .txt, .docx")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        # 2. Load nội dung
        text = load_file(temp_path)

        # 3. Chunking
        # chunks = recursive_chunking(text)
        chunks = smart_pdf_chunking(text)

        # 4. Vector hóa
        vectors = embed_chunks(chunks)
        

        # 5. Đẩy vào Qdrant
        client = init_qdrant("my_documents")
        upload_to_qdrant(client, "my_documents", chunks, vectors)

        return {"message": f"✅ Đã upload {len(chunks)} đoạn văn bản vào Qdrant."}
    finally:
        os.remove(temp_path)
# # Version 1:Pipeline đơn giản
# @app.get("/search")
# def search(query: str = Query(..., description="Câu hỏi bạn muốn tìm")):
#     query_vector = embed_text(query)
#     results = search_similar_chunks(query_vector, top_k=5)
    
#     # # Trả v ề 5 kết quả tốt nhất
#     # context_chunks = []
#     # seen = set()
#     # total_words = 0
#     # max_words = 500  # Giới hạn context trả về

#     # for r in results:
#     #     text = r.payload.get("text", "")
#     #     if text not in seen:
#     #         words = text.split()
#     #         if total_words + len(words) > max_words:
#     #             break
#     #         context_chunks.append(" ".join(words))
#     #         total_words += len(words)
#     #         seen.add(text)

#     # context = "\n".join(context_chunks)

#     # Trả về kết quả tốt nhất
#     context = results[0].payload.get("text", "") if results else ""
#     context = context.replace("\n", " ").replace("\r", " ")
    
#     # Nếu muốn trả lời bằng LLM: gọi GPT/LLaMA/Mistral API tại đây, truyền `context + query`
#     return {
#         "query": query,
#         "context": context,
#         # "answer": call_llm(context, query)
#     }

# # Version 2: Pipeline với Reranker tăng độ chính xác cao hơn
# @app.get("/search")
# def search(query: str = Query(...)):
#     query_vector = embed_text(query)
#     results = search_similar_chunks(query_vector, top_k=10)  # lấy nhiều hơn để re-rank

#     # # Trả về 3 kết quả tốt nhất sau khi rerank
#     # reranked_results = rerank_with_cross_encoder(query, results, top_n=3)

#     # context_chunks = [r.payload.get("text", "") for r in reranked_results]
#     # context = "\n".join(context_chunks)

#     # Trả về kết quả tốt nhất sau khi rerank
#     reranked_results = rerank_with_cross_encoder(query, results, top_n=1)
#     context = reranked_results[0].payload.get("text", "").replace("\n", " ").strip()

#     return {
#         "query": query,
#         "context": context
#     }

@app.get("/search")
def search(query: str = Query(...)):
    # 1. Viết lại câu hỏi cho rõ nghĩa hơn
    rewritten_query = rewrite_query(query)

    # 2. Vector hóa câu hỏi viết lại
    query_vector = embed_text(rewritten_query)

    # 3. Tìm các đoạn văn bản gần nhất từ Qdrant
    results = search_similar_chunks(query_vector, top_k=10)

    # 4. Re-rank lại bằng Cross-Encoder để chọn ra đoạn liên quan nhất
    reranked_results = rerank_with_cross_encoder(rewritten_query, results, top_n=3)

    # 5. Tổng hợp nội dung lại để tránh quá dài
    context_chunks = [r.payload.get("text", "") for r in reranked_results]
    context = " ".join(context_chunks).replace("\n", " ").strip()
    summarized_context = summarize_context(context)

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "summarized_context": summarized_context
        # Nếu cần có thể gọi tiếp GPT để tạo câu trả lời từ context này.
    }