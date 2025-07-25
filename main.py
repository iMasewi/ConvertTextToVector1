from fastapi import FastAPI, UploadFile, File, HTTPException
from utils.file_loader import load_file
from utils.chunking import recursive_chunking
from utils.embedding import embed_chunks
from utils.chunking import smart_pdf_chunking
from qdrant.qdrant_client import init_qdrant
from qdrant.uploader import upload_to_qdrant
import os
import tempfile

app = FastAPI()

@app.post("/upload/")
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
        client = init_qdrant("documents")
        upload_to_qdrant(client, "documents", chunks, vectors)

        return vectors.tolist()
    finally:
        os.remove(temp_path)
