from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
# Đảm bảo punkt được download trước khi import sent_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize
# 1. Fixed-size Chunking (Chunk theo độ dài cố định)
def fixed_chunking(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
# 2. Sentence-based Chunking (Chunk theo câu)
def sentence_chunking(text, max_tokens=512):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence.split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_length = 0
        current_chunk += sentence + " "
        current_length += len(sentence.split())
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
# 3. Recursive Text Splitting (Tách đệ quy theo cấu trúc logic)
def recursive_chunking(text, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
# 4. Sliding Window Chunking (Cửa sổ trượt có overlap)
def sliding_window_chunking(sentences, window_size=3, stride=1):
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        chunk = " ".join(sentences[i:i+window_size])
        chunks.append(chunk)
    return chunks
# 5. Paragraph-based Chunking (Chunk theo đoạn)
def paragraph_chunking(text, max_tokens=512):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for para in paragraphs:
        tokens = len(para.split())
        if current_tokens + tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_tokens = 0
        current_chunk += para + "\n\n"
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
# Chunking kết hợp 2 4 5
def hybrid_chunking(text: str, 
                    window_size: int = 3, 
                    stride: int = 1, 
                    max_chunk_len: int = 300):
    """
    - Bước 1: Cắt đoạn theo đoạn văn (paragraph)
    - Bước 2: Với mỗi đoạn, chia theo câu (sentence)
    - Bước 3: Áp dụng sliding window trên các câu
    """
    import nltk
    from nltk.tokenize import sent_tokenize

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    all_chunks = []

    for para in paragraphs:
        sentences = sent_tokenize(para)
        if len(sentences) <= window_size:
            chunk = " ".join(sentences)
            if len(chunk) <= max_chunk_len:
                all_chunks.append(chunk)
            else:
                # fallback nếu chunk quá dài
                all_chunks.extend(sliding_window_chunking(sentences, window_size, stride))
        else:
            all_chunks.extend(sliding_window_chunking(sentences, window_size, stride))

    return all_chunks

def sliding_window(sentences, window_size=3, stride=1):
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        chunk = " ".join(sentences[i:i + window_size])
        chunks.append(chunk)
    return chunks

def smart_pdf_chunking(text: str,
                       window_size: int = 3,
                       stride: int = 1,
                       max_chunk_tokens: int = 300,
                       recursive_chunk_size: int = 300,
                       recursive_overlap: int = 50):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        sentences = sent_tokenize(para)
        if len(sentences) <= window_size:
            chunk = " ".join(sentences)
            if len(chunk.split()) <= max_chunk_tokens:
                chunks.append(chunk)
            else:
                chunks.extend(
                    recursive_chunking(chunk, recursive_chunk_size, recursive_overlap)
                )
        else:
            sliding_chunks = sliding_window(sentences, window_size, stride)
            for chunk in sliding_chunks:
                if len(chunk.split()) <= max_chunk_tokens:
                    chunks.append(chunk)
                else:
                    chunks.extend(
                        recursive_chunking(chunk, recursive_chunk_size, recursive_overlap)
                    )

    return chunks