Tải môi trường
### `pip install -r requirements.txt`
### `pip install -U sentence-transformers`

Run code
### `uvicorn main:app --reload`
Nếu lỗi nltk.data.find('tokenizers/punkt')
Thực hiện theo các bước
Bước 1: Mở Python shell trong terminal
### `python`
Bước 2: Dán và chạy từng dòng sau:
### `import nltk`
### `nltk.download('punkt')        # cái này là gốc`
### `nltk.download('punkt_tab')    # cái này khắc phục lỗi bạn gặp`
Bước 3 Run code