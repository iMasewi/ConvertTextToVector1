from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_context(context: str, max_tokens: int = 200) -> str:
    system_prompt = "Bạn là một trợ lý AI. Hãy tóm tắt đoạn văn dưới đây ngắn gọn, đầy đủ ý chính."
    user_prompt = f"Hãy tóm tắt ngắn gọn đoạn văn sau:\n\n{context}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()
