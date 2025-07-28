from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_query(query: str) -> str:
    system_prompt = "Bạn là một trợ lý AI. Hãy viết lại câu hỏi ngắn gọn, rõ ràng và dễ hiểu hơn."
    user_prompt = f"Viết lại câu hỏi sau cho rõ ràng và dễ hiểu hơn:\n\n{query}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
