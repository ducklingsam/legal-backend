import os

from dotenv import load_dotenv

load_dotenv()

class Config:
    URL_SEARCH = os.getenv("URL_SEARCH")
    API_KEY = os.getenv("API_KEY")
    # LLAMA_KEY = os.getenv("LLAMA_KEY")
    EXAMPLE_QUERY = {
        "q": "молоко",
        "limit": 20
    }
    EXAMPLE_HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    DATABASE_URI = os.getenv("DATABASE_URI")
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = os.getenv("SMTP_PORT")
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    EMAIL_FROM = os.getenv("EMAIL_FROM")