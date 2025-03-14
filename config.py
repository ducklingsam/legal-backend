import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    URL_SEARCH = os.getenv("URL_SEARCH")
    API_KEY = os.getenv("API_KEY")
    EXAMPLE_QUERY = {
        "q": "молоко",
        "limit": 20
    }
    EXAMPLE_HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }