import os

from dotenv import load_dotenv


def load_config():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set in your .env file.")

    return {"GOOGLE_API_KEY": google_api_key}
