import os
from dotenv import load_dotenv
load_dotenv()
IBM_CLOUD_API_KEY = os.getenv('IBM_CLOUD_API_KEY')
SPACE_ID = os.getenv('SPACE_ID')
WML_URL = os.getenv('WML_URL')
if not IBM_CLOUD_API_KEY or not SPACE_ID or not WML_URL:
    raise EnvironmentError("Missing Watsonx.ai credentials in .env file")
