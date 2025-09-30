import os
import logging
import pickle
import json
from typing import Optional
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import boto3

# Load environment variables
load_dotenv()

# Configure logging (consistent with extract_data.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_auth.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress Google API discovery cache warning
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly'
]

def get_client_secret(secret_name: str, region_name: str) -> Optional[dict]:
    """
    Fetch client secret from AWS Secrets Manager or return None if not available.
    
    Args:
        secret_name (str): Name of the secret in AWS Secrets Manager.
        region_name (str): AWS region.
    
    Returns:
        Optional[dict]: Client secret JSON or None if fetch fails.
    """
    try:
        client = boto3.client('secretsmanager', region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        logger.info(f"Fetched secret {secret_name} from AWS Secrets Manager")
        # Save to local file for OAuth flow
        credentials_dir = os.path.join(os.path.dirname(__file__), '..', 'credentials')
        os.makedirs(credentials_dir, exist_ok=True)
        credentials_file = os.path.join(credentials_dir, 'client_secrets.json')
        with open(credentials_file, 'w') as f:
            json.dump(secret, f)
        logger.info(f"Saved client_secrets.json to {credentials_file}")
        return secret
    except Exception as e:
        logger.error(f"Failed to fetch secret {secret_name}: {e}")
        return None

def get_youtube_service() -> Optional[object]:
    """
    Authenticate with YouTube Data API v3 and return the service client.
    
    Returns:
        Optional[object]: YouTube API service client or None if authentication fails.
    """
    credentials_dir = os.path.join(os.path.dirname(__file__), '..', 'credentials')
    credentials_file = os.path.join(credentials_dir, 'client_secrets.json')
    token_file = os.path.join(credentials_dir, 'token.pickle')
    
    # Check if credentials file exists; try AWS Secrets Manager if not
    if not os.path.exists(credentials_file):
        secret_name = os.getenv('SECRETS_NAME', 'YouTubeAutomationClientSecret')
        region_name = os.getenv('AWS_REGION', 'us-east-1')
        get_client_secret(secret_name, region_name)
        if not os.path.exists(credentials_file):
            logger.error(f"Credentials file not found at {credentials_file}")
            logger.info("Download client_secrets.json from Google Cloud Console and save to credentials/")
            return None
    
    credentials = None
    # Load existing token if available
    if os.path.exists(token_file):
        try:
            with open(token_file, 'rb') as token:
                credentials = pickle.load(token)
            logger.info(f"Loaded OAuth token with scopes: {credentials.scopes}")
            if not all(scope in credentials.scopes for scope in SCOPES):
                logger.warning(f"Token scopes {credentials.scopes} do not include all required scopes {SCOPES}")
                os.remove(token_file)
                credentials = None
        except Exception as e:
            logger.error(f"Failed to load token.pickle: {e}")
            credentials = None
    
    # Refresh credentials if expired
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            logger.info("Refreshed OAuth token")
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            credentials = None
    
    # Run OAuth flow if no valid credentials
    if not credentials or not credentials.valid:
        try:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            credentials = flow.run_local_server(port=0)
            with open(token_file, 'wb') as token:
                pickle.dump(credentials, token)
            logger.info(f"Authenticated and saved OAuth token with scopes: {credentials.scopes}")
            if not all(scope in credentials.scopes for scope in SCOPES):
                logger.error(f"Authenticated token scopes {credentials.scopes} do not include all required scopes {SCOPES}")
                return None
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            return None
    
    try:
        youtube = build('youtube', 'v3', credentials=credentials)
        logger.info("Successfully built YouTube API service client")
        return youtube
    except Exception as e:
        logger.error(f"Failed to build YouTube API client: {e}")
        return None

def test_youtube_service(youtube):
    """
    Test the YouTube service by fetching the authenticated user's channel.
    
    Args:
        youtube: YouTube API service client.
    """
    if not youtube:
        logger.error("No YouTube service client available")
        return
    try:
        request = youtube.channels().list(part='snippet', mine=True)
        response = request.execute()
        channel_title = response['items'][0]['snippet']['title']
        logger.info(f"Authenticated as YouTube channel: {channel_title}")
    except Exception as e:
        logger.error(f"Failed to fetch channel info: {e}")

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'credentials'), exist_ok=True)
    youtube_service = get_youtube_service()
    test_youtube_service(youtube_service)