from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
CLIENT_SECRETS = 'client_secrets.json'

flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
creds = flow.run_local_server(port=0)

youtube = build('youtube', 'v3', credentials=creds)

request = youtube.videos().insert(
    part="snippet,status",
    body={
        "snippet": {
            "categoryId": "24",  # Entertainment
            "description": "Fun facts about old kids shows!",
            "title": "SpongeBob Fun Facts!"
        },
        "status": {
            "privacyStatus": "public"
        }
    },
    media_body=MediaFileUpload('video/final_video.mp4')
)
response = request.execute()
print(f"Video uploaded: {response['id']}")