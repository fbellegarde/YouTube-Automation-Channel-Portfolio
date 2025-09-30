from gtts import gTTS
from moviepy.editor import TextClip, concatenate_videoclips, ImageClip
import requests  # For fetching images (public domain)
from io import BytesIO
from PIL import Image

# Read script
with open('scripts/video_script.txt', 'r') as f:
    script = f.read()

# TTS Voiceover
tts = gTTS(script, lang='en')
tts.save('video/voiceover.mp3')

# Fetch dynamic visuals (safe public images, e.g., from Unsplash API - free, but for demo use placeholders)
# Example: Fetch image (replace with real API key if needed, but free tier)
urls = ["https://source.unsplash.com/random/1920x1080/?cartoon", "https://source.unsplash.com/random/1920x1080/?kids", "https://source.unsplash.com/random/1920x1080/?tv"]  # Safe HTTPS, no viruses

clips = []
for url in urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save('video/temp_img.jpg')
    clip = ImageClip('video/temp_img.jpg', duration=5).set_audio(None)  # Add text overlay
    text = TextClip("Fun Fact!", fontsize=70, color='white').set_position('center').set_duration(5)
    clips.append(clip.set_duration(5).overlay(text))

# Assemble video
final = concatenate_videoclips(clips)
final.write_videofile('video/final_video.mp4', fps=24, audio='video/voiceover.mp3')

print("Video created: final_video.mp4")