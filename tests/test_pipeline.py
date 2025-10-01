import pytest
from moviepy.editor import VideoFileClip
import os
def test_video_durations():
    for video in os.listdir('videos'):
        if video.endswith('.mp4'):
            clip = VideoFileClip(f'videos/{video}')
            assert 240 <= clip.duration <= 300, f"Video {video} duration {clip.duration}s is not 4-5 mins"
            clip.close()