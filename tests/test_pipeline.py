import pytest
from moviepy.editor import VideoFileClip
import os

def test_video_durations():
    for video in os.listdir('videos'):
        if video.endswith('.mp4'):
            clip = VideoFileClip(f'videos/{video}')
            assert 240 <= clip.duration <= 300, f"Video {video} duration {clip.duration}s is not 4-5 mins"
            clip.close()

def test_video_count():
    videos = [v for v in os.listdir('videos') if v.endswith('.mp4')]
    assert len(videos) == 7, f"Expected 7 videos, found {len(videos)}"

def test_no_none_type_errors():
    with open('logs/render.log', 'r') as f:
        log_content = f.read()
        assert "'NoneType' object has no attribute 'get_frame'" not in log_content, "NoneType errors found in log"