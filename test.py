from moviepy.audio.AudioClip import AudioClip, CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

from moviepy.editor import ColorClip, concatenate_videoclips, VideoFileClip
import moviepy.audio.fx.all as afx

from movie_builder.readers import GifReader, VideoReader, ImageReader, TextReader, AudioReader
from movie_builder import VideoBuilder


gif_path = "tmp/gif.gif"

video_path = "tmp/yi.mp4"

jpg_path = "tmp/bg.jpg"
png_path = "tmp/face.png"

infer_path = "tmp/video.mp4"
infer_mask_path = "tmp/mask.mp4"

audio_path = "./tmp/voice.wav"
mv_path = "./tmp/mv.mp4"


gif_reader = GifReader(
    file_path=gif_path,
    width=1149,
    height=731,
    top=100,
    left=100,
    angle=0,
    opacity=100,
    flip=0,
    start_at=3000,
    end_at=6000,
    layer=1
)

video_reader = VideoReader(
    file_path=video_path,
    width=1149,
    height=731,
    top=100,
    left=100,
    angle=0,
    opacity=100,
    flip=3,
    start_at=0,
    end_at=3000,
    layer=1
)

image_reader = ImageReader(
    file_path=jpg_path,
    width=640,
    height=805,
    top=-100,
    left=-100,
    angle=0,
    opacity=100,
    flip=1,
    start_at=3000,
    end_at=6000,
    layer=1
)
png_reader = ImageReader(
    file_path=png_path,
    width=640,
    height=805,
    top=-100,
    left=-100,
    angle=0,
    opacity=100,
    flip=1,
    start_at=3000,
    end_at=6000,
    layer=1
)

text_reader = TextReader(
    text="hello world你好世界",
    align="center",
    stroke_width=2,
    stroke_color='#000000',
    font=".//temp/cache/2df4c63f00d0508e33e02698a5a1ccb0.ttf",
    fontsize=50,
    color='#ffffff',
    size=(640, 805),
    top=-100,
    left=-100,
    opacity=50,
    start_at=0,
    end_at=6000,
    layer=1,
    letter_spacing=0,
)

infer_reader = VideoReader(
    file_path=infer_path,
    width=640,
    height=805,
    top=-100,
    left=-100,
    angle=0,
    opacity=100,
    flip=0,
    start_at=0,
    end_at=6000,
    layer=1,
    mask_file_path=infer_mask_path,
    start_index=0,
)

mv_reader = VideoReader(
    file_path=mv_path,
    width=1280,
    height=720,
    top=300,
    left=300,
    angle=0,
    opacity=100,
    flip=0,
    start_at=3000,
    end_at=10000,
)

audio_reader = AudioReader(
    file_path=audio_path,
    start_at=0,
    end_at=3000,
    volume=100,
)


import logging
logger = logging.getLogger(__name__)

readers = [text_reader, gif_reader, video_reader, image_reader, png_reader, infer_reader, mv_reader, audio_reader]

with VideoBuilder(
    readers=readers,
    background="#aa0000",
    width=1920,
    height=1080,
    logger=logger,
) as builder:
    builder.build(base_dir="./")




