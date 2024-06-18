import os

from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.editor import ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from movie_builder.utils import hex_to_rgb
from uuid import uuid4


class VideoBuilder:
    def __init__(
            self,
            readers,
            background: str = '',
            resolution: tuple = (1920, 1080),
            logger=None,
    ):
        """
        视频构建器
        :param readers: image, gif, video, text, audio的读取器
        :param background: 背景颜色
        :param resolution: 分辨率
        :param logger:
        """
        self.logger = logger
        # 处理基础color_clip作为背景
        max_end_at = 0
        v_clips = []
        a_clips = []
        readers = sorted(readers, key=lambda x: x.layer)
        for reader in readers:
            end_at = reader.end_at
            v_clips.append(reader.get_v_clip())
            a_clips.append(reader.get_a_clip())
            if end_at > max_end_at:
                max_end_at = end_at
        self.duration = max_end_at
        color = hex_to_rgb(background)
        base_clip = ColorClip(size=resolution, color=color, duration=self.duration)
        v_clips = list(filter(lambda x: x is not None, [base_clip] + v_clips))
        a_clips = list(filter(lambda x: x is not None, [] + a_clips))
        self.video_clips = v_clips
        self.audio_clips = a_clips

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.video_clips
        del self.audio_clips

    def build(self, base_dir='') -> str:
        """
        构建视频
        :param base_dir:
        :return: 视频路径
        """
        logger = self.logger
        logger.info('开始构建视频')
        video_path = os.path.join(base_dir, f'{uuid4()}.mp4')
        video_mix = CompositeVideoClip(self.video_clips)
        if len(self.audio_clips):
            audio_mix = CompositeAudioClip(self.audio_clips)
            audio_mix = audio_mix.set_duration(video_mix.duration)
            video_mix = video_mix.set_audio(audio_mix)
        video_mix.write_videofile(video_path, fps=25, codec="libx264", audio_codec="aac")
        logger.info(f'视频构建完成: {video_path}')
        return video_path
