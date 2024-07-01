import os.path

from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from moviepy.video.fx.all import mirror_x, mirror_y
from movie_builder.utils import get_video_info, FPS
from typing import Optional


def get_clip(
        file_path: str,
        width: int,
        height: int,
        angle: int = 0,
        opacity: int = 100,
        flip: int = 0,
        start_at: int = 0,
        end_at: int = 0,
        volume: int = 100,
        fade_in: int = 0,
        fade_out: int = 0,
        start_index: int = 0,
        is_mask: bool = False,
) -> (VideoFileClip, AudioFileClip):
    """
    获取clip
    :param file_path: 文件路径
    :param width: 宽度
    :param height: 高度
    :param angle: 旋转角度, 0-360
    :param opacity: 不透明度, 0-100, 100: 不透明, 0: 完全透明
    :param flip: 翻转, 0: 不翻转, 1: 水平翻转, 2: 垂直翻转, 3: 水平垂直翻转
    :param start_at: 开始时间, ms
    :param end_at: 结束时间, ms
    :param volume: 音量, 0-100
    :param fade_in: 淡入时间, ms
    :param fade_out: 淡出时间, ms
    :param start_index: int, 开始的帧数
    :param is_mask: bool, 是否是mask
    :return: VideoFileClip, AudioFileClip
    """
    #ms, _, _ = get_video_info(file_path)
   # seconds = ms / 1000
    clip = VideoFileClip(file_path)
    seconds=clip.duration
    clip = clip.resize((width, height))
    start_seconds = start_at / 1000
    end_seconds = end_at / 1000
    duration = end_seconds - start_seconds
    if duration<0:
        raise Exception("视频结束时间不能早于开始时间")
    clip = clip.rotate(angle)
    if not is_mask:
        opacity = opacity / 100 if opacity else 1
        clip = clip.set_opacity(opacity)
        loop_clip = concatenate_videoclips([clip] * (int(duration / seconds) + 1))
    else:
        t_start = start_index * FPS / 1000
        copy_clip = clip.copy()
        #start_clip = clip.subclipsubclip(t_start=t_start)
        start_clip = clip.subclip(t_start=t_start)
        loop_clip = concatenate_videoclips([start_clip] + [copy_clip] * (int(duration / seconds)))

    clip = loop_clip.set_start(start_seconds).set_end(end_seconds)
    if flip == 1:
        clip = clip.fx(mirror_x)
    elif flip == 2:
        clip = clip.fx(mirror_y)
    elif flip == 3:
        clip = clip.fx(mirror_x).fx(mirror_y)
    audio_clip = clip.audio
    if audio_clip is not None:
        # 设置audio_clip的音量
        audio_clip = audio_clip.volumex(volume / 100)
        # 设置audio_clip的淡入淡出
        audio_clip = audio_clip.audio_fadein(fade_in).audio_fadeout(fade_out)
    return clip, audio_clip


class VideoReader:
    def __init__(
            self,
            file_path: str,
            width: int,
            height: int,
            top: int = 0,
            left: int = 0,
            angle: int = 0,
            opacity: int = 100,
            flip: int = 0,
            start_at: int = 0,
            end_at: int = 0,
            volume:int=100,
            layer: int = 1,
            mask_file_path: Optional[str] = None,
            start_index: int = 0,
    ):
        self.end_at = end_at
        self.layer = layer
        """
        视频读取器
        :param file_path: 文件路径
        :param width: 最终宽度
        :param height: 最终高度
        :param top: 离顶部距离
        :param left: 离左边距离
        :param angle: 旋转角度
        :param opacity: 透明度, 0~100, 0为完全透明, 100为完全不透明
        :param flip: 翻转, 0为不翻转, 1为水平翻转, 2为垂直翻转, 3为水平垂直翻转
        :param start_at: 在视频的哪个时间点开始
        :param end_at: 在视频的哪个时间点结束
        :param layer: 在哪一层
        """
        video_clip, audio_clip = get_clip(
            file_path=file_path,
            width=width,
            height=height,
            angle=angle,
            opacity=opacity,
            flip=flip,
            start_at=start_at,
            end_at=end_at,
            volume=volume,
            start_index=0,
            is_mask=False,
        )
        if mask_file_path:
            mask_clip, _ = get_clip(
                file_path=mask_file_path,
                width=width,
                height=height,
                angle=angle,
                opacity=opacity,
                flip=flip,
                start_at=start_at,
                end_at=end_at,
                volume=volume,
                start_index=start_index,
                is_mask=True,
            )
            clip = video_clip.set_mask(mask_clip.to_mask())
        else:
            clip = video_clip

        start_at = round(start_at / 1000, 2)
        end_at = round(end_at / 1000, 2)
        opacity = opacity / 100 if opacity < 100 else 1
        clip = clip.set_start(start_at) \
            .set_end(end_at) \
            .set_duration(end_at - start_at) \
            .set_position((left, top)) \
            .set_opacity(opacity)
        self.clip = clip
        self.audio_clip = audio_clip
        self.layer = layer
        self.end_at = end_at

    def get_v_clip(self):
        return self.clip

    def get_a_clip(self):
        return self.audio_clip
