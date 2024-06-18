from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.all import mirror_x, mirror_y
from movie_builder.utils import get_video_info


class GifReader:
    def __init__(
            self,
            file_path: str,
            width: int,
            height: int,
            top: int,
            left: int,
            angle: int,
            opacity: int,
            flip: int,
            start_at: int,
            end_at: int,
            layer: int,
    ):
        """
        gif读取器
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

        # ms, _, _ = get_video_info(file_path)
        # seconds = ms / 1000
        clip = VideoFileClip(file_path, has_mask=True)
        seconds = clip.duration
        clip = clip.resize((width, height))
        start_seconds = start_at / 1000
        end_seconds = end_at / 1000
        duration = end_seconds - start_seconds
        self.end_at = end_seconds
        loop_clip = concatenate_videoclips([clip] * (int(duration / seconds) + 1))
        clip = loop_clip.set_start(start_seconds).set_end(end_seconds).set_position((left, top)).set_duration(duration)
        clip = clip.rotate(angle)
        opacity = opacity / 100 if opacity else 1
        clip = clip.set_opacity(opacity)
        if flip == 1:
            clip = clip.fx(mirror_x)
        elif flip == 2:
            clip = clip.fx(mirror_y)
        elif flip == 3:
            clip = clip.fx(mirror_x).fx(mirror_y)
        self.clip = clip
        self.layer = layer

    def get_v_clip(self):
        return self.clip

    def get_a_clip(self):
        return None
