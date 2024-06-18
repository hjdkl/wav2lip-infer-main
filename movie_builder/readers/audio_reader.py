import moviepy.audio.fx.all as afx
from movie_builder.utils import get_video_info
from moviepy.editor import concatenate_audioclips, AudioFileClip


class AudioReader:
    def __init__(
            self,
            file_path: str,
            start_at: int,
            end_at: int,
            start_offset: int = 0,
            end_offset: int = 0,
            volume: int = 100,
            fade_in: int = 0,
            fade_out: int = 0,
            loop: bool = False,
    ) -> None:
        """
        初始化audio读取器
        :param file_path: str, 文件路径
        :param start_at: int, 毫秒, 开始时间
        :param end_at: int, 毫秒, 开始时间
        :param start_offset: int, 毫秒
        :param end_offset: int, 毫秒
        :param volume: 音量, 0~, 100是标准音量大小
        :param fade_in: int, 渐入时长, ms
        :param fade_out: int, 渐出时长, ms
        :param loop: bool, 是否循环, 默认False
        """
        clip = AudioFileClip(file_path)
        start_seconds = round(start_at / 1000, 2)
        end_seconds = round(end_at / 1000, 2)
        self.end_at = end_seconds
        if start_offset or end_offset:
            start_offset_seconds = round(start_offset / 1000, 2)
            end_offset_seconds = round(end_offset / 1000, 2)
            clip = clip.subclip(t_start=start_offset_seconds, t_end=end_offset_seconds)
        volume_scale = round(volume / 100, 2) if volume != 100 else 1
        clip = clip.volumex(volume_scale)
        if loop:
            _, _, ms = get_video_info(file_path)
            duration = end_at - start_at
            loop_times = int(duration / ms) + 1
            clip = concatenate_audioclips([clip] * loop_times)
        if fade_in or fade_out:
            fade_in_second = round(fade_in / 1000, 2)
            fade_out_second = round(fade_out / 1000, 2)
            clip = clip.fx(afx.audio_fadein, fade_in_second).fx(afx.audio_fadeout, fade_out_second)
        self.clip = clip.set_start(start_seconds).set_end(end_seconds).set_duration(end_seconds - start_seconds)
        self.layer = 0

    def get_a_clip(self) -> AudioFileClip:
        return self.clip

    def get_v_clip(self):
        return None

