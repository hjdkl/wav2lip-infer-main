from moviepy.video.VideoClip import TextClip
from moviepy.editor import CompositeVideoClip
from typing import Tuple

from movie_builder.utils import rbg_hex_to_bgr


class TextReader:
    def __init__(
        self,
        text: str,
        font: str,
        size: Tuple[int, int],
        top: int,
        left: int,
        align: str,
        fontsize: int,
        color: str,
        stroke_color: str,
        stroke_width: int,
        letter_spacing: int,
        opacity: int,
        start_at: int,
        end_at: int,
        layer: int,
    ) -> None:
        """
        文字读取器
        :param text: 需要显示的文字
        :param font: 字体位置
        :param size: box大小
        :param top: 上边距
        :param left: 左边距
        :param align: 对齐类型
        :param fontsize: 文字尺寸
        :param color: 字体颜色
        :param stroke_color: 描边颜色
        :param stroke_width: 描边宽度
        :param letter_spacing: 字间距
        :param opacity: 不透明度, 0~100, 0为完全透明, 100为完全不透明
        :param start_at: 开始时间
        :param end_at: 结束时间
        :param layer: 图层
        """
        self.layer = layer
        direction_map = {
            'center': 'center',
            'left': 'West',
            'right': 'East',
            'top': 'North',
            'bottom': 'South'
        }
        align = direction_map.get(align, 'center')
        #color = rbg_hex_to_bgr(color)
        self.start_at = round(start_at / 1000, 2)
        self.end_at = round(end_at / 1000, 2)
        self.opacity = opacity / 100
        self.left = left
        self.top = top

        # if stroke_color is not None:
            # stroke_color = rbg_hex_to_bgr(stroke_color)

        # 创建字体clip
        self.text_clip = TextClip(text, font=font, fontsize=fontsize, size=size,
                                  method='caption', align=align, color=color, kerning=letter_spacing)
        self.text_clip = self.text_clip

        # 有描边宽度和颜色，创建描边clip，字体颜色和描边颜色一样（纯色）
        if stroke_width > 0 and stroke_color is not None:
            self.stroke_clip = TextClip(text, font=font, fontsize=fontsize, size=size,
                                        method='caption', align=align, color=stroke_color, kerning=letter_spacing,
                                        stroke_color=stroke_color, stroke_width=stroke_width)
            self.stroke_clip = self.stroke_clip
        else:
            self.stroke_clip = None

    def get_v_clip(self):
        start_at = self.start_at
        end_at = self.end_at
        opacity = self.opacity
        left = self.left
        top = self.top
        if self.stroke_clip is None:
            final_clip = self.text_clip
        else:
            final_clip = CompositeVideoClip([self.stroke_clip, self.text_clip])
        final_clip = final_clip.set_start(start_at).set_end(end_at).set_position((left, top)).set_opacity(opacity).set_duration(
            end_at - start_at)
        return final_clip

    def get_a_clip(self):
        return None
