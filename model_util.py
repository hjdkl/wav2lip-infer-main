from pydantic import BaseModel, conlist

from typing import List, Optional


class AudioNodeItem(BaseModel):
    url: str  # 音频地址，支持mp3、wav
    file_path: Optional[str] = None  # 音频文件路径，如果是本地文件，可以不传
    volume: Optional[int] = 100  # 音量，0~200，100是原始音量
    startAt: Optional[int] = 0  # 音频开始的时间，单位ms
    startOffset: Optional[int] = 0  # 音频开始的偏移量，单位ms
    endAt: Optional[int] = 0  # 音频结束的时间，单位ms
    endOffset: Optional[int] = 0  # 音频结束的偏移量，单位ms
    loop: Optional[bool] = False  # 音频是否循环，默认否
    fadeInDuration: Optional[int] = 0  # 淡入时长，单位ms
    fadeOutDuration: Optional[int] = 0  # 淡出时长，单位ms


class VideoNodeItem(BaseModel):
    url: str  # 视频地址，支持mp4
    file_path: Optional[str] = None  # 视频文件路径，如果是本地文件，可以不传
    width: int  # 最终呈现的宽度
    height: int  # 最终呈现的高度
    layer: int  # 所在图层，-inf~inf
    rotate: int  # 旋转角度，0~360，顺时针
    opacity: int  # 不透明度，0~100，100是完全不透明
    marginLeft: int  # 左边距，-inf~inf
    marginTop: int  # 上边距，-inf~inf
    startAt: int  # 视频开始的时间，单位ms
    startOffset: int  # 视频开始的偏移量，单位ms
    endAt: int  # 视频结束的时间，单位ms
    endOffset: int  # 视频结束的偏移量，单位ms
    volume: Optional[int] = 100  # 音量，0~200，100是原始音量
    flip: int = 0  # 翻转，0不翻转，1水平翻转，2垂直翻转，3水平垂直翻转
    loop: Optional[bool] = False  # 视频是否循环，默认否
    fadeInDuration: Optional[int] = 0  # 淡入时长，单位ms
    fadeOutDuration: Optional[int] = 0  # 淡出时长，单位ms


class ImageNodeItem(BaseModel):
    url: str  # 图片地址，支持jpg、png、jpeg
    file_path: Optional[str] = None  # 图片文件路径，如果是本地文件，可以不传
    width: int  # 最终呈现的宽度
    height: int  # 最终呈现的高度
    layer: int  # 所在图层，-inf~inf
    rotate: int  # 旋转角度，0~360，顺时针
    opacity: int  # 不透明度，0~100，100是完全不透明
    marginLeft: int  # 左边距，-inf~inf
    marginTop: int  # 上边距，-inf~inf
    startAt: int  # 开始时间，单位ms
    endAt: int  # 结束时间，单位ms
    flip: int = 0  # 翻转，0不翻转，1水平翻转，2垂直翻转，3水平垂直翻转


class GifNodeItem(BaseModel):
    url: str  # gif地址
    file_path: Optional[str] = None  # gif文件路径，如果是本地文件，可以不传
    width: int  # 最终呈现的宽度
    height: int  # 最终呈现的高度
    layer: int  # 所在图层，-inf~inf
    rotate: int  # 旋转角度，0~360，顺时针
    opacity: int  # 不透明度，0~100，100是完全不透明
    marginLeft: int  # 左边距，-inf~inf
    marginTop: int  # 上边距，-inf~inf
    startAt: int  # 开始时间，单位ms
    endAt: int  # 结束时间，单位ms
    flip: int = 0  # 翻转，0不翻转，1水平翻转，2垂直翻转，3水平垂直翻转


class TextNodeItem(BaseModel):
    text: str  # 文本内容
    fontSize: int  # 字体大小
    fontFamily: str  # 字体文件地址
    file_path: Optional[str] = ''  # 字体文件本地地址
    fontStyle: Optional[str] = None  # 字体样式，''、italic、bold
    align: Optional[str] = 'center'  # 对齐方式，left、center、right, top, bottom
    color: str  # 字体颜色，#000000
    strokeColor: Optional[str] = None  # 描边颜色，#000000
    stroke: Optional[int] = 0  # 描边宽度，0~100
    width: int  # 文字框的宽度，涉及到换行、溢出
    height: int  # 文字框的高度，涉及到换行、溢出
    layer: Optional[int] = None  # 所在图层，0~inf
    marginLeft: int  # 左边距，-inf~inf
    marginTop: int  # 上边距，-inf~inf
    opacity: int  # 不透明度，0~100，100是完全不透明
    startAt: int  # 开始时间，单位ms
    endAt: int  # 结束时间，单位ms
    letterSpacing: int = 0  # 字间距


class ModelItem(BaseModel):
    url: str  # 推理视频地址
    weight_path: Optional[str] = None  # 模特权重地址
    file_path: Optional[str] = None  # 推理视频文件路径，用于设置推理后的视频位置, 非接口参数
    maskUrl: Optional[str] = None  # 推理视频对应的蒙版视频，要求：1.宽高一致，2.帧率一致，3.时长一致，4.背景为黑色，5.前景为白色
    mask_path: Optional[str] = None  # 推理视频对应的蒙版视频文件路径，用于下载后设置蒙版视频位置, 非接口参数
    width: int  # 模特最后的宽度
    height: int  # 模特最后的高度
    layer: int  # 所在图层，-inf~inf
    rotate: int  # 旋转角度，0~360，顺时针
    opacity: int  # 不透明度，0~100，100是完全不透明
    marginLeft: int  # 左边距，-inf~inf
    marginTop: int  # 上边距，-inf~inf
    startAt: int  # 开始时间，单位ms
    endAt: int  # 结束时间，单位ms
    startFrameIndex: int  # 开始帧，int
    flip: int = 0  # 翻转，0不翻转，1水平翻转，2垂直翻转，3水平垂直翻转


class SpeakerItem(BaseModel):
    url: str  # 说话的wav音频，16000采样率，单声道，pcm_s16le
    volume: int  # 音量，0~200，100是原始音量
    fadeInDuration: int  # 淡入时长，单位ms
    fadeOutDuration: int  # 淡出时长，单位ms
    denoising: bool  # 是否降噪，暂时没处理


class TransitionItem(BaseModel):
    effect: str = 'fade'  # 转场效果，参考https://trac.ffmpeg.org/wiki/Xfade，ffmpeg支持的效果，暂时不自己实现
    duration: int = 120  # 转场时长，单位ms，120~1000


class SceneItem(BaseModel):
    audio_nodes: List[AudioNodeItem]
    video_nodes: List[VideoNodeItem]
    image_nodes: List[ImageNodeItem]
    text_nodes: List[TextNodeItem]  # 文字节点, 8.18新增
    gif_nodes: List[GifNodeItem]
    # 推理模特视频、音频，如果任一是None，则跳过推理，只合成
    model: Optional[ModelItem] = None
    speaker: Optional[SpeakerItem] = None  # 说话人
    background: str  # 背景，颜色/图片
    transition: Optional[TransitionItem] = None  # 转场, 8.18新增
    sort: int  # 排序


# 使用的模型，本文件的其他模型，都是这个模型的子集
class InferRequest(BaseModel):
    bitrate: float  # 比特率，暂时不会接收
    fps: int  # 帧率，暂时不会接收，固定为25
    width: int  # 视频宽度，单位px
    height: int  # 视频高度，单位px
    callback: str  # 回调地址
    scenes: conlist(SceneItem, min_length=1)  # 场景列表
