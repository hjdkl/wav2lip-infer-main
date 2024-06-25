import json
import logging
import os
from typing import Any, List, Union
from uuid import uuid4
from moviepy.editor import *
import subprocess

import cv2
import hashlib
import numpy as np
import requests

FPS = 25  # 帧率
CACHE_DIR = './temp/cache'


class Transition:
    def __init__(self, effect: str = 'fade', duration: int = 120):
        """
        转场效果
        :param effect: 转场效果
        :param duration: 转场时长
        """
        self.effect = effect
        self.duration = duration


class CombinedVideoItem:
    def __init__(self, video_path, transition: Transition = None):
        """
        视频列表对象, 用于构建带转场的视频
        :param video_path: 视频路径
        :param transition: 转场效果
        """
        self.video_path = video_path
        if transition:
            self.transition = transition
        else:
            self.transition = Transition()


class CombinedVideoList:
    def __init__(self, video_list: List[CombinedVideoItem] = []):
        self.video_list = video_list

    def merge(self, target_dir: str = '', logger: Any = None):
        """
        合并视频
        :param target_dir: 输出目录
        :return: 合并后的视频路径
        """
        if len(self.video_list) == 0:
            raise Exception('视频列表为空')
        if len(self.video_list) == 1:
            return self.video_list[0].video_path
        file_list, video_complex, audio_complex = get_xfade_filter_cmd(self.video_list)
        out_name = os.path.join(target_dir, f'{uuid4()}.mp4')
        cmd = f'ffmpeg -loglevel warning {file_list} -filter_complex "{video_complex}{audio_complex}" -vcodec libx264 -movflags +faststart -y {out_name}'
        os.system(cmd)
        return out_name


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(asctime)s - %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    base_name = os.path.basename(name)
    logger = logging.getLogger(base_name)
    return logger

def has_audio(video_path):  
    try:  
        video = VideoFileClip(video_path)  
        return video.audio is not None  
    except Exception as e:  
        print(f"Error loading video: {e}")  
        return False  
    finally:  
        if 'video' in locals() and isinstance(video, VideoFileClip):  
            video.close()  
            del video  
  

def get_xfade_filter_cmd(video_list: List[CombinedVideoItem]) -> (str, str, str):
    """获取转场滤镜的命令
    args:
        video_list: 视频列表
    return:
        file_list: ffmpeg的-i参数
        video_complex: ffmpeg的-filter_complex参数
        audio_complex: ffmpeg的-filter_complex参数
    """
    valid_effects = ['fade', 'fadeblack', 'fadewhite', 'distance', 'wipeleft', 'wiperight', 'wipeup', 'wipedown',
                     'slideleft', 'slideright', 'slideup', 'slidedown', 'smoothleft', 'smoothright', 'smoothup',
                     'smoothdown',
                     'rectcrop', 'circlecrop', 'circleclose', 'circleopen', 'horzclose', 'horzopen', 'vertclose',
                     'vertopen',
                     'diagbl', 'diagbr', 'diagtl', 'diagtr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'dissolve',
                     'pixelize',
                     'radial', 'hblur', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'fadegrays', 'squeezev', 'squeezeh',
                     'zoomin']
    if len(video_list) <= 1:
        raise Exception('视频数量必须大于1')
    last_offset = 0
    file_list = ''
    video_complex = ''
    audio_complex = ''
     
     # 没有音频的视频中添加静音音频
    audio_streams = [has_audio(item.video_path) for item in video_list]
    for j in range(len(video_list)): 
      if  not audio_streams[j]:  
            video = VideoFileClip(video_list[j].video_path)
            audio_dir = "./temp"
            if not os.path.exists(audio_dir):  
                os.makedirs(audio_dir)
            video_path = video_list[j].video_path  # 获取当前视频的实际路径  
            base_name, ext = os.path.splitext(os.path.basename(video_path))  # 获取文件名和扩展名  
            temp_file_path = os.path.join(audio_dir, f"{base_name}_with_silent_audio{ext}")  # 构建临时文件路径
            audio_cmd = [  

            'ffmpeg',  

            '-i', video_path,  # 输入视频文件路径  

            '-f', 'lavfi',  

            '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # 输入静音音频  

            '-c:v', 'copy',  # 复制视频流，不重新编码  

            '-c:a', 'aac',  # 使用AAC编码音频流  

            '-shortest',  # 输出文件的时长与最短的输入流相匹配（即视频时长）  

            temp_file_path  # 输出文件路径  
        ]  
            subprocess.run(audio_cmd, check=True)    
            os.replace(temp_file_path, video_path)

    
    need_filter_len = len(video_list) - 1  # 需要添加滤镜的视频数量, 最后一个视频不需要添加滤镜)
    for i in range(need_filter_len):
        item = video_list[i]
        video_path = item.video_path
        effect = item.transition.effect
        duration = item.transition.duration  # 毫秒
        if effect not in valid_effects:
            raise Exception("{}使用了无效的转场效果: {}".format(video_path, effect))
        file_list += f' -i {video_path} '
        if i == 0:
            v_pre = '[0]'
            a_pre = '[0:a]'
        else:
            v_pre = f'[vfade{i}]'
            a_pre = f'[afade{i}]'
        v_index = f'[{i + 1}:v]'
        a_index = f'[{i + 1}:a]'
        if i == need_filter_len - 1:
            v_end = ',format=yuv420p;'
            a_end = ''
        else:
            v_end = f'[vfade{i + 1}];'
            a_end = f'[afade{i + 1}];'
        video_duration, _, _ = get_video_info(video_path)
        offset = int(video_duration + last_offset - duration)
        last_offset = offset
        video_complex += f'{v_pre}{v_index}xfade=transition={effect}:duration={duration}ms:offset={offset}ms{v_end}'
        #if audio_streams[i]:
        audio_complex += f'{a_pre}{a_index}acrossfade=d={duration}ms{a_end}'
    file_list += f' -i {video_list[-1].video_path}'  # 最后一个视频
    return file_list, video_complex, audio_complex



def split_video_audio(file_path: str, target_fps: Union[int, float] = 0, output_dir: str = CACHE_DIR) -> (str, str):
    """
    分离视频和音频
    args:
        file_path: 视频路径
        target_fps: 目标帧率
        output_dir: 输出目录
    return:
        video_path, audio_path 视频路径和音频路径
    """
    _, fps, audio_duration = get_video_info(file_path)
    has_audio = audio_duration > 0
    video_path = audio_path = ''
    # 处理音频
    if has_audio:
        audio_path = os.path.join(output_dir, f"{uuid4()}.wav")
        audio_cmd = f"ffmpeg -loglevel warning -i {file_path} -acodec pcm_s16le -y {audio_path}"
        os.system(audio_cmd)
    # 处理视频
    if target_fps == 0 or target_fps == fps:
        video_path = file_path
    else:
        video_path = os.path.join(output_dir, f"{uuid4()}.mp4")
        video_cmd = f"ffmpeg -loglevel warning -i {file_path} -r {target_fps} -y {video_path}"
        os.system(video_cmd)
    return video_path, audio_path


def cut_video(file_path: str, start_offset: int = 0, end_offset: int = 0, output_dir: str = CACHE_DIR) -> str:
    """切分视频, 如果start_offset和end_offset都为0, 则不切分, 直接返回原视频路径
    args:
        file_path: 视频路径
        start_offset: 开始偏移
        end_offset: 结束偏移
        output_dir: 输出目录
    return:
        切分后的视频路径
    """
    if start_offset == 0 and end_offset == 0:
        return file_path
    video_duration, _, _ = get_video_info(file_path)
    cut_duration = video_duration - end_offset
    out_name = os.path.join(output_dir, f'{uuid4()}.mp4')
    cmd = f'ffmpeg -loglevel warning -i {file_path} -ss {start_offset}ms -t {cut_duration}ms -q:v 0 -y {out_name}'
    os.system(cmd)
    return out_name


def get_video_info(file_path: str) -> (float, float, float):
    """
    获取视频的信息
    args: 
        file_path 视频路径
    return: 
        video_duration 视频时长ms
        fps 视频帧率
        audio_duration 音频时长ms
    """
    if os.path.isfile(file_path) is False:
        raise Exception("文件不存在: {}".format(file_path))
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {file_path}'
    result = os.popen(cmd).read()
    result = json.loads(result)
    video_stream = [x for x in result.get('streams', []) if x.get('codec_type', '') == 'video']
    video_stream = video_stream[0] if len(video_stream) > 0 else None
    video_stream_duration = video_stream.get('duration') if video_stream else 0.0
    audio_stream = [x for x in result.get('streams', []) if x.get('codec_type', '') == 'audio']
    audio_stream = audio_stream[0] if len(audio_stream) > 0 else None
    audio_stream_duration = audio_stream.get('duration') if audio_stream else 0.0
    r_frame_rate = video_stream.get('r_frame_rate')  if video_stream is not None else '25/1'
    frame_count, ms_count = r_frame_rate.split('/')
    fps = int(frame_count) / int(ms_count)
    return float(video_stream_duration) * 1000, float(fps), float(audio_stream_duration) * 1000


def get_trim_size(img: np.ndarray, img_top: int, img_left: int, bg: np.ndarray) -> (np.ndarray, int, int):
    """
    裁剪图片, 返回裁剪后的图片, 和在背景图上的top, left
    args:
        img: 要剪裁的图片
        img_top: 图片在背景图上的top, 可能是负数
        img_left: 图片在背景图上的left, 可能是负数
        bg: 背景图, 用来计算边界
    return:
        trimmed_img: 裁剪后的图片
        on_bg_top: 在背景图上的top
        on_bg_left: 在背景图上的left
    """
    bg_height, bg_width, _ = bg.shape
    img_height = img.shape[0]
    img_width = img.shape[1]
    top = 0 if img_top >= 0 else -img_top
    bottom = img_height if img_top + img_height <= bg_height else bg_height - img_top
    left = 0 if img_left >= 0 else -img_left
    right = img_width if img_left + img_width <= bg_width else bg_width - img_left
    trimmed_img = img[top:bottom, left:right]
    on_bg_top = 0 if img_top < 0 else img_top
    on_bg_left = 0 if img_left < 0 else img_left
    return trimmed_img, on_bg_top, on_bg_left


def get_md5(to_md5_str: str) -> str:
    md5 = hashlib.md5()
    md5.update(to_md5_str.encode("utf-8"))
    return md5.hexdigest()


def download_file_with_dir(url: str, local_dir="./temp"):
    """Download file from url and save it to a temporary file."""
    if url is None:
        return None
    if not url.startswith('https://') and not url.startswith('http://'):
        url = 'http://' + url
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    suffix = url.split('.')[-1]
    md5_name = os.path.join(local_dir, get_md5(url) + "." + suffix)
    if os.path.isfile(md5_name):
        return md5_name
    res = requests.get(url)
    res.raise_for_status()
    with open(md5_name, 'wb') as f:
        f.write(res.content)
    return md5_name


def rbg_hex_to_bgr(hex_color: str) -> str:
    """
    rgb的hex颜色转gbr的hex颜色
    args:
        hex_color: rgb的hex颜色
    return:
        gbr的hex颜色
    """
    hex_color = hex_color.replace('#', '')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return '#%02x%02x%02x' % (b, g, r)

def hex_to_rgb(hex_color: str) -> (int, int, int):
    hex_color = hex_color.replace('#', '')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def get_background_cv(background: str, width: int, height: int) -> np.ndarray:
    """
    获取背景图的numpy数组
    args:
        background: 背景图
        width: 背景图宽度
        height: 背景图高度
    return:
        背景图的numpy数组
    """
    if background.startswith('#'):
        # rgb转成bgr
        color = rbg_hex_to_bgr(background)
        b = int(color[1:3], 16)
        g = int(color[3:5], 16)
        r = int(color[5:7], 16)
        # 生成一个三通道的图片
        result = np.zeros((height, width, 3), dtype=np.uint8)
        result[:, :, 0] = b
        result[:, :, 1] = g
        result[:, :, 2] = r
        # 数组转成numpy数组，并带shape
        result = np.array(result, dtype=np.uint8)
        return result
    else:
        bg = download_file_with_dir(background)
        result = cv2.imread(bg)
        result = cv2.resize(result, (width, height))
        return result


# 左上角为基准点进行旋转
# 最后放到背景图上的时候, 需要把旋转后的图片, 放到背景图上的对应位置
# 步骤：
# 1. 判断是四通道还是三通道（png是四通道, jpg是三通道）
# 2. 如果是三通道的, 用自己的图片, 生成一个四通道的图片；如果是四通道的, 用自己的图片生成一个mask
# 3. 以左上角为基准点, 旋转图片
# 返回：
# 1. 旋转后的图片
# 2. 旋转后的mask（兼容三通道和四通道）
# 3. 旋转后新的左上角离初始图左上角的offset_x
# 4. 旋转后新的左上角离初始图左上角的offset_y
def rotate_img_clockwise(img, angle):
    is_4_channel = img.shape[2] == 4
    # 如果是四通道, mask就是第四个通道, 否则是全白的mask
    mask = img[:, :, 3] if is_4_channel else np.ones(img.shape[:2], dtype=np.uint8) * 255
    # mask转成三通道
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rows, cols = img.shape[:2]
    if angle == 0:
        return img, mask, 0, 0
    # angle是逆时针旋转的, 所以要取反
    angle = -angle
    # 以为中心点, 顺时针旋转angle度
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # 旋转后, 图片的宽高会变化, 所以要重新计算
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int(rows * sin + cols * cos)
    new_height = int(rows * cos + cols * sin)
    # 旋转后, 图片的中心点会变化, 所以要重新计算
    M[0, 2] += (new_width - cols) / 2
    M[1, 2] += (new_height - rows) / 2
    rotated_img = cv2.warpAffine(img, M, (new_width, new_height))
    rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height))
    offset_x = int((new_width - cols) / 2)
    offset_y = int((new_height - rows) / 2)
    return rotated_img, rotated_mask, offset_x, offset_y


def flip_img(img, direction):
    """
    翻转图片

    参数:
    - img: 输入图片
    - direction: 翻转方向，0-不翻转，1-水平翻转，2-垂直翻转，3-水平垂直翻转
    
    返回:
    - 翻转后的图片
    """
    if direction == 0:
        # 不翻转
        return img
    elif direction == 1:
        # 水平翻转
        return cv2.flip(img, 1)
    elif direction == 2:
        # 垂直翻转
        return cv2.flip(img, 0)
    elif direction == 3:
        # 水平垂直翻转
        return cv2.flip(img, -1)
