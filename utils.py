import requests
import os
from socket import create_connection
from obs_util import ObsUtil
import logging


class TaskLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        task_id = self.extra.get('task_id', '')
        if task_id != "":
            return '{} {}'.format(task_id, msg), kwargs
        else:
            return msg, kwargs


def get_logger(name, task_id=""):
    logging.basicConfig(level = logging.INFO,format = '[%(levelname)s] - %(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
    base_name = os.path.basename(name)
    logger = logging.getLogger(base_name)
    task_logger = TaskLogger(logger, {"task_id": task_id})
    return task_logger



def create_socket(url):
    # 创建永远不会断开的socket
    socket = create_connection(url, timeout=60 * 60 * 24 * 365)
    return socket


def callback_to_url(url, code, msg, task_id, data=None, server_id=None):
    """Send callback to url."""
    # task_id合并到data里
    if data is None:
        data = {}
    data['task_id'] = task_id
    result = Result(code, msg, data=data).to_dict()
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url
    # print("发送数据", result)
    headers = {'Content-Type': 'application/json'}
    if server_id:
        headers['X-Server-ID'] = server_id
    res = requests.post(url, json=result, headers=headers)
    res.raise_for_status()
    # 获取返回的数据，text或者json
    if res.headers['Content-Type'] == 'application/json':
        res_json = res.json()
        if res_json['code'] != 0:
            raise Exception(res_json['msg'])
        return res_json['data']


def upload_to_obs(file_path, logger=None):
    obs = ObsUtil(logger=logger)
    return obs.upload(file_path)



def convert_audio_to_16k(audio_path) -> str:
    """
    把音频转换成16k的音频, 如果已经是16k的wav音频则不转换
    args:
        audio_path: str, 音频文件的路径
    return:
        str, 转换后的音频文件路径
    """
    # 获取wav文件的hz数
    cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 {audio_path}'
    sample_rate = os.popen(cmd).read()
    sample_rate = float(sample_rate)
    if sample_rate == 16000 and audio_path.endswith('.wav'):
        return audio_path
    # 生成一个随机的文件名
    file_name = audio_path + '_16k.wav'
    # 转换音频
    os.system(f'ffmpeg -i {audio_path} -acodec pcm_s16le -ac 1 -ar 16000 -y {file_name}')
    return file_name


# rgb的hex颜色转gbr的hex颜色
def rbg_hex_to_bgr(hex_color):
    hex_color = hex_color.replace('#', '')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return '#%02x%02x%02x' % (b, g, r)


class Result:
    def __init__(self, code, msg, data=None):
        self.code = code
        self.msg = msg
        self.data = data

    def to_dict(self):
        return {"code": self.code, "msg": self.msg, "data": self.data}


# 创建CODE的枚举
class Code:
    # 默认正常返回
    OK = 0
    UNKOWN_ERROR = -1

    # 业务正常返回
    PROGRESS = 100001
    RESULT = 100002

    # 业务错误返回
    OVER_THREAD = 200002  # 超出并发限制
    NO_FACE_DETECTED = 200003  # 未检测到人脸
    DOWNLOAD_ERROR = 200004  # 下载文件错误
    INFER_ERROR = 200005  # 推理错误
    COMBINE_ERROR = 200006  # 合成错误
    ENHANCE_ERROR = 200007  # 面部增强错误


