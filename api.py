import datetime
import glob
import json
import math
import traceback
import torch
from fastapi import FastAPI, BackgroundTasks
import uvicorn, yaml, argparse
from typing import Any, Union, Type
from utils import *
from inference import wav_to_lip_with_chin
from utils import Result, callback_to_url
from model_util import *
from threading import Lock
import gc
from exceptions import *
from uuid import uuid4
from fastapi.responses import JSONResponse
from movie_builder.readers import GifReader, VideoReader, ImageReader, TextReader, AudioReader
from movie_builder.utils import download_file_with_dir, CombinedVideoList, CombinedVideoItem
from movie_builder import VideoBuilder

server_logger = get_logger(__file__, task_id="SERVER")

# Create the FastAPI app object
app = FastAPI()

obs_config = None

callback_url_dict = {}

TEMP_DIR = "./temp"

CACHE_DIR = "./temp/cache"

JOB_DIR = "./temp/job"

MAX_TASK_COUNT = 2
running_thread_count = 0
running_task_ids = []

server_id = None

lock = Lock()

# 多线程处理的时候, 最大用到的worker数量
workers = int(os.cpu_count() * 0.5)
workers = workers if workers > 0 else 1
server_logger.info(f'将使用{workers}个worker进行多线程处理')

debug = False


# Define the main route
@app.get("/")
def index():
    return "0.1.1"


def release_job(task_id):
    job_path = os.path.join(JOB_DIR, f"{task_id}.json")
    if os.path.exists(job_path):
        os.remove(job_path)
    handle_running_count(-1)


def handle_running_count(increment):
    global running_thread_count
    lock.acquire()
    running_thread_count += increment
    lock.release()


# 释放torch的资源
def gc_collect():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


AUDIO_READER = "audio"
VIDEO_READER = "video"
IMAGE_READER = "image"
TEXT_READER = "text"
GIF_READER = "gif"


# TODO, 优化, 拆分一下, 代码太多了
def create_reader(
        reader_type: str,
        file_path: str,
        width: int = 0,
        height: int = 0,
        top: int = 0,
        left: int = 0,
        angle: int = 0,
        opacity: int = 100,
        flip: int = 0,
        start_at: int = 0,
        end_at: int = 0,
        start_offset: int = 0,
        end_offset: int = 0,
        layer: int = 0,
        text: str = "",
        align: str = "center",
        fontsize: int = 0,
        color: str = "#000000",
        stroke_color: str = "#000000",
        stroke_width: int = 0,
        letter_spacing: int = 0,
        volume: int = 100,
        fade_in: int = 0,
        fade_out: int = 0,
        loop: bool = False,
        mask_path: str = "",
        start_index: int = 0,
        logger=None,
):
    """
    创建reader对象
    :param reader_type: str, reader类型, AUDIO_READER | VIDEO_READER | IMAGE_READER | TEXT_READER | GIF_READER
    :param file_path: str, 文件路径
    :param width: int, 宽度
    :param height: int, 高度
    :param top: int, 顶部位置
    :param left: int, 左侧位置
    :param angle: int, 角度
    :param opacity: int, 不透明度, 0-100
    :param flip: int, 翻转, 0: 不翻转, 1: 水平翻转, 2: 垂直翻转, 3: 水平垂直翻转
    :param start_at: int, 开始时间, ms
    :param end_at: int, 结束时间, ms
    :param start_offset: int, 开始偏移, ms
    :param end_offset: int, 结束偏移, ms
    :param layer: int, 层级
    :param text: str, 文本内容
    :param align: str, 对齐方式, left | center | right
    :param fontsize: int, 字体大小
    :param color: str, 字体颜色
    :param stroke_color: str, 描边颜色
    :param stroke_width: int, 描边宽度
    :param letter_spacing: int, 字间距
    :param volume: int, 音量, 0-100
    :param fade_in: int, 淡入时间, ms
    :param fade_out: int, 淡出时间, ms
    :param loop: bool, 是否循环
    :param mask_path: str, mask路径
    :param start_index: int, 开始索引
    :param logger: 日志对象
    :return: GifReader | ImageReader | VideoReader | AudioReader | TextReader
    """
    if reader_type == IMAGE_READER:
        return ImageReader(
            file_path=file_path,
            width=width,
            height=height,
            top=top,
            left=left,
            angle=angle,
            opacity=opacity,
            flip=flip,
            start_at=start_at,
            end_at=end_at,
            layer=layer,
        )
    elif reader_type == VIDEO_READER:
        return VideoReader(
            file_path=file_path,
            width=width,
            height=height,
            top=top,
            left=left,
            angle=angle,
            opacity=opacity,
            flip=flip,
            start_at=start_at,
            end_at=end_at,
            layer=layer,
            mask_file_path=mask_path,
            start_index=start_index,
        )
    elif reader_type == AUDIO_READER:
        return AudioReader(
            file_path=file_path,
            start_at=start_at,
            end_at=end_at,
            start_offset=start_offset,
            end_offset=end_offset,
            volume=volume,
            fade_in=fade_in,
            fade_out=fade_out,
            loop=loop,
        )
    elif reader_type == TEXT_READER:
        return TextReader(
            text=text,
            font=file_path,
            size=(width, height),
            top=top,
            left=left,
            align=align,
            fontsize=fontsize,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            opacity=opacity,
            letter_spacing=letter_spacing,
            start_at=start_at,
            end_at=end_at,
            layer=layer,
        )
    elif reader_type == GIF_READER:
        return GifReader(
            file_path=file_path,
            width=width,
            height=height,
            top=top,
            left=left,
            angle=angle,
            opacity=opacity,
            flip=flip,
            start_at=start_at,
            end_at=end_at,
            layer=layer,
        )
    else:
        msg = "不支持的reader类型: {}".format(reader_type)
        logger.error(msg)
        raise Exception(msg)


def download_file_with_node(node, url, logger=None):
    try:
        file_path = download_file_with_dir(url, local_dir=CACHE_DIR)
        node.file_path = file_path
        return node
    except Exception as e:
        msg = "下载文件失败 {}".format(e)
        logger.error(msg)
        raise Exception(msg)


def infer_single_video(
        video_path: str,
        audio_path: str,
        model_weight: str = "weights/pt17.pth",
        start_index: int = 0,
        base_dir: str = "./temp",
        logger: Any = None,
        scene_index: int = 0,
):
    """
    推理单个视频
    :param video_path: str, 视频路径
    :param audio_path: str, 音频路径
    :param model_weight: str, 模型权重地址
    :param start_index: int, 开始帧
    :param base_dir: str, 临时文件夹
    :param logger: Any, 日志对象
    :param scene_index: int, 场景索引
    :return: str, 本地视频路径
    """
    logger.info("推理单个视频")
    # 获取audio的时长
    cmd = f"ffprobe -i {audio_path} -show_entries format=duration -v quiet -of csv=\"p=0\""
    audio_duration = float(os.popen(cmd).read())
    total_frame = math.ceil(audio_duration * 25)
    try:
        local_video = wav_to_lip_with_chin(audio_path, video_path, model_weight,
                                           with_sound=True, base_dir=base_dir,
                                           start_index=start_index,
                                           total_frame=total_frame, 
                                           with_restorer=True,
                                           task_logger=logger, file_prefix=f'scene_{scene_index}')
    except EnhanceException as e:
        traceback.print_exc()
        raise e
    except Exception as e:
        traceback.print_exc()
        msg = "推理失败 {}".format(e)
        logger.error(msg)
        raise Exception(msg)
    finally:
        gc_collect()
    return local_video


model_path = None


def _infer_multiple_video(task_id: str, infer_request: InferRequest) -> None:
    """
    遍历推理每个场景, 并合成视频
    :param task_id: str, 任务id
    :param infer_request: InferRequest, 推理请求
    :return: None
    """
    global model_path
    
    
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    base_dir = f"./temp/{date_str}/{task_id}"
    os.makedirs(base_dir, exist_ok=True)

    width = infer_request.width
    height = infer_request.height
    resolution = (width, height)
    scenes = infer_request.scenes
    # scenes按sort排序
    scenes.sort(key=lambda x: x.sort)
    video_list = []
    task_logger = get_logger(__file__, task_id=task_id)
    for i in range(len(scenes)):
        task_logger.info("处理场景进度: {}/{}".format(i + 1, len(scenes)))
        scene = scenes[i]
        if len(scene.video_nodes) ==0 and len(scene.image_nodes) ==0 and len(scene.gif_nodes )==0 :
            if scene.model is None or scene.speaker is None:
                continue

        model = scene.model
        speaker = scene.speaker
        background = scene.background
        need_infer = True
        # 如果有model则推理
        if model is None or speaker is None:
            task_logger.warning("model or speaker is None, skip inference")
            video_url = None
            mask_video_url = None
            audio_url = None
            need_infer = False
        else:
            video_url = model.url
            mask_video_url = model.maskUrl
            audio_url = speaker.url

        readers = []
        if need_infer:
            task_logger.info('step1. 推理视频')
            # 现在需要推理的音频和视频
            video_path = download_file_with_dir(video_url, local_dir=CACHE_DIR)
            audio_path = download_file_with_dir(audio_url, local_dir=CACHE_DIR)
            if mask_video_url:
                mask_path = download_file_with_dir(mask_video_url, local_dir=CACHE_DIR)
                model.mask_path = mask_path
            try:
                model_weight = model.weight_path if model.weight_path is not None and os.path.isfile(
                    model.weight_path) else model_path
                infer_video_path = infer_single_video(video_path,
                                                      audio_path,
                                                      model_weight=model_weight,
                                                      start_index=model.startFrameIndex,
                                                      base_dir=base_dir,
                                                      logger=task_logger,
                                                      scene_index=i)
            except EnhanceException as e:
                callback_to_url(infer_request.callback, Code.ENHANCE_ERROR, f"{e}", task_id, server_id=server_id)
                release_job(task_id)
                return
            except Exception as e:
                callback_to_url(infer_request.callback, Code.INFER_ERROR, f"{e}", task_id, server_id=server_id)
                release_job(task_id)
                return
            inference_video_reader = create_reader(
                reader_type=VIDEO_READER,
                file_path=infer_video_path,
                width=model.width,
                height=model.height,
                top=model.marginTop,
                left=model.marginLeft,
                angle=model.rotate,
                opacity=model.opacity,
                flip=model.flip,
                start_at=model.startAt,
                end_at=model.endAt,
                layer=model.layer,
                logger=task_logger,
                start_index=model.startFrameIndex,
                mask_path=model.mask_path
            )
            readers.append(inference_video_reader)
        else:
            task_logger.info('step1. 不需要推理视频')


        task_logger.info('step2. 多线程下载素材')
        nodes = scene.image_nodes + scene.audio_nodes + scene.video_nodes + scene.gif_nodes + scene.text_nodes
        remote_url_types = [ImageNodeItem, AudioNodeItem, VideoNodeItem, GifNodeItem]
        for node in nodes:
            url = node.url if type(node) in remote_url_types else node.fontFamily
            file_path = download_file_with_dir(url, local_dir=CACHE_DIR)
            width = getattr(node, 'width', 0)
            height = getattr(node, 'height', 0)
            top = getattr(node, 'marginTop', 0)
            left = getattr(node, 'marginLeft', 0)
            angle = getattr(node, 'rotate', 0)
            opacity = getattr(node, 'opacity', 100)
            flip = getattr(node, 'flip', 0)
            start_at = getattr(node, 'startAt', 0)
            end_at = getattr(node, 'endAt', 0)
            start_offset = getattr(node, 'startOffset', 0)
            end_offset = getattr(node, 'endOffset', 0)
            layer = getattr(node, 'layer', 0)
            text = getattr(node, 'text', None)
            align = getattr(node, 'align', 'center')
            fontsize = getattr(node, 'fontSize', 0)
            color = getattr(node, 'color', '#000000')
            stroke_color = getattr(node, 'strokeColor', '#000000')
            stroke_width = getattr(node, 'stroke', 0)
            letter_spacing = getattr(node, 'letterSpacing', 0)

            class_name = node.__class__.__name__
            reader_type_dict = {
                'ImageNodeItem': IMAGE_READER,
                'AudioNodeItem': AUDIO_READER,
                'VideoNodeItem': VIDEO_READER,
                'GifNodeItem': GIF_READER,
                'TextNodeItem': TEXT_READER,
            }
            reader_type = reader_type_dict.get(class_name, None)
            reader = create_reader(
                reader_type=reader_type,
                file_path=file_path,
                width=width,
                height=height,
                top=top,
                left=left,
                angle=angle,
                opacity=opacity,
                flip=flip,
                start_at=start_at,
                end_at=end_at,
                start_offset=start_offset,
                end_offset=end_offset,
                layer=layer,
                logger=task_logger,
                text=text,
                fontsize=fontsize,
                align=align,
                color=color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                letter_spacing=letter_spacing,
            )
            readers.append(reader)

        task_logger.info('step3. 合成视频')
        with VideoBuilder(readers,
                          resolution=resolution,
                          background=background,
                          logger=task_logger) as video_builder:
            video_path = video_builder.build(base_dir=base_dir)
        video_item = CombinedVideoItem(video_path, transition=scene.transition)
        video_list.append(video_item)
    task_logger.info('step4. 合并视频')
    combined_video_list = CombinedVideoList(video_list)
    final_video_path = combined_video_list.merge(target_dir=base_dir, logger=task_logger)
    task_logger.info(f'step5. 上传最终视频 {final_video_path}')
    try:
        obs_path = upload_to_obs(final_video_path, logger=task_logger)
        abs_url = obs_config.get("base_url") + obs_path
    except Exception as e:
        # 获取错误的栈
        traceback.print_exc()
        callback_to_url(infer_request.callback, Code.UNKOWN_ERROR, f"{e}", task_id, server_id=server_id)
        release_job(task_id)
        return
    body = {"url": abs_url}
    task_logger.info("step6. callback to url: {}, body : {}".format(infer_request.callback, body))
    try:
        res_json = callback_to_url(infer_request.callback, Code.OK, "success", task_id, body, server_id=server_id)
    except Exception as e:
        traceback.print_exc()
        task_logger.error('callback error {}'.format(e))
        release_job(task_id)
        return
    release_job(task_id)
    task_logger.info("step7. get response {}".format(res_json))


@app.post("/infer_multiple_video")
def infer_multi_video(infer_request: InferRequest, background_tasks: BackgroundTasks):
    global running_thread_count, MAX_TASK_COUNT, running_task_ids
    if running_thread_count >= MAX_TASK_COUNT:
        result = Result(Code.OVER_THREAD, "too many request, please try again later").to_dict()
        response = JSONResponse(content=result, status_code=200)
        response.headers['X-Server-ID'] = server_id
        return response
    handle_running_count(1)
    task_id = str(uuid4())[:8]
    # infer_request转成dict
    infer_dict = infer_request.dict()
    # 保存到本地缓存
    with open(os.path.join(JOB_DIR, f"{task_id}.json"), "w") as f:
        json.dump(infer_dict, f)
    running_task_ids.append(task_id)
    background_tasks.add_task(_infer_multiple_video, task_id, infer_request)
    result = Result(Code.OK, "success", data={"task_id": task_id}).to_dict()
    response = JSONResponse(content=result, status_code=200)
    response.headers['X-Server-ID'] = server_id
    return response


# 删除任务, todo
@app.delete("/task/{task_id}")
def delete_task(task_id: str):
    pass


@app.get("/weights")
def get_weights():
    """
    获取模型可用权重
    """
    weight_path = 'weights'
    weights = glob.glob(os.path.join(weight_path, "*.pth"))
    result = Result(Code.OK, "success", data={"weights": weights}).to_dict()
    response = JSONResponse(content=result, status_code=200)
    response.headers['X-Server-ID'] = server_id
    return response


def get_lan_ip():
    """获取局域网IP地址"""
    import socket
    try:
        # 创建一个socket对象
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 使用Google的公共DNS服务器地址和一个不常用的端口号
        # 这个操作不会真的发送数据包，但会要求操作系统解析路由
        s.connect(("8.8.8.8", 80))
        # 获取socket对象的当前活动接口的IP地址
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        server_logger.error("获取局域网ip失败, " + str(e))
        raise e


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-c", "--config", type=str, required=True, help="Path to yaml configuration file")
    args.add_argument("-m", "--model", type=str, default=None, help="model ckpt to cover default model in config file")
    args.add_argument("--id", type=str, default="server-0", help="server id")
    args.add_argument("--debug", action="store_true", help="debug mode")
    args = args.parse_args()
    server_id = args.id
    debug = args.debug
    if os.path.exists(args.config) is False:
        raise ValueError("Configuration file does not exist.")
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    check_point = config["check_point"]
    check_point = args.model if args.model else check_point
    if check_point is None or os.path.exists(check_point) is False:
        raise ValueError("check point does not exist.")
    model_path = check_point
    obs_config = config["obs"]
    #host = "0.0.0.0"  # get_lan_ip()
    host="127.0.0.1"
    port = config["port"]
    os.makedirs(JOB_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    to_do_jobs = glob.glob(os.path.join(JOB_DIR, "*.json"))
    server_logger.warning(f'has {len(to_do_jobs)} jobs to be done')
    for job_json in to_do_jobs:
        with open(job_json, "r") as f:
            infer_request = InferRequest(**json.load(f))
        _infer_multiple_video(job_json.split("/")[-1].split(".")[0], infer_request)
    server_logger.info(f'about to start server at {host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level="info")
