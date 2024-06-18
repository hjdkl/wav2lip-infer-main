import hashlib
import time
import uuid
from os import listdir, path
import numpy as np
import scipy, cv2, os, audio

import subprocess
from tqdm import tqdm
import torch, face_detection
from typing import Any, Tuple, List

from models import Wav2LipHQ
from merge_face import MergeFace
from hparams import hparams
from gfpgan import GFPGANer
from utils import *
from exceptions import *
from merge_face import create_mask, draw_img_with_mask

logger = get_logger(__file__)

# 面部识别batch_size
face_det_batch_size = 4
# 唇形转化batch_size
wav2lip_batch_size = 8
pads = [0, 15, 0, 0]
crop = [0, -1, 0, -1]
box = [-1, -1, -1, -1]
img_size = 96


# todo, 这个函数是干什么的
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def cache_face_coords(full_frames: List[np.ndarray], video_path: str, logger=None) -> str:
    """
    检测图片中的人脸, 并缓存人脸的坐标
    args:
        full_frames: 图片数组
        video_path: 视频路径
        logger: 日志
    returns:
        cache_file: 缓存文件路径
    """
    md5_name = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    base_dir = "./temp/face_detect/"
    os.makedirs(base_dir, exist_ok=True)
    cache_file = os.path.join(base_dir, md5_name + '.txt')
    # 存在缓存则直接返回
    if os.path.isfile(cache_file):
        logger.info("从缓存中获取人脸坐标")
        return cache_file
    # 不存在缓存则进行检测
    logger.info("从视频中检测人脸")
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(full_frames), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(full_frames[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            logger.info('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, full_frames):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        # 面部上、左、右各延伸5%
        width = x2 - x1
        height = y2 - y1
        extra_w = int(width * 0.05)
        extra_h = int(height * 0.05)
        x1 = x1 - extra_w
        x2 = x2 + extra_w
        y2 = y2 + extra_h
        results.append([x1, y1, x2, y2])
    with open(cache_file, "w") as f:
        for i, cor in enumerate(results):
            line = ",".join([str(x) for x in cor])
            if i == len(results) - 1:
                f.write(line)
            else:
                f.write(line + "\n")
    return cache_file


def get_cached_face_cors(cache_file: str) -> list:
    """
    从缓存文件中获取人脸坐标
    args:
        cache_file: 缓存文件路径
    returns:
        results: 人脸坐标数组, [[x1, y1, x2, y2], ...]
    """
    if not os.path.isfile(cache_file):
        raise ValueError("cache file not exist")
    results = []
    with open(cache_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            results.append([int(x) for x in line.strip().split(",")])
    return results


def get_faces_from_cache(full_frames: List[np.ndarray], cache_file: str) -> list:
    """
    从缓存中获取人脸的图片
    args:
        full_frames: 所有帧
        cache_file: 缓存文件路径
    returns:
        results: [[face, (x1, y1, x2, y2)], ...]
    """
    face_cors = get_cached_face_cors(cache_file)
    boxes = np.array(face_cors)
    boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (int(y1), int(y2), int(x1), int(x2))] for image, (x1, y1, x2, y2) in
               zip(full_frames, boxes)]
    return results


def datagen(frames, mels, cache_file, start_index=0) -> Any:
    """数据生成的迭代器
    args:
        frames: 图片帧
        mels: 音频mel
        cache_file: 缓存文件路径
        start_index: 开始帧
    returns:
        img_batch: 图片
        mel_batch: 音频mel
        frame_batch: 帧
        coords_batch: 坐标
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    face_det_results = get_faces_from_cache(frames, cache_file)
    for i, mel in enumerate(mels):
        idx = (i + start_index) % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (hparams.img_size, hparams.img_size_height))  # 192x288
        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, hparams.img_size_height // 2:] = 0  # 192x288
            # img_masked[:, img_size // 2:] = 0  # 96x96

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, hparams.img_size_height // 2:] = 0  # 192x288
        # img_masked[:, img_size // 2:] = 0  # 96x96

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16  # todo, 这个参数是什么意思

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))
ON_CUDA = device == 'cuda'


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    elif device == 'mps':
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(checkpoint_path: str, logger=None):
    model = Wav2LipHQ()
    logger.info("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = _load(checkpoint_path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def get_frames(video_path: str):
    """
    获取视频帧
    args:
        video_path: 视频路径
    returns:
        frames: 帧数组
    """
    video_stream = cv2.VideoCapture(video_path)
    frames = []
    while 1:
        ret, frame = video_stream.read()
        if not ret:
            video_stream.release()
            break
        frames.append(frame)
    video_stream.release()
    return frames


def get_mel_chunks(audio_path: str, fps: int = 25, total_frame: int = 0):
    """获取音频mel
    args:
        audio_path: 音频路径
        fps: 帧率
        total_frame: 总帧数
    returns:
        mel_chunks: mel数组
    """
    audio_path = convert_audio_to_16k(audio_path)
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    mel_chunks = []
    mel_idx_multiplier = 80. / fps  # mel和帧的比例, 会造成少一些帧
    """
    fps固定是25, mel_idx_multiplier是3.2
    去整后, mel_idx_multiplier是3, 丢失了精度
    结合mel_step_size, 会造成提前break
    total_frame 用来解决少帧的问题
    """
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    if total_frame == 0:
        return mel_chunks
    missing_count = total_frame - len(mel_chunks)
    last_mel = mel_chunks[-1].copy()
    if missing_count > 0:
        mel_chunks.extend([last_mel] * missing_count)
    return mel_chunks


def wav_to_lip_with_chin(
        audio_path: str,
        video_path: str,
        model_path: str,
        with_sound: bool = True,
        start_index: int = 0,
        base_dir: str = "./temp",
        total_frame: int = 0,
        with_restorer: bool = False,
        task_logger: Any = None,
        file_prefix: str = "",
) -> str:
    """
    推理audio to lip
    args:
        audio_path: 音频路径
        video_path: 视频路径
        model_path: 模型路径
        with_sound: 是否合并音频
        start_index: 开始帧
        base_dir: 临时文件夹
        total_frame: 总帧数
        with_restorer: 是否使用面部修复模型
        task_logger: 日志
        file_prefix: 文件前缀
    returns:
        outfile: 输出文件路径
    """
    if task_logger is None:
        task_logger = logger
    model = load_model(model_path, logger=task_logger)
    if with_restorer and ON_CUDA:
        # 面部修复模型
        restorer = GFPGANer(
            model_path="./gfpgan/weights/GFPGANv1.3.pth",
            upscale=1,
            arch='clean',
            channel_multiplier=2
        )

    assert os.path.isfile(video_path), 'video argument must be a valid path to video/image file'
    # 推理模型的stream（可以优化, 提前获取）
    video_stream = cv2.VideoCapture(video_path)
    # 固定帧率
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))
    assert fps == 25, f"视频帧率必须是25, 当前帧率是{fps}"

    # 获取推理视频的帧
    task_logger.info('step1. 获取视频帧')
    full_frames = get_frames(video_path)

    # 提前把所有帧的人脸检测坐标缓存起来
    task_logger.info('step2. 缓存人脸检测结果')
    cache_file = cache_face_coords(full_frames, video_path, logger=task_logger)

    task_logger.info('step3. 获取音频mel')
    mel_chunks = get_mel_chunks(audio_path, fps, total_frame=total_frame)

    task_logger.info("step4. 生成数据")
    gen = datagen(full_frames.copy(), mel_chunks, cache_file, start_index=start_index)

    task_logger.info("step5. 写入视频")
    batch_size = wav2lip_batch_size
    file_base_name = file_prefix if file_prefix != "" else uuid.uuid4().hex
    tmp_file = os.path.join(base_dir, file_base_name + '.avi')
    total = int(np.ceil(float(len(mel_chunks)) / batch_size))
    # scale = 1.1  # 用于扩大融合的face, 方便做边缘羽化
    # todo 面部融合没做代码检查
    with MergeFace() as merge_face:
        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            if i % 10 == 0 or i == total - 1:
                cur = total if i == total - 1 else i
                task_logger.info(f"推理进度: {cur}/{total}")
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(tmp_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            with torch.no_grad():
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for gen_face, frame, coord in zip(pred, frames, coords):
                y1, y2, x1, x2 = coord
                gen_face = cv2.resize(gen_face.astype(np.uint8), (x2 - x1, y2 - y1))
                _, mask = merge_face.get_face_small(gen_face)
                if with_restorer and ON_CUDA:
                    try:
                        _, _, gen_face = restorer.enhance(
                            gen_face,
                            has_aligned=False,
                            only_center_face=True,
                            paste_back=True,
                            weight=0.9,
                        )
                    except Exception as e:
                        task_logger.error(e)
                        raise EnhanceException("面部增强失败")
                result = draw_img_with_mask(gen_face, mask, frame, ksize=29, top=y1, left=x1)
                out.write(result)
        out.release()

    video_stream.release()
    video_stream = None
    full_frames = None
    mel_chunks = None
    gen = None
    del model
    if with_restorer and ON_CUDA:
        del restorer

    if not with_sound:
        return tmp_file
    outfile = os.path.join(base_dir, uuid.uuid4().hex + '.mp4')
    task_logger.info("step6. 合并音频, 使用最高质量")
    command = f'ffmpeg -loglevel warning -y -i {tmp_file} -i {audio_path} -c:v libx264 -crf 18 -strict -2 -q:v 0 -shortest {outfile}'
    subprocess.call(command, shell=True)
    os.remove(tmp_file)
    return outfile


if __name__ == '__main__':
    a = time.time()
    wav_to_lip_with_chin("xiaoyi.wav", "wang_demo.mp4", "weights/pt24.pth", with_restorer=True)
    b = time.time()
    print("time", b - a)

