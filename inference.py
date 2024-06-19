import hashlib
import sys
import time
import cv2
import torch
import uuid
from basicsr.utils import img2tensor, tensor2img
from infer_utils import *
from utils import get_logger
import numpy as np
import cupy as cp

from os import listdir, path
import scipy, cv2, os, audio


from typing import Any, List
from hparams import hparams
from torchvision.transforms.functional import normalize
from resnet50.Resnet50v1 import init_detection_model
from wav2lip.wav2lip_hq import init_wav2lip
from gfpgan_infer import init_gfpgan
from parnet.Parsenet import init_parsenet
import subprocess

logger = get_logger(__file__)

##模型运行过程是完全分开的batch_size可以独立设置print(face_input[2])
# 面部识别batch_size
face_det_batch_size = 16
# 唇形转化batch_size
wav2lip_batch_size = 8
# 面部超分batch_size
gfpgan_batch_size = 8
# 面部粘贴batch_size
parse_batch_size = 8


# 数据准备函数

def get_frames(video_path: str):
    """
    获取视频帧
    args:
        video_path: 视频路径
    returns:
        frames: 帧列表
    """
    global frame_num
    video_stream = cv2.VideoCapture(video_path)
    frames = []
    while 1:
        ret, frame = video_stream.read()
        if not ret:
            video_stream.release()
            break
        frames.append(frame)
        frame_num += 1
    video_stream.release()
    return frames

def get_face_from_cache(video_path: str,frequency:str):
    md5_name = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    base_dir = "temp/face_detect/"
    os.makedirs(base_dir, exist_ok=True)
    box_file = os.path.join(base_dir, md5_name +"face"+"_"+frequency+".npy")
    mark_file= os.path.join(base_dir, md5_name +"mask"+"_"+frequency+".npy")
    # 存在缓存则直接返回
    face_list=None
    all_landmarks_5=None
    if os.path.isfile(box_file):
        logger.info("从缓存中获取人脸框")
        cache=np.load(box_file).astype(np.int16)
        face_list=[]
        for i in cache:
            face_list.append(i)
    if os.path.isfile(mark_file):
        logger.info("从缓存中获取面部坐标")
        cache = np.load(mark_file).astype(np.int16)
        all_landmarks_5 = []
        for i in cache:
            all_landmarks_5.append(i)
    if face_list!=None and all_landmarks_5!=None and len(face_list)!=len(all_landmarks_5):
        print("错误文件名：",box_file,mark_file)
        raise ValueError("缓存错误，长度不一致")
    return face_list,all_landmarks_5




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


# 获取对齐图像和仿射矩阵
def align_warp_face(input_img, landmark, border_mode='constant'):
    affine_matrix = cv2.estimateAffinePartial2D(landmark, face_template, method=cv2.LMEDS)[0]
    if border_mode == 'constant':
        border_mode = cv2.BORDER_CONSTANT
    elif border_mode == 'reflect101':
        border_mode = cv2.BORDER_REFLECT101
    elif border_mode == 'reflect':
        border_mode = cv2.BORDER_REFLECT
    cropped_face = cv2.warpAffine(input_img, affine_matrix, (512, 512), borderMode=border_mode,
                                  borderValue=(135, 133, 132))  # gray
    return cropped_face, affine_matrix


def paste_faces_to_input_image(input_img:List, restored_face:List, affine_matrix:list, face_parse):
    """
    脸部粘贴回原来的面部
    input_img: 整个的图片
    restored_face: 推理后，修复好的人脸
    """
    h, w, _ = input_img[0].shape


    # 把推理后的人脸放到cuda里
    face_input_list=[]
    for i in range(len(restored_face)):
        # inference 准备
        face_input = cv2.resize(restored_face[i], (512, 512), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("face input ", face_input)
        # cv2.waitKey(0)
        face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
        normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_input_list.append(face_input)
    face_input = torch.stack(face_input_list, 0).to(device)
    del face_input_list
#!s = time.time()
    with torch.no_grad():
        result,_ = face_parse(face_input)
    result = result.cpu().numpy()
    #! print('infer ', time.time() - s)
   #! s = time.time()

    output_img = []
    for i in range(len(restored_face)):
        out=result[i].argmax(axis=0)

        mask = np.zeros(out.shape)
        
        MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
        for idx, color in enumerate(MASK_COLORMAP):
            mask[out == idx] = color
        #  blur the mask
        mask = cv2.GaussianBlur(mask, (49, 49), 11)
        # mask = cv2.GaussianBlur(mask, (101, 101), 11)
        # remove the black borders
        thres = 10
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        
        mask = mask / 255.
        # mask可以缓存
        mask = cv2.resize(mask, restored_face[i].shape[:2])
        # cv2.imshow('mask', mask)
        # cv2.waitKey(1)
        # 反仿射变换
        #！i_s = time.time()
        inverse_affine = cv2.invertAffineTransform(affine_matrix[i])
        inv_restored = cv2.warpAffine(restored_face[i], inverse_affine, (w, h))

        mask = cv2.warpAffine(mask, inverse_affine, (w, h), flags=3)
        # print(mask.shape)
        # print(time.time() - i_s)
        inv_soft_mask = mask[:, :, None]
        pasted_face = inv_restored
        # print(pasted_face.shape)
        # cv2.imshow('inv_soft_mask', inv_restored)
        #！u_s = time.time()
        # TODO, 这块要加速，一个突破需要37毫秒，很慢
        # print(type(inv_soft_mask))
        # print(type(pasted_face))
        # print(type(input_img[i]))
        # inv_soft_mask = cp.asarray(inv_soft_mask)
        # pasted_face = cp.asarray(pasted_face)
        # input_img = cp.asarray(input_img[i])
        upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * input_img[i]
        upsample_img = cp.asnumpy(upsample_img)
        upsample_img = upsample_img.astype(np.uint8)
        # print('u_s', time.time() - u_s)
        # cv2.imshow('upsample_img', upsample_img)
        # cv2.waitKey(1)
        output_img.append(upsample_img)
   #！ print('back ', time.time() - s)
    return output_img


##数据生成器
def get_faces_from_cache(full_frames: List[np.ndarray], det_faces: list) -> list:
    """
    从缓存中获取人脸的图片
    args:
        full_frames: 所有帧
        det_faces: 对应的人脸列表
    returns:
        results: [[face, (x1,y1,x2,y2)], ...]
    """

    results = [[image[y1:y2, x1:x2], (x1, y1, x2, y2)] for image, (x1, y1, x2, y2) in
               zip(full_frames, det_faces)]
    return results


def datagen(frames, mels, det_faces, start_index=0) -> Any:
    """数据生成的迭代器
    args:
        frames: 图片帧
        mels: 音频mel
        start_index: 开始帧
    returns:
        img_batch: 图片
        mel_batch: 音频mel
    """
    img_batch, mel_batch= [], []
    face_det_results = get_faces_from_cache(frames, det_faces)
    for i, mel in enumerate(mels):
        idx = (i + start_index) % len(frames)
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (hparams.img_size, hparams.img_size_height))  # 192x288
        img_batch.append(face)
        mel_batch.append(mel)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, hparams.img_size_height // 2:] = 0  # 192x288
            # img_masked[:, img_size // 2:] = 0  # 96x96

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch
            img_batch, mel_batch = [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, hparams.img_size_height // 2:] = 0  # 192x288
        # img_masked[:, img_size // 2:] = 0  # 96x96

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch


#####------------主要代码从这里开始----------------------


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))
ON_CUDA = device == 'cuda'

frame_num = 0  # 总帧数声明
frame_idx = 0  # 进行到了那一帧


###对齐部分变量
face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],  # 脸部模板
                          [201.26117, 371.41043], [313.08905, 371.15118]])

face_size = 512  # 对齐后图片大小
face_template *= (face_size / 512.0)
# 宽高初始化
w = 0
h = 0


all_landmarks_5 = []  # 存放五官坐标用于对齐
det_faces = []  # 存放人脸框坐标

twice_all_landmarks_5_ = []  # 存放二次检测五官坐标
twice_det_faces = []  # 存放二次检测人脸框坐标
#存放所有视频帧
full_frames=[]
#存放矫正人脸
face_img=[]
#wav2lip结果
wav2_result=[]
#gfpgan结果
gfp_result=[]

affine_matrices = []  # 仿射矩阵(对齐要用)

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
        file_prefix: str = ""
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
    assert os.path.isfile(video_path), 'video argument must be a valid path to video/image file'
    # 推理模型的stream（可以优化, 提前获取）
    video_stream = cv2.VideoCapture(video_path)
    # 固定帧率
    fps = int(video_stream.get(cv2.CAP_PROP_FPS))
    assert fps == 25, f"视频帧率必须是25, 当前帧率是{fps}"

    global frame_idx, wav2_result, gfp_result, affine_matrices, full_frames, frame_num,face_img
    frame_idx=0
    wav2_result=[]
    gfp_result=[]
    affine_matrices=[] 
    full_frames=[]
    frame_num=0
    face_img=[]
    
    # 模型初始化
    face_det = init_detection_model("retinaface_resnet50", half=True, device=device)
    wav2lip = init_wav2lip(model_path, device, logger=task_logger)
    gfpgan = init_gfpgan()
    parnet = init_parsenet()
    # 获取推理视频的帧
    task_logger.info('step1. 获取视频帧')

    full_frames = get_frames(video_path)  # 返回所所有帧的数
    task_logger.info('step2. 获取音频mel')
    mel_chunks = get_mel_chunks(audio_path, fps, total_frame=total_frame)  ##返回所有音频的列表

    global det_faces, all_landmarks_5, twice_det_faces, twice_all_landmarks_5

    task_logger.info('step3. 获得人脸')
    #缓存人脸存在直接获取，不存在就检测
    det_faces,all_landmarks_5=get_face_from_cache(video_path,"once")
    if det_faces==None or all_landmarks_5==None:
        md5_name = hashlib.md5(video_path.encode("utf-8")).hexdigest()
        cache_dir = "temp/face_detect/"
        os.makedirs(base_dir, exist_ok=True)
        box_file = os.path.join(cache_dir, md5_name + "face_once" )
        mark_file = os.path.join(cache_dir, md5_name + "mask_once")
        det_faces=[]
        all_landmarks_5=[]
        task_logger.info('检测人脸')
        with torch.no_grad():
            total = int(np.ceil(len(full_frames) / face_det_batch_size))
            for i in range(total):
                if i != total - 1:
                    img = full_frames[i * face_det_batch_size:(i + 1) * face_det_batch_size]
                elif i == total - 1:
                    img = full_frames[i * face_det_batch_size:]
                boxs, landmark = face_det.detect_faces(img,
                                                       0.97)  # boxs为人脸二维数组，数组的每个元素为长15的列表0~3定位人脸(只有脸部)，4为置信度，后5~14定位眼睛鼻子嘴巴
                all_landmarks_5 += landmark  # 五官位置用于对齐
                det_faces += boxs  # 人脸框[x1,y1,x2,y2]
        np.save(box_file,np.stack(det_faces))
        np.save(mark_file,np.stack(all_landmarks_5))
    else:
        print('已在缓存中获得')

    # 矫正
    task_logger.info('获取正着的人脸')
    for i in range(len(full_frames)):
        face, affine_matrix = align_warp_face(full_frames[i], all_landmarks_5[i])
        face_img.append(face)
        affine_matrices.append(affine_matrix)

    #二次人脸检测
    task_logger.info('二次检测，获取矫正后的人脸的位置')
    # 缓存人脸存在直接获取，不存在就检测
    twice_det_faces, twice_all_landmarks_5 = get_face_from_cache(video_path,"twice")
    if twice_det_faces == None or twice_all_landmarks_5 == None:
        md5_name = hashlib.md5(video_path.encode("utf-8")).hexdigest()
        cache_dir = "temp/face_detect/"
        os.makedirs(base_dir, exist_ok=True)
        box_file = os.path.join(cache_dir, md5_name + "face_twice")
        mark_file = os.path.join(cache_dir, md5_name + "mask_twice")
        twice_det_faces = []
        twice_all_landmarks_5 = []
        with torch.no_grad():
            with torch.no_grad():
                total = int(np.ceil(len(face_img) / face_det_batch_size))
                for i in range(total):
                    if i != total - 1:
                        img = face_img[i * face_det_batch_size:(i + 1) * face_det_batch_size]  #face_img.shape=(512,512,3)
                    elif i == total - 1:
                        img = face_img[i * face_det_batch_size:]
                    boxs, landmark = face_det.detect_faces(img,
                                                           0.97)  # boxs为人脸二维数组，数组的每个元素为长15的列表0~3定位人脸(只有脸部)，4为置信度，后5~14定位眼睛鼻子嘴巴
                    twice_all_landmarks_5 += landmark  # 五官位置用于对齐
                    twice_det_faces += boxs  # 人脸框[x1,y1,x2,y2]
            scaling_ratio=0.1   #框适当放大
            for i in range(len(twice_det_faces)):
                x1, y1, x2, y2=[j for j in twice_det_faces[i]]
                h=(y2-y1)*0.1//2
                w=(x2-x1)*0.1//2
                x1=x1-w if x1-w>0 else 0
                y1 =y1-h if y1-h>0 else 0
                x2 = x2+ w if x2 +w <512 else 512
                y2 = y2+ h if y2 +h <512 else 512
                twice_det_faces[i]=np.array([x1,y1,x2,y2]).astype(np.int16)
        np.save(box_file, np.stack(twice_det_faces))
        np.save(mark_file, np.stack(twice_all_landmarks_5))
    else:
        print('第二次人脸检测，已在缓存中获得')
    

    task_logger.info("step4. 生成数据")
    gen = datagen(face_img.copy(), mel_chunks, twice_det_faces, start_index=start_index)  # 数据迭代器具

    task_logger.info("step5.模型处理")

    file_base_name = file_prefix if file_prefix != "" else uuid.uuid4().hex
    tmp_file = os.path.join(base_dir, file_base_name + '.avi')




    # wav2lip
    with torch.no_grad():
        total=int(np.ceil(float(len(mel_chunks)) / wav2lip_batch_size))
        for i, (img_batch, mel_batch) in enumerate(gen):
            # print(img_batch.shape)
            # print(type(img_batch))
            # cv2.imshow("aa",img_batch[0,:,:,0:3])
            # cv2.imshow("bb", img_batch[0, :, :, 3:7])
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            if i % 10 == 0 or i == total - 1:
                cur = total if i == total - 1 else i
                task_logger.info(f"wav2推理进度: {cur}/{total}")
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(tmp_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            pred = wav2lip(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for j in range(pred.shape[0]):
                wav2_result.append(pred[j].astype(np.uint8))
    wav2_frame_num=len(wav2_result)

    if wav2_frame_num>frame_num:
        full_frames+=full_frames[0:wav2_frame_num-frame_num]
        all_landmarks_5+=all_landmarks_5[0:wav2_frame_num-frame_num]
        det_faces += det_faces[0:wav2_frame_num-frame_num]
        face_img += face_img[0:wav2_frame_num - frame_num]
        twice_det_faces += twice_det_faces[0:wav2_frame_num - frame_num]
        affine_matrices+=affine_matrices[0:wav2_frame_num - frame_num]
        frame_num=wav2_frame_num
    del wav2lip
    frame_idx = 0
    # gfpgan
    with torch.no_grad():
        total = int(np.ceil(wav2_frame_num / gfpgan_batch_size))
        for i in range(total):
            if i % 10 == 0 or i == total - 1:
                cur = total if i == total - 1 else i
                task_logger.info(f"gfp推理进度: {cur}/{total}")

            if i!=total - 1:
                gfp_batch = wav2_result[i*gfpgan_batch_size:(i+1)*gfpgan_batch_size]
            elif i==total - 1:
                gfp_batch=wav2_result[i*gfpgan_batch_size:]
            for j in range(len(gfp_batch)):
                gen_face=wav2_result[frame_idx]
                x1, y1, x2, y2 =  twice_det_faces[frame_idx]
                gen_face = cv2.resize(gen_face, (x2 - x1, y2 - y1))
                # cv2.imshow("gen_face", gen_face)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                ##对齐人脸

                # 原图粘贴
                background = face_img[frame_idx].copy()
                background[y1:y2, x1:x2, :] = gen_face
                # cv2.imshow("background", background)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

                cropped_face_t = img2tensor(background / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                frame_idx+=1
                gfp_batch[j]=cropped_face_t
            gfp_batch=torch.stack(gfp_batch,dim=0).to(device)
            output = gfpgan(gfp_batch, return_rgb=False, randomize_noise=False)[0]
            for j in range(len(gfp_batch)):
                # convert to image
                restored_face = tensor2img(output[j], rgb2bgr=True, min_max=(-1, 1))
                restored_face = restored_face.astype('uint8')
                gfp_result.append(restored_face)
                # cv2.imshow("gfp", restored_face)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()
    del wav2_result,all_landmarks_5,gfpgan

    frame_idx=0
    # parnet 粘贴人脸

    #！back_s = time.time()
    with torch.no_grad():
        total = int(np.ceil(wav2_frame_num / parse_batch_size))
        for i in range(total):
            if i % 10 == 0 or i == total - 1:
                cur = total if i == total - 1 else i
                task_logger.info(f"粘贴进度: {cur}/{total}")
            if i != total - 1:
                input_img= full_frames[i * parse_batch_size:(i + 1) * parse_batch_size]
                gfp_img=gfp_result[i * parse_batch_size:(i + 1) * parse_batch_size]
                input_affine_matrices=affine_matrices[i * parse_batch_size:(i + 1) * parse_batch_size]
                result = paste_faces_to_input_image(input_img, gfp_img, input_affine_matrices, parnet)
                for j in result:
                    out.write(j)
            elif i == total - 1:
                input_img = full_frames[i * parse_batch_size:]
                gfp_img = gfp_result[i * parse_batch_size:]
                input_affine_matrices = affine_matrices[i * parse_batch_size:]
                result = paste_faces_to_input_image(input_img, gfp_img, input_affine_matrices, parnet)
                for j in result:
                    out.write(j)
    #！print(time.time() - back_s)

    out.release()

    video_stream.release()

    if not with_sound:
        return tmp_file
    outfile = os.path.join(base_dir, uuid.uuid4().hex + '.mp4')
    task_logger.info("step6. 合并音频, 使用最高质量")
    command = f'ffmpeg -loglevel warning -y -i {tmp_file} -i {audio_path} -strict -2 -q:v 0 -shortest {outfile}'
    subprocess.call(command, shell=True)
    os.remove(tmp_file)
    return outfile


if __name__ == '__main__':
    a = time.time()
    print(wav_to_lip_with_chin("test.mp3_16k.wav", "test.mp4", "pt35.pth"))
    
    # print(main("北影推理视频和音频/北影音频.wav", "beiying.mp4", "weight/yz.pth"))
    b = time.time()
    print("time", b - a)

