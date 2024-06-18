import math
from inference import load_model, wav_to_lip_with_chin
import argparse
import os


def infer(check_point_path, face_path, audio_path, start_index, hq):
    # 获取audio的时间
    cmd = "ffprobe -i {} -show_entries format=duration -v quiet -of csv=\"p=0\"".format(audio_path)
    result = os.popen(cmd)
    total_time = float(result.read())
    # 计算总帧数，并向上取整
    total_frame = math.ceil(total_time * 25)
    print("total frame: {}".format(total_frame))
    res = wav_to_lip_with_chin(audio_path, face_path, model_path=check_point_path, start_index=start_index,
                               with_sound=True, total_frame=total_frame, with_restorer=hq)
    return res


def get_file_list(dir_path):
    files = []
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(".mp4"):
                files.append(os.path.join(dir_path, file))
        return files
    else:
        return [dir_path]


def main(check_point_path, face_path, audio_path, start_index, hq):
    if not os.path.exists(check_point_path):
        raise Exception("check point not exist")
    if not os.path.exists(face_path):
        raise Exception("face not exist")
    if not os.path.exists(audio_path):
        raise Exception("audio not exist")

    files = get_file_list(face_path)

    for file in files:
        base_name = "推理-" + os.path.basename(file)
        r = infer(check_point_path, file, audio_path, start_index, hq)
        os.rename(r, os.path.join(os.path.dirname(file), base_name))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-f", "--face", required=True, type=str)
    parser.add_argument("-a", "--audio", type=str, required=True)
    parser.add_argument('-s', '--start_index', type=int, default=0)
    parser.add_argument('--hq', action='store_true', help='use high quality mode')
    args = parser.parse_args()
    check_point_path = args.model
    face_path = args.face
    audio_path = args.audio
    start_index = args.start_index
    hq = args.hq
    main(check_point_path, face_path, audio_path, start_index, hq)

