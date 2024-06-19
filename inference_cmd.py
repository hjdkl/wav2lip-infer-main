import math
from inference import main as infer_main
import argparse
import os


def main(check_point_path, face_path, audio_path, start_index):
    if not os.path.exists(check_point_path):
        raise Exception("check point not exist")
    if not os.path.exists(face_path):
        raise Exception("face not exist")
    if not os.path.exists(audio_path):
        raise Exception("audio not exist")
    # 获取audio的时间
    cmd = "ffprobe -i {} -show_entries format=duration -v quiet -of csv=\"p=0\"".format(audio_path)
    result = os.popen(cmd)
    total_time = float(result.read())
    # 计算总帧数，并向上取整
    total_frame = math.ceil(total_time * 25)
    print("total frame: {}".format(total_frame))
    res = infer_main(audio_path, face_path, model_path=check_point_path, start_index=start_index, with_sound=True, total_frame=total_frame)
    print("已合成结果保存在 {}".format(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint_path", default="pt4-wav2lip471.pth", type=str)
    parser.add_argument("-f", "--face", default="temp/ori.jpg", type=str)
    parser.add_argument("-a", "--audio", type=str, default="temp/000001.wav")
    parser.add_argument('-s', '--start_index', type=int, default=0)

    args = parser.parse_args()
    check_point_path = args.checkpoint_path
    face_path = args.face
    audio_path = args.audio
    start_index = args.start_index
    main(check_point_path, face_path, audio_path, start_index)

