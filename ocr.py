import json

import whisper
from paddleocr import PaddleOCR, draw_ocr
from util import *
import os
import shutil
import re

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
a = r'/mnt/data.coronaryct.1/ZhuYichen/data/test_video'
whisper_model = whisper.load_model("large")
for path in os.listdir(a):
    video_path = os.path.join(os.path.join(a, path), 'video.mp4')
    subtitle = list()
    try:
        whisper_result = whisper_model.transcribe(video_path)
        for segment in whisper_result['segments']:
            subtitle.append({
                'begin': int(segment['start']),
                'end': int(segment['end']),
                'text': segment['text'],
            })
    except Exception as e:
        print(e)
    features_path = os.path.join(os.path.join(a, path), 'features.json')
    with open(features_path, "r+") as file:
        data = json.load(file)
        data['features']['whisper'] = subtitle
        file.seek(0)  # Move the file pointer to the beginning of the file
        json.dump(data, file, indent=4,
                  ensure_ascii=False)  # Write the updated data back to the file
        file.truncate()  # Remove any remaining data after the end of the updated content

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
videos_path = r'/mnt/data.coronaryct.1/ZhuYichen/data/test_video'
videos = list()
for item in os.listdir(videos_path):
    # if int(item) == 12:
    #     continue
    ocr_path = os.path.join(videos_path, item, 'features.json')
    video_path = os.path.join(videos_path, item, 'video.mp4')
    try:
        data = loadvideo_decord_origin(video_path)
    except Exception as e:
        print(e)
        continue
    text_result = list()

    for i in range(data.shape[0]):
        result = ocr.ocr(data[i], cls=True)
        text_per_image = list()
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if line[1][1] < 0.8:
                    continue
                text = dict()
                text['pos'] = [line[0][0][0], line[0][0][1], line[0][2][0], line[0][2][1]]
                text['text'] = line[1][0]
                text_per_image.append(text)
        if len(text_per_image) > 0:
            text_result.append({
                'begin': i,
                'text': text_per_image
            })
    output = {
        'ocr': text_result
    }
    with open(ocr_path, "r+") as file:
        data = json.load(file)
        data['features']['ocr'] = text_result
        file.seek(0)  # Move the file pointer to the beginning of the file
        json.dump(data, file, indent=4,
                  ensure_ascii=False)  # Write the updated data back to the file
        file.truncate()  # Remove any remaining data after the end of the updated content
    # ocr_output = ''
    # for index, text_per_image in enumerate(text_result):
    #     if len(text_per_image) == 0:
    #         continue
    #     ocr_output += f'Second {index}: '
    #     for text in text_per_image:
    #         ocr_output += "'{}' {}, ".format(text['text'], text['pos'])
    #     ocr_output += '\n'
    # print(ocr_output)
# img_path = '/mnt/data.coronaryct.1/ZhuYichen/videochat/images/0000/output_image_0.jpg'
