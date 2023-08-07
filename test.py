from collections import defaultdict

import math
import os
import numpy as np
import random
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
import ffmpeg
from fractions import Fraction

from models.tag2text import tag2text_caption
from util import *
import gradio as gr
from chatbot import *
from load_internvideo import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from simplet5 import SimpleT5
from models.grit_model import DenseCaptioning
# from paddleocr import PaddleOCR
import whisper
from datetime import datetime
import configparser
from Ifasr_new import RequestApi

config = configparser.ConfigParser()
config.read('configs.ini')
args = {
    'videos_path': config.get('Arguments', 'videos_path'),
    'openai_api_key': os.environ["OPENAI_API_KEY"],
    'output_path': config.get('Arguments', 'output_path'),
    'images_path': config.get('Arguments', 'images_path'),
    'evaluate_path': config.get('Arguments', 'evaluate_path'),
    'segment_length': int(config.get('Arguments', 'segment_length')),
    'remarks': config.get('Arguments', 'remarks'),
    'llm': config.get('Arguments', 'llm'),
    'predict': config.get('Arguments', 'predict') == 'True',
    'qa': config.get('Arguments', 'qa') == 'True',
    'evaluate': config.get('Arguments', 'evaluate') == 'True',
    'mode': config.get('Arguments', 'mode'),
    'qa_mode': config.get('Arguments', 'qa_mode'),
}


class VideoChat:
    def __init__(self):
        self.bot = ConversationBot()

    def load_model(self):
        image_size = 384
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])

        # define model
        self.model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size,
                                      vit='swin_b')
        self.model.eval()
        self.model = self.model.to(device)
        print("[INFO] initialize caption model success!")

        self.model_T5 = SimpleT5()
        if torch.cuda.is_available():
            self.model_T5.load_model(
                "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=True)
        else:
            self.model_T5.load_model(
                "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=False)
        print("[INFO] initialize summarize model success!")
        # action recognition
        self.intern_action = load_intern_action(device)
        self.trans_action = transform_action()
        self.topil = T.ToPILImage()
        print("[INFO] initialize InternVideo model success!")

        self.dense_caption_model = DenseCaptioning(device)
        self.dense_caption_model.initialize_model()
        print("[INFO] initialize dense caption model success!")

        self.whisper_model = whisper.load_model("large")
        print("[INFO] initialize ASR model success!")

    def inference(self, video_path, input_tag):
        # Whisper
        subtitle = ''
        try:
            whisper_result = self.whisper_model.transcribe(video_path)
            for segment in whisper_result['segments']:
                subtitle += str(int(segment['start'])) + 's-' + str(int(segment['end'])) + 's: ' + segment[
                    'text'] + '\n'
        except Exception as e:
            subtitle = 'No subtitle'
            print(e)

        # 讯飞API
        # try:
        #     api = RequestApi(appid="d1c4610a",
        #                      secret_key="baf471c32e59924820be4c5c18027247",
        #                      upload_file_path=video_path)
        #
        #     api.get_result()
        #     subtitle = api.result2text()
        # except Exception as e:
        #     subtitle = 'No subtitle'
        #     print(e)

        data_seg, time_index = loadvideo_decord_time_segment(video_path, segment_length=args['segment_length'])
        predictions = []
        # InternVideo
        for data in data_seg:
            action_index = np.linspace(0, len(data) - 1, 8).astype(int)
            tmp, tmpa = [], []
            for i, img in enumerate(data):
                tmp.append(self.transform(img).to(device).unsqueeze(0))
                if i in action_index:
                    tmpa.append(self.topil(img))
            action_tensor = self.trans_action(tmpa)
            TC, H, W = action_tensor.shape
            action_tensor = action_tensor.reshape(1, TC // 3, 3, H, W).permute(0, 2, 1, 3, 4).to(device)
            with torch.no_grad():
                prediction = self.intern_action(action_tensor)
                prediction = F.softmax(prediction, dim=1).flatten()
                confidence = prediction[prediction.argmax()].item()
                prediction = kinetics_classnames[str(int(prediction.argmax()))]
                predictions.append(prediction)

        action_caption = ' '.join([f"Second {i + 1} : {j}.\n" for i, j in zip(time_index, predictions)])

        del action_tensor, tmpa, data_seg, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        data = loadvideo_decord_origin(video_path)
        tmp = []
        for i, img in enumerate(data):
            tmp.append(self.transform(img).to(device).unsqueeze(0))
        # dense caption
        dense_caption = []
        dense_foot = 1
        dense_index = np.arange(0, len(data) - 1, dense_foot)
        original_images = data[dense_index, :, :, ::-1]
        with torch.no_grad():
            for index, original_image in zip(dense_index, original_images):
                dense_caption.append(
                    self.dense_caption_model.run_caption_tensor(original_image, video_path, index, args['images_path']))
            dense_caption = ' '.join([f"Second {i + 1} : {j}.\n" for i, j in zip(dense_index, dense_caption)])

        # ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
        # text_result = list()
        # for i in range(data.shape[0]):
        #     result = ocr.ocr(data[i], cls=True)
        #     text_per_image = list()
        #     for idx in range(len(result)):
        #         res = result[idx]
        #         for line in res:
        #             text = dict()
        #             text['pos'] = line[0]
        #             text['text'] = line[1][0]
        #             text_per_image.append(text)
        #     text_result.append(text_per_image)
        ocr_output = ''
        # for index, text_per_image in enumerate(text_result):
        #     ocr_output += f'Second {index}: '
        #     for text in text_per_image:
        #         ocr_output += "'{}' ".format(text['text'])
        #     ocr_output += '\n'

        del data, original_images
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Video Caption
        image = torch.cat(tmp).to(device)

        self.model.threshold = 0.68
        if input_tag == '' or input_tag == 'none' or input_tag == 'None':
            input_tag_list = None
        else:
            input_tag_list = [input_tag.replace(',', ' | ')]
        with torch.no_grad():
            caption, tag_predict = self.model.generate_sublists(image, tag_input=input_tag_list, max_length=50,
                                                                return_tag_predict=True)
            frame_caption = ' '.join([f"Second {i + 1}:{j}.\n" for i, j in enumerate(caption)])
            if input_tag_list is None:
                tag_1 = set(tag_predict)
                tag_2 = ['none']
            else:
                _, tag_1 = self.model.generate(image, tag_input=None, max_length=50, return_tag_predict=True)
                tag_2 = set(tag_predict)
            synth_caption = self.model_T5.predict('. '.join(caption))

        del image, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        features = dict()
        features['subtitle'] = subtitle
        features['dense'] = dense_caption
        features['frame'] = frame_caption
        # return ' | '.join(tag_1), ' | '.join(tag_2), frame_caption, dense_caption, synth_caption[
        #     0], action_caption, subtitle, ocr_output
        return features

    def inference_second(self, video_path, input_tag, orderId=None):
        # Whisper
        subtitle = list()
        # try:
        #     whisper_result = self.whisper_model.transcribe(video_path)
        #     for segment in whisper_result['segments']:
        #         subtitle.append({
        #             'begin': int(segment['start']),
        #             'end': int(segment['end']),
        #             'text': segment['text'],
        #         })
        # except Exception as e:
        #     print(e)

        # 讯飞API
        try:
            api = RequestApi(appid="d1c4610a",
                             secret_key="baf471c32e59924820be4c5c18027247",
                             upload_file_path=video_path)

            _, orderId = api.get_result(orderId)
            subtitle = api.result2text(return_list=True)
        except Exception as e:
            print(e)

        data = loadvideo_decord_origin(video_path)
        tmp = []
        for i, img in enumerate(data):
            tmp.append(self.transform(img).to(device).unsqueeze(0))
        # dense caption
        dense_caption = list()
        dense_foot = 1
        dense_index = np.arange(0, len(data), dense_foot)
        original_images = data[dense_index, :, :, ::-1]
        with torch.no_grad():
            for index, original_image in zip(dense_index, original_images):
                dense_caption.append({
                    'begin': index,
                    'text': self.dense_caption_model.run_caption_tensor(original_image, video_path, index,
                                                                        args['images_path']),
                })
        del data, original_images
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Video Caption
        image = torch.cat(tmp).to(device)

        self.model.threshold = 0.68
        if input_tag == '' or input_tag == 'none' or input_tag == 'None':
            input_tag_list = None
        else:
            input_tag_list = [input_tag.replace(',', ' | ')]
        with torch.no_grad():
            caption, tag_predict = self.model.generate_sublists(image, tag_input=input_tag_list, max_length=50,
                                                                return_tag_predict=True)
            frame_caption = list()
            for i, j in enumerate(caption):
                frame_caption.append({
                    'begin': i,
                    'text': j,
                })

        del image, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        features = dict()
        features['subtitle'] = subtitle
        features['dense'] = dense_caption
        features['frame'] = frame_caption

        return features, orderId

    def inference_baseline(self, video_path, input_tag):
        subtitle = ''
        try:
            whisper_result = self.whisper_model.transcribe(video_path)
            for segment in whisper_result['segments']:
                subtitle += str(int(segment['start'])) + 's-' + str(int(segment['end'])) + 's: ' + segment[
                    'text'] + '\n'
        except Exception as e:
            subtitle = 'No subtitle'
            print(e)
        data = loadvideo_decord_origin(video_path)

        # InternVideo
        action_index = np.linspace(0, len(data) - 1, 8).astype(int)
        tmp, tmpa = [], []
        for i, img in enumerate(data):
            tmp.append(self.transform(img).to(device).unsqueeze(0))
            if i in action_index:
                tmpa.append(self.topil(img))
        action_tensor = self.trans_action(tmpa)
        TC, H, W = action_tensor.shape
        action_tensor = action_tensor.reshape(1, TC // 3, 3, H, W).permute(0, 2, 1, 3, 4).to(device)
        with torch.no_grad():
            prediction = self.intern_action(action_tensor)
            prediction = F.softmax(prediction, dim=1).flatten()
            confidence = prediction[prediction.argmax()].item()
            prediction = kinetics_classnames[str(int(prediction.argmax()))]
            print(f"Confidence: {confidence}")

        del action_tensor, tmpa
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # dense caption
        dense_caption = []
        dense_index = np.arange(0, len(data) - 1, 5)
        original_images = data[dense_index, :, :, ::-1]
        with torch.no_grad():
            for original_image in original_images:
                dense_caption.append(self.dense_caption_model.run_caption_tensor(original_image))
            dense_caption = ' '.join([f"Second {i + 1} : {j}.\n" for i, j in zip(dense_index, dense_caption)])

        del data, original_images
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Video Caption
        image = torch.cat(tmp).to(device)

        self.model.threshold = 0.68
        if input_tag == '' or input_tag == 'none' or input_tag == 'None':
            input_tag_list = None
        else:
            input_tag_list = [input_tag.replace(',', ' | ')]
        with torch.no_grad():
            caption, tag_predict = self.model.generate_sublists(image, tag_input=input_tag_list, max_length=50,
                                                                return_tag_predict=True)
            frame_caption = ' '.join([f"Second {i + 1}:{j}.\n" for i, j in enumerate(caption)])
            if input_tag_list is None:
                tag_1 = set(tag_predict)
                tag_2 = ['none']
            else:
                _, tag_1 = self.model.generate(image, tag_input=None, max_length=50, return_tag_predict=True)
                tag_2 = set(tag_predict)
            synth_caption = self.model_T5.predict('. '.join(caption))

        del image, tmp
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return ' | '.join(tag_1), ' | '.join(tag_2), frame_caption, dense_caption, synth_caption[
            0], prediction, subtitle


class InputVideo:
    def __init__(self, videos_path):
        self.output_path = None
        self.cur_time = None
        self.evaluate_path = ''
        self.videos_path = videos_path
        self.video_chat = VideoChat()
        self.videos = list()
        self.questions = list()
        self.short_result_list = list()
        self.features = list()
        self.exist_features = True
        print('Test video numbers:', len(os.listdir(self.videos_path)))
        for item in os.listdir(self.videos_path):
            if int(item) == 12:
                continue
            json_path = os.path.join(self.videos_path, item, 'data.json')
            video_path = os.path.join(self.videos_path, item, 'video.mp4')
            features_path = os.path.join(self.videos_path, item, 'features.json')
            if not os.path.exists(features_path):
                self.exist_features = False
            self.features.append(features_path)
            video_processor = VideoProcessor()
            self.short_result_list.append(video_processor.shot(video_path))
            self.questions.append(json_path)
            self.videos.append(video_path)

    def start_test(self):
        self.exist_features = False
        if not self.exist_features:
            self.video_chat.load_model()
        self.cur_time = datetime.now()
        self.output_path = os.path.join(args['output_path'], self.cur_time.strftime("%Y%m%d%H%M%S"))
        for index, video_path in enumerate(self.videos):
            if args['mode'] == 'normal':
                if os.path.exists(self.features[index]):
                    with open(self.features[index], 'r') as file:
                        data = json.load(file)
                        features = data['features']
                else:
                    with open(self.questions[index], 'r+') as file:
                        data = json.load(file)
                        orderId = data.get('orderId')
                        try:
                            features, new_orderId = self.video_chat.inference_second(video_path, '', orderId)
                            if not orderId:
                                data['orderId'] = new_orderId  # Save the returned orderId to the JSON data
                                file.seek(0)  # Move the file pointer to the beginning of the file
                                json.dump(data, file, indent=4,
                                          ensure_ascii=False)  # Write the updated data back to the file
                                file.truncate()  # Remove any remaining data after the end of the updated content
                        except Exception as e:
                            continue
                prompt = self.video_chat.bot.init_agent(args['openai_api_key'], features)
                self.qa_test(index, features, video_path)
            elif args['mode'] == 'baseline':
                try:
                    model_tag_output, user_tag_output, image_caption_output, dense_caption_output, video_caption_output, action_output, subtitle = self.video_chat.inference_baseline(
                        video_path, '')
                except Exception as e:
                    continue
                prompt = self.video_chat.bot.init_agent_baseline(args['openai_api_key'], image_caption_output,
                                                                 dense_caption_output,
                                                                 video_caption_output,
                                                                 model_tag_output, action_output, subtitle)
                self.qa_test(index, prompt, video_path)
            elif args['mode'] == 'shot':
                if os.path.exists(self.features[index]):
                    with open(self.features[index], 'r') as file:
                        data = json.load(file)
                        features = data['features']
                else:
                    with open(self.questions[index], 'r+') as file:
                        data = json.load(file)
                        orderId = data.get('orderId')
                        try:
                            features, new_orderId = self.video_chat.inference_second(video_path, '', orderId)
                            if not orderId:
                                data['orderId'] = new_orderId  # Save the returned orderId to the JSON data
                                file.seek(0)  # Move the file pointer to the beginning of the file
                                json.dump(data, file, indent=4,
                                          ensure_ascii=False)  # Write the updated data back to the file
                                file.truncate()  # Remove any remaining data after the end of the updated content
                        except Exception as e:
                            continue
                begin_time = 0
                summary_list = list()
                for shot_time in self.short_result_list[index]:
                    end_time = shot_time
                    if end_time == 0:
                        continue
                    shot_features = dict()
                    for feature_type, feature_content in features.items():
                        shot_features[feature_type] = list()
                        for item in feature_content:
                            if begin_time <= item['begin'] < end_time:
                                shot_features[feature_type].append(item)
                    if len(shot_features['frame']) == 0:
                        continue
                    dense_with_ocr = find_text_in_dense(shot_features)
                    shot_features['dense_with_ocr'] = dense_with_ocr
                    prompt, question = self.video_chat.bot.init_agent_with_features(args['openai_api_key'],
                                                                                    shot_features)
                    summary = self.video_chat.bot.run_text(question, args['llm'])
                    summary_list.append({
                        'begin': begin_time,
                        'end': end_time,
                        'text': summary,
                    })
                    begin_time = shot_time
                subtitle_output = ''
                for item in features['subtitle']:
                    subtitle_output += 'Second{}: Speaker{}: {}\n'.format(item['begin'], item['speaker'], item['text'])
                dense_with_ocr = find_text_in_dense(features)
                ocr_output = ''
                for item in dense_with_ocr:
                    ocr_output += 'Second{}: {}\n'.format(item['begin'], item['text'])

                output = {
                    'summary': summary_list,
                    'subtitle': subtitle_output,
                    # 'ocr': ocr_output,
                }
                if self.output_path:
                    folder = os.path.basename(os.path.dirname(video_path))
                    folder_path = os.path.join(self.output_path, folder)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    save_path = os.path.join(folder_path, 'summary.json')
                    with open(save_path, "w") as json_file:
                        json.dump(output, json_file, indent=4, ensure_ascii=False)
                prompt = self.video_chat.bot.init_agent_with_summary(args['openai_api_key'], output)
                self.qa_test(index, prompt, video_path)

    def qa_test(self, index, features, video_path):
        output = dict()
        qa_output = list()

        with open(self.questions[index]) as file:
            data = json.load(file)
            output['video_name'] = data['video_name']
            output['test_time'] = self.cur_time.strftime("%Y-%m-%d %H:%M:%S")
            output['remarks'] = args['remarks']
            if args['qa']:
                for qa in data['qa']:
                    if args['qa_mode'] == 'think':
                        question = '请你给出你的思考过程：猜测' + qa['q']
                    else:
                        question = qa['q']
                    answer = qa['a']
                    infer_answer = self.video_chat.bot.run_text(question, args['llm'])
                    if args['qa_mode'] == 'think':
                        question = {
                            'q': qa['q'],
                            'predict': infer_answer
                        }
                        change_prompt = \
                            '''
                                你需要对指定的回答进行总结。
                                用户将输入以下格式的内容：
                                {
                                    "q": "华强用刀做了哪些事情？",
                                    "predict": "根据描述和字幕中的信息，我们得知视频中出现了一个男人使用刀的场景。以下是我根据描述和字幕提供的信息的猜测：\n\n1. 在Second74 to Second76这段视频中，一个男人在一个市场的水果摊前切割了一个西瓜。可以推测，这个男人使用刀来切开了西瓜。\n\n所以，根据视频的描述，我们可以推测华强使用刀来切割了一个西瓜。"
                                },
                                其中q是问题，predict是一个包含了思考过程的回答。你需要对这个回答进行总结，去掉其中的思考过程，保留主要的答案。
                                对于上面这个例子，你的总结应该是：“华强使用刀来切割了一个西瓜。”
                                
                                下面是用户的输入：
                            '''
                        infer_answer = self.video_chat.bot.run_text(str(question), args['llm'], change_prompt)
                    if not infer_answer:
                        print('token过长，无法回答')
                        return None
                    qa_output.append({'q': qa['q'], 'a': answer, 'predict': infer_answer})
                output['qa'] = qa_output
            output['features'] = features
        if self.output_path:
            folder = os.path.basename(os.path.dirname(video_path))
            folder_path = os.path.join(self.output_path, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            save_path = os.path.join(folder_path, 'output.json')
            with open(save_path, "w") as json_file:
                json.dump(output, json_file, indent=4, ensure_ascii=False, cls=CustomEncoder)

    def evaluate_by_chatgpt(self):
        if args['predict'] and args['evaluate']:
            all_predict_results = os.listdir(args['output_path'])
            int_list = [int(s) for s in all_predict_results]
            self.evaluate_path = os.path.join(args['output_path'], str(max(int_list)))
        elif not args['predict'] and args['evaluate']:
            self.evaluate_path = args['evaluate_path']

        evaluate_result = dict()
        total_score = 0
        qa_number = 0
        evaluate_save_path = os.path.join(self.evaluate_path, 'evaluate_result_new.json')
        answer_list = []
        for output_folder in os.listdir(self.evaluate_path):
            # if int(output_folder) != 14:
            #     continue
            if output_folder.endswith(".json"):
                continue
            output_file = os.path.join(self.evaluate_path, output_folder, 'output.json')
            if not os.path.exists(output_file):
                continue
            with open(output_file) as file:
                data = json.load(file)
                cur_score = 0
                cur_number = 0
                cur_result = dict()
                cur_answer = list()
                for qa in data['qa']:
                    answer = self.video_chat.bot.evaluate_qa(args['openai_api_key'], str(qa), args['llm'])
                    total_score += int(answer['score'])
                    qa_number += 1
                    cur_score += int(answer['score'])
                    cur_number += 1
                    cur_answer.append(answer)
                cur_result['video'] = output_folder
                cur_result['mean_score'] = cur_score / cur_number
                cur_result['answer'] = cur_answer
                answer_list.append(cur_result)
        evaluate_result['mean_score'] = total_score / qa_number
        evaluate_result['remarks'] = args['remarks']
        evaluate_result['answer_list'] = answer_list
        with open(evaluate_save_path, "w") as json_file:
            json.dump(evaluate_result, json_file, indent=4, ensure_ascii=False)


def find_text_in_dense(features):
    ocr = features['ocr']
    dense = features['dense']
    dense_with_ocr = list()
    start_time = dense[0]['begin']
    for cur_ocr in ocr:
        time = cur_ocr['begin']
        cur_dense_text = dense[time - start_time]['text']
        cur_dense_with_ocr = defaultdict(str)
        for cur_text in cur_ocr['text']:
            cur_text_pos = cur_text['pos']
            cur_text_text = cur_text['text']
            if cur_text_pos[1] > 900:
                print(f'{cur_text_text} 可能是字幕')
                continue
            if len(cur_dense_text) == 0:
                continue
            belong_obj = find_pos(cur_text_pos, cur_dense_text)
            if belong_obj:
                cur_dense_with_ocr[belong_obj] += f'{cur_text_text} '
            # if belong_obj:
            #     cur_dense_with_ocr += f'{belong_obj} with the words "{cur_text_text}" on it, '
        if len(cur_dense_with_ocr) > 0:
            final_text = ''
            for key, value in cur_dense_with_ocr.items():
                final_text += f'{key} with the words "{value.strip()}" on it, '
            dense_with_ocr.append({
                'begin': time,
                'text': final_text.strip(', '),
            })
    return dense_with_ocr


def find_pos(text_pos, dense_text):
    obj_list = dense_text.split(';')
    min_square = math.inf
    belong_obj = None
    for obj in obj_list:
        obj_item = obj.split(': ')[0]
        if obj_item == ' ':
            continue
        obj_pos = ast.literal_eval(obj.split(': ')[1])
        if obj_pos[0] <= text_pos[0] and obj_pos[1] <= text_pos[1] and obj_pos[2] >= text_pos[2] and obj_pos[3] >= \
                text_pos[3]:
            if compute_square(obj_pos) < min_square:
                belong_obj = obj_item
                min_square = compute_square(obj_pos)
    return belong_obj


def compute_square(pos):
    return (pos[2] - pos[0]) * (pos[3] - pos[1])


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.int64):
            return int(o)  # Convert int64 to Python int
        return super().default(o)


class VideoProcessor:
    def shot(self, video_path):
        shot_result_path = os.path.join(os.path.dirname(video_path), 'video.mp4.scenes.txt')
        time_intervals = [0]

        frame_rate = self.get_frame_rate(video_path)

        with open(shot_result_path, 'r') as f:
            for line in f:
                start_frame, end_frame = map(int, line.strip().split())
                start_time = self.frames_to_seconds(start_frame, frame_rate)
                end_time = self.frames_to_seconds(end_frame, frame_rate)
                time_intervals.append(int(end_time))

        return time_intervals

    def frames_to_seconds(self, frame_number, frame_rate):
        return frame_number / frame_rate

    def get_frame_rate(self, video_path):
        info = ffmpeg.probe(video_path)
        vs = next(c for c in info['streams'] if c['codec_type'] == 'video')
        fps = float(Fraction(vs['r_frame_rate']))
        return fps


def main():
    input_video = InputVideo(args['videos_path'])
    if args['predict']:
        input_video.start_test()
    if args['evaluate']:
        input_video.evaluate_by_chatgpt()


if __name__ == "__main__":
    main()
