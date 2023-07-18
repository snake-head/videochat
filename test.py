import os
import numpy as np
import random
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
from chatbot import *
from load_internvideo import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from simplet5 import SimpleT5
from models.grit_model import DenseCaptioning
import whisper
from datetime import datetime
import configparser

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
    'evaluate': config.get('Arguments', 'evaluate') == 'True',
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

        del action_tensor, tmpa, data_seg
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        data = loadvideo_decord_origin(video_path)
        # dense caption
        dense_caption = []
        dense_foot = 5
        dense_index = np.arange(0, len(data) - 1, dense_foot)
        original_images = data[dense_index, :, :, ::-1]
        with torch.no_grad():
            for index, original_image in zip(dense_index, original_images):
                dense_caption.append(
                    self.dense_caption_model.run_caption_tensor(original_image, video_path, index, args['images_path']))
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
            0], action_caption, subtitle


class InputVideo:
    def __init__(self, videos_path):
        self.evaluate_path = ''
        self.videos_path = videos_path
        self.video_chat = VideoChat()
        self.videos = list()
        self.questions = list()
        print('Test video numbers:', len(os.listdir(self.videos_path)))
        for item in os.listdir(self.videos_path):
            # if int(item) < 12:
            #     continue
            json_path = os.path.join(self.videos_path, item, 'data.json')
            video_path = os.path.join(self.videos_path, item, 'video.mp4')
            self.questions.append(json_path)
            self.videos.append(video_path)

    def start_test(self):
        self.video_chat.load_model()
        cur_time = datetime.now()
        output_path = os.path.join(args['output_path'], cur_time.strftime("%Y%m%d%H%M%S"))
        for index, video_path in enumerate(self.videos):
            try:
                model_tag_output, user_tag_output, image_caption_output, dense_caption_output, video_caption_output, action_output, subtitle = self.video_chat.inference(
                    video_path, '')
            except Exception as e:
                continue
            self.video_chat.bot.init_agent(args['openai_api_key'], image_caption_output, dense_caption_output,
                                           video_caption_output,
                                           model_tag_output, action_output, subtitle)
            output = dict()
            qa_output = list()

            with open(self.questions[index]) as file:
                data = json.load(file)
                output['video_name'] = data['video_name']
                output['test_time'] = cur_time.strftime("%Y-%m-%d %H:%M:%S")
                output['remarks'] = args['remarks']
                for qa in data['qa']:
                    question = qa['q']
                    answer = qa['a']
                    infer_answer = self.video_chat.bot.run_text(question, args['llm'])
                    qa_output.append({'q': question, 'a': answer, 'predict': infer_answer})
                output['qa'] = qa_output
            if output_path:
                folder = os.path.basename(os.path.dirname(video_path))
                folder_path = os.path.join(output_path, folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                save_path = os.path.join(folder_path, 'output.json')
                with open(save_path, "w") as json_file:
                    json.dump(output, json_file, indent=4, ensure_ascii=False)

    def evaluate_by_chatgpt(self):
        if args['predict']:
            all_predict_results = os.listdir(args['output_path'])
            int_list = [int(s) for s in all_predict_results]
            self.evaluate_path = max(int_list)
        if args['evaluate']:
            self.evaluate_path = args['evaluate_path']

        evaluate_result = dict()
        total_score = 0
        qa_number = 0
        evaluate_save_path = os.path.join(self.evaluate_path, 'evaluate_result.json')
        answer_list = []
        for output_folder in os.listdir(self.evaluate_path):
            if output_folder.endswith(".json"):
                continue
            output_file = os.path.join(self.evaluate_path, output_folder, 'output.json')
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
        evaluate_result['answer_list'] = answer_list
        with open(evaluate_save_path, "w") as json_file:
            json.dump(evaluate_result, json_file, indent=4, ensure_ascii=False)


def main():
    input_video = InputVideo(args['videos_path'])
    if args['predict']:
        input_video.start_test()
    if args['evaluate']:
        input_video.evaluate_by_chatgpt()


if __name__ == "__main__":
    main()
