'''
Description: 
Version: 1.0
Autor: ZhuYichen
Date: 2023-07-03 10:56:48
LastEditors: ZhuYichen
LastEditTime: 2023-07-09 19:22:10
'''
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import gradio as gr
import openai
import ast


def cut_dialogue_history(history_memory, keep_last_n_words=400):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class ConversationBot:
    def __init__(self):
        self.system_prompt = None
        self.openai_api_key = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = []

    def run_text(self, question, llm):
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        # res = self.agent({"input": text.strip()})
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # res['output'] = res['output'].replace("\\", "/")
        # response = res['output'] 
        # print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        answer = response.choices[0].message.content
        print(f"\nQuestion: {question}\nAnswer: {answer}\n")
        return answer

    def init_agent(self, openai_api_key, image_caption, dense_caption, video_caption, tags, action, subtitle):
        self.openai_api_key = openai_api_key
        chat_history = ''
        system_prompt = \
            f"""
                You are a chatbot that conducts conversations based on video descriptions. You mainly answer based on the given description, and you can also answer the relevant knowledge of the person or object contained in the video.
                The description is given in the following format: the purpose of the description - the type of description - the content of the description.
                '''
                The action description provides the main actions that appear in the video for the next five seconds every five seconds.                '''
                Action description:
                {action}
                '''
                The frame description is a description for one second, so that you can convert it into time. When describing, please mainly refer to the frame description.
                Frame description:
                {image_caption}
                '''
                Dense caption is to give content every five seconds, you can disambiguate them in timing.
                Dense caption:
                {dense_caption}
                '''
                Subtitles provide the content of the person's speech during a certain time period in the video. You can infer what happened in the video based on it.
                Subtitle:
                {subtitle}
            """
        self.memory.clear()
        if not openai_api_key.startswith('sk-'):
            print('OPEN_API_KEY ERROR')
        self.system_prompt = system_prompt
        print(self.system_prompt)
        # openai.api_base = 'https://closeai.deno.dev/v1'
        # self.agent = initialize_agent(
        #     self.tools,
        #     self.llm,
        #     agent="conversational-react-description",
        #     verbose=True,
        #     memory=self.memory,
        #     return_intermediate_steps=True,
        #     agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, )
        return openai_api_key

    def evaluate_qa(self, openai_api_key, qa, llm):
        openai.api_key = openai_api_key
        system_prompt = \
            """
                评估以下视频问答模型的预测准确性，按照0-5的评分标准：

                ‘’’
                
                0：预测答案完全无关或没有回答问题。
                
                1：预测答案部分相关，但主要的信息都没有涵盖。
                
                2：预测答案中包含了部分正确信息，但并没有完全回答问题。
                
                3：预测答案回答了问题的大部分内容，但缺少一些关键信息。
                
                4：预测答案几乎完全正确，只是在细节上有一些小的遗漏。
                
                5：预测答案完全准确，全面回答了问题。
                
                ‘’’
                你需要给出评分的理由，再进行评分。
                
                你的回答必须以字典格式写出：
                
                {'reason': 评分理由,'score': 分数,}
                
                问题、标准答案以及预测答案分别如下：
            """
        response = openai.ChatCompletion.create(
            model=llm,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": qa
                }
            ],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].message.content
        print(f"\nAnswer: {answer}\n")
        answer = answer.strip("{}").replace("'", "")
        dict_answer = dict(item.split(": ") for item in answer.split(", "))
        return dict_answer


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
