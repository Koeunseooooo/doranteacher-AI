from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
from rq.notebooks.notebook_utils import TextEncoder, load_model, get_generated_images_by_texts
import numpy as np
import itertools
import requests
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import transformers
import yaml
import torch
import torchvision
import clip
import torch.nn.functional as F
import time

import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# tokenizer = PreTrainedTokenizerFast.from_pretrained(
#     'digit82/kobart-summarization')
# sum_model = BartForConditionalGeneration.from_pretrained(
#     'digit82/kobart-summarization')
chat_model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def commentAPI(text):
    start = time.time()

    global sum_model
    global chat_model

    global tokenizer

    text = text.replace('\n', ' ')

    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + \
        raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = sum_model.generate(torch.tensor(
        [input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
    input = tokenizer.decode(
        summary_ids.squeeze().tolist(), skip_special_tokens=True)
    print(input)
    input = "오늘은 친구들과 축구를 했다. 날씨도 좋고 축구도 이겼다. 오늘은 정말 행복한 날이다. 칭찬해줘."
    # train_data = pd.read_csv('ChatBotData.csv')

    result = return_comment(input, train_data)
    print(result)
    end = time.time()
    print(f"{end - start:.5f} sec")

#


def embadding(train_data):
    train_data['embedding'] = train_data.apply(
        lambda row: chat_model.encode(row.Q), axis=1)


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def return_comment(content, train_data):
    embedding = chat_model.encode(content)
    train_data['score'] = train_data.apply(
        lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']


if __name__ == '__main__':
    # text = "점심으로 짜장밥을 먹고나서 바로 피아노학원을 갔다. 그런데 피아노학원에서부터 자꾸 배가 아팠다.짜방밥 때문인가?하는 생각도 들었다.집에 돌아와서도 눈물이 날 정도로 배가 아팠다.그래서 바로 병원으로 갔다.의사선생님께 배가 아프다고 하자 배를 눌러보시고 꼭 맹장같다고 하셨다.의사선생님은 외과로 가보라고 하셨다.외과에 가자 힘도 없고 배는 계속 아팠다. 너무 슬프다."
    # comment(text)
    # train_data = pd.read_csv('ChatBotData.csv')
    # print(train_data['embedding'].head(10))
    # input = "오늘은 친구들과 축구를 했다. 날씨도 좋고 축구도 이겼다. 오늘은 정말 행복한 날이다. 칭찬해줘."
    # a = return_comment(input, train_data)
    # print(a)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
    train_data = pd.read_csv('ChatBotData.csv')
    train_data.head()

    # train_data['embedding'] = train_data.apply(
    #     lambda row: chat_model.encode(row.Q), axis=1)
    # print(train_data.head(10))
    # print("fin")
    # train_data.to_csv("chatBotData.csv", mode='w')

    # print(train_data['label'].head(10))
    # data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    # dataframe = pd.DataFrame(data)
    # dataframe.to_csv("hello.csv")
