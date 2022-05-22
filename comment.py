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
from numpy import char, dot, ndarray
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
import io

import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_models():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        'digit82/kobart-summarization')
    sum_model = BartForConditionalGeneration.from_pretrained(
        'digit82/kobart-summarization')
    chat_model = SentenceTransformer(
        'xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    return tokenizer, sum_model, chat_model


def comment(text):
    tokenizer, sum_model, chat_model = load_models()
    start = time.time()

    text = text.replace('\n', ' ')

    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + \
        raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = sum_model.generate(torch.tensor(
        [input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
    text = tokenizer.decode(
        summary_ids.squeeze().tolist(), skip_special_tokens=True)
    print(text)
    train_data = pd.read_csv('ChatBotData.csv')

    # 병목지점..(5-6분 소요)
    embadding(train_data, chat_model)
    result = return_comment(text, train_data, chat_model)
    print(result)
    end = time.time()
    print(f"{end - start:.5f} sec")
    return result


def embadding(train_data, chat_model):
    train_data['embedding'] = train_data.apply(
        lambda row: chat_model.encode(row.Q), axis=1)
    return train_data['embedding']


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def return_comment(text, train_data, chat_model):
    embedding = chat_model.encode(text)
    train_data['score'] = train_data.apply(
        lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']


# if __name__ == '__main__':
#     text = "오늘은 도서관에 가서 책을 읽었다. 졸렸지만 책 한 권을 다 읽어서 뿌듯했다."
#     commentAPI(text)
