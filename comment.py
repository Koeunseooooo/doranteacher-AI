from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import numpy as np
import itertools
import requests
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import char, dot, ndarray
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import time
from utils import *

import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def comment(text):
    print("comment 시작")
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

    result = return_comment(text,  chat_model, train_data)
    end = time.time()
    print("comment 끝")
    print(f"{end - start:.5f} sec")
    return result


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def return_comment(text, chat_model, train_data):
    embedding = chat_model.encode(text)
    train_data['score'] = train_data.apply(
        lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']
