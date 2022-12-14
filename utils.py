import tokenizers
from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

print("utils 시작")
start = time.time()
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'digit82/kobart-summarization')
sum_model = BartForConditionalGeneration.from_pretrained(
    'digit82/kobart-summarization')
chat_model = SentenceTransformer(
    'xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

train_data = pd.read_csv('ChatBotData.csv')
# 병목지점..(5-6분 소요)
train_data['embedding'] = train_data.apply(
    lambda row: chat_model.encode(row.Q), axis=1)

end = time.time()
print("utils 끝")
print(f"{end - start:.5f} sec")
