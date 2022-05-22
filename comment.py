from rq.notebooks.notebook_utils import TextEncoder, load_model, get_generated_images_by_texts
import numpy as np
import itertools
import requests
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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


def comment(doc):
    pass
