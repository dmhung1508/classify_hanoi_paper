
import torch
import json
import re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import regex as re
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
PHONE = re.compile(r"[0-9]{10,11}")
URL = re.compile(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
DATETIME = re.compile(r"\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}")
PUNC = re.compile(r"[^\w\s,.]") #any chars other than \w \s , .

def preprocess_text(txt):
    #txt = txt.lower()
    txt = re.sub(EMAIL, ' ', txt)
    txt = re.sub(PHONE, ' ', txt)
    txt = re.sub(URL, ' ', txt)
    txt = re.sub(DATETIME, ' ', txt)
    txt = re.sub('&#?[a-z0-9]+;', ' ', txt)
    txt = re.sub(PUNC, ' ', txt)
    return txt
def get_mask(data):
	masks = []
	for sen in data:
		mask = [int(token>0) for token in sen]
		masks.append(mask)
	return masks
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    F1_score = f1_score(pred_flat, labels_flat, average='weighted')

    return accuracy_score(pred_flat, labels_flat), F1_score
with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = set(file.read().splitlines())

# Hàm để xóa stop words khỏi một đoạn văn bản
def tokenlizes(text):
    words = rdrsegmenter.tokenize(text)[0]

    sentence = ' '.join(words)
    return sentence
def clean(text):
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    a = ' '.join(filtered_words)
    a = tokenlizes(a)
    return a

#Xử lý các dòng bị bỏ trống ở mục 'content'
def preprocess(df):
  for i in df.index:
    if type(df['text'].at[i]) != str:  #Các dòng trống khi đọc ra thì nó trả về là NaN chứ không phải là chuỗi rỗng
      df = df.drop(i) #delete row
  df = df.reset_index(drop=True)  #update index
  return df
# Mục dích loại bỏ những dóng không dấu và tiếng Anh
def non_accent(sent):
  count = 0
  letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ',']
  for word in sent: # loop over all words
    if word in letters:
      count += 1
      if count == len(sent):
        return True
    else:
      break
  return False

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
