import numpy as np
import torch,json
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from utils import *
from typing import List
from arguments import load_args
from run import *
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)
from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from torch.utils.data import (DataLoader, TensorDataset, Dataset)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
import torch.nn.functional as F
import config

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
    default=config.model_bpe,
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file(config.vocab)
seed_everything(86)
#---------
app = FastAPI()
args = load_args()
NUM_LABEL = config.num_class
device = torch.device(0)

# Load Pretrained model vừa lưu ở trên
config2 = None
BERT_SA2 = None

def predict(text):
  text = bpe.encode(tokenlizes(text))
  encode_ = vocab.encode_line('<s> ' + text + ' </s>',append_eos=True, add_if_not_exist=False).long().tolist()
  encode_text = pad_sequences([encode_], maxlen=config.max_length, dtype="long", value=0, truncating="post", padding="post")

  predict_masks = get_mask(encode_text)
  predict_masks = torch.tensor(predict_masks,dtype = torch.int64)
  predict_inputs = torch.tensor(encode_text)

  predict_inputs = predict_inputs.to(device)
  predict_masks = predict_masks.to(device)

  with torch.no_grad():
    outputs = BERT_SA2(predict_inputs, token_type_ids=None, attention_mask=predict_masks)
    logits = outputs[0]

    probabilities = F.softmax(logits, dim=1)
    print(probabilities)
    # Lấy tỉ lệ đúng (xác suất của lớp có giá trị lớn nhất)
    max_probability = torch.max(probabilities, dim=1)[0].item()
    tile = f"{max_probability*100:.2f}"
    print(f"Tỉ lệ đúng: {tile}")
    logits = logits.detach().cpu().numpy()

    predict = np.argmax(logits)

    if predict == 0:
      cls = "NO"
    elif predict == 1:
      cls = "YES"
    return cls
class text_sample(BaseModel):
    text: str


class batch(BaseModel):
    text: str

@app.on_event("startup")
def reload_model():
    global config2, BERT_SA2
    config2 = RobertaConfig.from_pretrained(config.model_config)
    BERT_SA2 = RobertaForSequenceClassification.from_pretrained(
        config.model,
        config=config2
    )
    BERT_SA2.cuda()
    return "succes"
@app.get("/")
def read_root():
    return "USE POST"
@app.post("/train/")
async def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model)
    return {"message": "Training started"}
def train_model():
   file_path = os.path.join('weight', 'weight.json')
   
   print("trainning")
   train_models()
@app.get("/reload/")
def reload_model():
    global config2, BERT_SA2
    config2 = RobertaConfig.from_pretrained(config.model_config)
    BERT_SA2 = RobertaForSequenceClassification.from_pretrained(
        config.model,
        config=config2
    )
    BERT_SA2.cuda()
    return "succes"
@app.post("/hanoi")
async def predict_batch(item: batch):
    item = item.text
    item = preprocess_text(item)
    list_text = clean(item)
    return predict(list_text)
@app.get("/status")
async def read_json_file():
    file_path = os.path.join('weight', 'weight.json')
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
@app.get("/loss_plot")
async def get_loss_plot():
    # Trả về ảnh từ đường dẫn tệp 'loss_plot.png'
    return FileResponse("weight/loss_plot.png")
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=4505, reload=True)

