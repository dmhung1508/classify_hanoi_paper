#pretrain model
roberta_config='longformer/config.json'
roberta='longformer/pytorch_model.bin'
model_pretrained='longformer'
model_saved='longformer_run'
bpe_codes = 'longformer/bpe.codes'
vocab = "longformer/vocab.txt"
data_path="hanoi.csv"
save_path = "models"

#config
max_length=1024
num_class = 2
gpu_id=0
device = "cuda"
batch_size= 5
epochs=5
kfold=5

#run model
model_config = 'longformer/config.json'
model_bpe = 'longformer/bpe.codes'
model = 'models'