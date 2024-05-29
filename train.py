from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
import os
import pandas as pd
import numpy as np
import torch,json
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from utils import *
seed_everything(86)

class Trainer():
    def __init__(self, BERT_SA, config,  dataloader, df_train, device = "cuda"):
        self.model = BERT_SA
        self.config = config
        self.dataloader = dataloader
        self.device = device
        self.df_train = df_train

    def train(self, ):
        if not os.path.exists('weight'):
            os.makedirs('weight')
        label_counts = self.df_train['label'].value_counts()
        epochs = self.config.epochs
        best_eval_f1_score = 0
        device = 'cuda'
        best_eval_f1_score = 0
        jso = {}
        jso['Process'] = {}
        jso['Data'] = {}
        jso['Data'] = label_counts.to_dict()
        jso['Realtime'] = {}
        for fold in range(self.config.kfold):
            print(f'-----------Fold: {fold + 1} ------------------')
            jso['Realtime'][f'Fold {fold + 1}'] = {}
            train_dataloader, val_dataloader = self.dataloader.prepare_loaders(fold, self.df_train)
            BERT_SA = self.model
            param_optimizer = list(BERT_SA.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)

            for epoch_i in range(0, epochs):
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')
                total_loss = 0
                BERT_SA.train()
                train_accuracy = 0
                nb_train_steps = 0
                train_f1 = 0

                for step, batch in tqdm(enumerate(train_dataloader)):
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)

                    BERT_SA.zero_grad()
                    outputs = BERT_SA(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
                    loss = outputs[0]
                    total_loss += loss.item()

                    logits = outputs[1].detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
                    train_accuracy += tmp_train_accuracy
                    train_f1 += tmp_train_f1
                    nb_train_steps += 1

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
                    optimizer.step()

                avg_train_loss = total_loss / len(train_dataloader)
                print(" Accuracy: {0:.4f}".format(train_accuracy / nb_train_steps))
                print(" F1 score: {0:.4f}".format(train_f1 / nb_train_steps))
                print(" Average training loss: {0:.4f}".format(avg_train_loss))
                jso['Realtime'][f'Fold {fold + 1}'][f'{epoch_i + 1}/{epochs}'] = {}
                jso['Realtime'][f'Fold {fold + 1}'][f'{epoch_i + 1}/{epochs}']['Training'] = {}
                jso['Realtime'][f'Fold {fold + 1}'][f'{epoch_i + 1}/{epochs}']['Training'] = {
                    "Accuracy": round(train_accuracy / nb_train_steps, 4),
                    "F1 score": round(train_f1 / nb_train_steps, 4),
                    "Average training loss": round(avg_train_loss, 4),
                }
                print("Running Validation...")
                BERT_SA.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                eval_f1 = 0
                for batch in tqdm(val_dataloader):
                    batch = tuple(t.to(device) for t in batch)

                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        outputs = BERT_SA(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask)
                        logits = outputs[0]
                        logits = logits.detach().cpu().numpy()
                        label_ids = b_labels.to('cpu').numpy()

                        tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)

                        eval_accuracy += tmp_eval_accuracy
                        eval_f1 += tmp_eval_f1
                        nb_eval_steps += 1

                print(" Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
                print(" F1 score: {0:.4f}".format(eval_f1 / nb_eval_steps))
                jso['Realtime'][f'Fold {fold + 1}'][f'{epoch_i + 1}/{epochs}']['Validation'] = {}
                jso['Realtime'][f'Fold {fold + 1}'][f'{epoch_i + 1}/{epochs}']['Validation']= {
                    "Accuracy": round(eval_accuracy / nb_eval_steps, 4),
                    "F1 score": round(eval_f1 / nb_eval_steps, 4),
                }
                if fold == 0 and epoch_i == 0:
                    BERT_SA.save_pretrained(self.config.save_path)
                    print('Saved Pretrain!')
                    best_eval_f1_score = eval_f1 / nb_eval_steps
                    jso['Process'] = {
                        'Complete' :  round((fold + 1 / self.config.kfold), 2),
                        'Best_F1' : round(best_eval_f1_score, 4),
                        'Best_Accuracy' : round(eval_accuracy / nb_eval_steps, 4),
                        'Status' : 'Training',
                    }
                else:
                    if best_eval_f1_score < eval_f1 / nb_eval_steps:
                        BERT_SA.save_pretrained(self.config.save_path)
                        print('Update Saved Pretrain!')
                        best_eval_f1_score = eval_f1 / nb_eval_steps
                        jso['Process'] = {
                            'Complete' :  round((fold + 1 / self.config.kfold), 2),
                            'Best_F1' : round(best_eval_f1_score, 4),
                            'Best_Accuracy' : round(eval_accuracy / nb_eval_steps, 4),
                            'Status' : 'Training',
                        }
                with open('weight/weight.json', 'w') as f:
                        json.dump(jso, f, indent=4)

        print("Training complete!")
        jso['Process']['Status'] = 'Done'
        with open('weight/weight.json', 'w') as f:
            json.dump(jso, f, indent=4)
