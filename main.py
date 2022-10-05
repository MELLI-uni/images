import logging
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)

import matplotlib
import matplotlib.pyplot as plt
import statistics

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaTokenizer, RobertaModel

device = 'cuda' if cuda.is_available() else 'cpu'
# logging.set_verbosity_warning()
# logging.set_verbosity_error()

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

LEARNING_RATE = 1e-05

data = load_dataset("ag_news")

# ag_news_data contains {'train': (120000, 2), 'test': (7600, 2)}
training_data = data['train']
testing_data = data['test']

class Data(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_len,
            return_token_type_ids = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class RobertaClass(torch.nn.Module):
    def __init__(self, model_name):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)

        return output

def data_loading(df_train, df_test, tokenizer):
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    training_set = Data(df_train, tokenizer, MAX_LEN)
    testing_set = Data(df_test, tokenizer, MAX_LEN)

    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': TEST_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    return training_loader, testing_loader

def train(model, training_loader, epoch, loss_function, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()

    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim = 1)
        n_correct += (big_idx==targets).sum().item()
        
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _%5000==0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples

            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct *100)/nb_tr_examples

    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

def valid(model, testing_loader, loss_function):
    model.eval()

    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    actual = []
    predicted = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += (big_idx==targets).sum().item()

            predicted.extend(big_idx.tolist())
            actual.extend(targets.tolist())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct * 100)/nb_tr_examples

                print(f"Validation Loss per 100 Steps: {loss_step}")
                print(f"Validation Accuracy per 100 Steps: {accu_step}")

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples

    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    f1_micro = f1_score(actual, predicted, average='micro', zero_division = 0)

    return f1_micro
            
df_train = pd.DataFrame(data=training_data)
df_test = pd.DataFrame(data=testing_data)
df_data = pd.concat([df_train, df_test], ignore_index=True)

df_data.to_csv('sample.csv')

kf = KFold(n_splits=5, random_state=99, shuffle=True)
num_split = kf.get_n_splits(df_data)

f1_micros = []

for train_index, test_index in kf.split(df_data):
    df_train = df_data.iloc[train_index]
    df_test = df_data.iloc[test_index]

    model = RobertaClass('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True, max_length=MAX_LEN)

    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    training_loader, testing_loader = data_loading(df_train, df_test, tokenizer)

    EPOCHS = 1
    
    for epoch in range(EPOCHS):
        train(model, training_loader, epoch, loss_function, optimizer)

    f1_micro = valid(model, testing_loader, loss_function)
    f1_micros.append(f1_micro)

avg = "{:.2f}".format((statistics.mean(f1_micros)) * 100)
std = "{:.2f}".format((statistics.stdev(f1_micros)) * 100)

item_text = avg + "+-" + std

print(item_text)

# Instantiate model
#model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")