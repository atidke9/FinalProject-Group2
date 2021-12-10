from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, DataCollatorWithPadding, get_scheduler
#from transformers import DistilBertTokenizer, DataCollatorWithPadding, get_scheduler
#from transformers import ConvBertTokenizer,  DataCollatorWithPadding, get_scheduler
import torch
from torch.utils.data import DataLoader
from transformers import ElectraForSequenceClassification, AdamW
#from transformers import DistilBertForSequenceClassification, AdamW
#from transformers import ConvBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
import numpy as np

num_epochs = 3
#checkpoint = 'distilbert-base-uncased'
#checkpoint = 'YituTech/conv-bert-base'
checkpoint = 'google/electra-small-discriminator'
max_len = 200

train_df = pd.read_csv('train.csv')[:20000]
train_df.columns = ['label','text']
train_df['label'] = train_df['label'] - 1
train_texts = train_df['text'].values.tolist()
train_labels = train_df['label'].values.tolist()

test_df = pd.read_csv('test.csv')[:1000]
test_df.columns = ['label','text']
test_df['label'] = test_df['label'] - 1
test_texts = test_df['text'].values.tolist()
test_labels = test_df['label'].values.tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
tokenizer = ElectraTokenizer.from_pretrained(checkpoint)
#tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
#tokenizer = ConvBertTokenizer.from_pretrained(checkpoint)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_len)

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = YelpDataset(train_encodings, train_labels)
val_dataset = YelpDataset(val_encodings, val_labels)
test_dataset = YelpDataset(test_encodings, test_labels)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = ElectraForSequenceClassification.from_pretrained(checkpoint)
#model = DistilBertForSequenceClassification.from_pretrained(checkpoint)
#model = ElectraForSequenceClassification.from_pretrained(checkpoint)
model.to(device)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(val_dataset,
                             batch_size=8, collate_fn=data_collator)

optim = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optim,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
        lr_scheduler.step()
        progress_bar.update(1)


PRED = []
Y = []
def update(pred,y):
    PRED.append(pred.detach().cpu().numpy())
    Y.append(y.detach().cpu().numpy())


num_eval_steps = num_epochs * len(eval_dataloader)
progress_bar = tqdm(range(num_eval_steps))

model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids,labels=labels)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    update(predictions,labels)
    progress_bar.update(1)


Y = np.array(Y).reshape(-1,1)
PRED = np.array(PRED).reshape(-1,1)
res = accuracy_score(Y, PRED)
print("\nValidation accuracy:",res)