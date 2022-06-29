from datasets import Dataset, load_dataset, load_metric
from sklearn.metrics import f1_score
from transformers import AdamW, get_scheduler, AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from os import sys

def train(model,iterator,optimizer,train_pretrain=False):
  epoch_loss = 0.0
  epoch_acc = 0.0
  epoch_f1 = 0.0

  model.train()
  metric = load_metric("accuracy")
  metric2 = load_metric("f1")
  for batch in iterator:
      optimizer.zero_grad()

      if train_pretrain:
        b_input_ids = batch["input_ids"]
        b_input_mask = batch["attention_mask"]
        b_labels = batch["target"]
        outputs = model(b_input_ids,token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = outputs.loss
        predictions = outputs.logits
        predictions = torch.argmax(predictions, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["target"])
        metric2.add_batch(predictions=predictions, references=batch["target"])
        epoch_loss += loss.cpu().detach().numpy()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

  if  train_pretrain:
     return epoch_loss / len(iterator), metric.compute()["accuracy"], metric2.compute(average="weighted")["f1"]

  if not train_pretrain:
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def evaluate(model,iterator,train_pretrain=False):

    epoch_loss = 0.0
    epoch_acc = 0.0
    epoch_f1 = 0.0

    # deactivate the dropouts
    model.eval()
    metric = load_metric("accuracy")
    metric2 = load_metric("f1")
    # Sets require_grad flat False
    with torch.no_grad():
        for batch in iterator:
            if train_pretrain:
              b_input_ids = batch["input_ids"]
              b_input_mask = batch["attention_mask"]
              b_labels = batch["target"]
              outputs = model(b_input_ids,token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)

              loss = outputs.loss
              predictions = outputs.logits
              predictions = torch.argmax(predictions, dim=-1)
              lr_scheduler.step()
              metric.add_batch(predictions=predictions, references=batch["target"])
              metric2.add_batch(predictions=predictions, references=batch["target"])
              epoch_loss += loss.cpu().detach().numpy()

    if  train_pretrain:
      return epoch_loss / len(iterator), metric.compute()["accuracy"], metric2.compute(average="weighted")["f1"]
    if not train_pretrain:
      return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)


def test(model,iterator,train_pretrain=False):
    preds = []
    idss = []
    probas = []
    with torch.no_grad():
        for batch in iterator:
            if train_pretrain:
              b_input_ids = batch["input_ids"]
              b_input_mask = batch["attention_mask"]
              outputs = model(b_input_ids,token_type_ids=None,
                                  attention_mask=b_input_mask)
              predictions = outputs.logits
              idss+=[int(i) for i in batch["id"]]
              preds += torch.argmax(predictions, dim=-1).cpu().reshape(-1).int().tolist()

              predictions=predictions-predictions.min()
              predictions = predictions[:,1]/predictions.sum(axis=1)
              probas+= predictions.cpu().reshape(-1).tolist()



        mapp=dict(zip(idss,preds))
        mapp2=dict(zip(idss,probas))
        # pred=pd.read_csv("base_task_2.csv")
        # pred["polarity"]=pred["input-id-number"].map(mapp)
        # pred2 = pred.copy()
        # pred2[""]=pred2["id"].map(mapp2)
        return mapp, mapp2

def get_aspect_phrase(review, aspect_start, aspect_end):
    padded_review = "." + review + "."
    start = aspect_start
    end = aspect_end
    while padded_review[start] != '.' or padded_review[end] != '.':
        if padded_review[start] != '.':
            start -= 1
        if padded_review[end] != '.':
            end += 1
    return padded_review[start+1:end+1]

def preprocess_review(row):
    row['review'] = get_aspect_phrase(row['review'], int(row['start_position']), int(row['end_position']))
    row['polarity'] = str(int(row['polarity']) + 1)
    return row

def preprocess_review_final(row):
    row['review'] = get_aspect_phrase(row['review'], int(row['start_position']), int(row['end_position']))
    # row['polarity'] = str(int(row['polarity']) + 1)
    return row

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

# train_data_filepath = 'dataset-bert/train.csv'
# test_data_filepath = 'dataset-bert/test.csv'

train_data_filepath = 'dataset-bert/train_full.csv'
test_data_filepath = 'dataset-bert/test_full.csv'

final_eval_filepath = '../dataset/test/test_task2.csv'

raw_datasets_train = load_dataset('csv', data_files=train_data_filepath, delimiter=';')
preprocessed_datasets_train = raw_datasets_train.map(preprocess_review)
tokenized_datasets_train = preprocessed_datasets_train.map(lambda x: tokenizer(x['review']), batched=True)
tokenized_datasets_train = tokenized_datasets_train.rename_column('polarity', 'target')
tokenized_datasets_train = tokenized_datasets_train.remove_columns(['id', 'review', 'aspect', 'start_position', 'end_position'])
tokenized_datasets_train.set_format("torch")

raw_datasets_test = load_dataset('csv', data_files=test_data_filepath, delimiter=';')
preprocessed_datasets_test = raw_datasets_test.map(preprocess_review)
tokenized_datasets_test = preprocessed_datasets_test.map(lambda x: tokenizer(x['review']), batched=True)
tokenized_datasets_test = tokenized_datasets_test.rename_column('polarity', 'target')
tokenized_datasets_test = tokenized_datasets_test.remove_columns(['id', 'review', 'aspect', 'start_position', 'end_position'])
tokenized_datasets_test.set_format("torch")

raw_datasets_final = load_dataset('csv', data_files=final_eval_filepath, delimiter=';')
preprocessed_datasets_final = raw_datasets_final.map(preprocess_review_final)
tokenized_datasets_final = preprocessed_datasets_final.map(lambda x: tokenizer(x['review']), batched=True)
# tokenized_datasets_final = tokenized_datasets_final.rename_column('polarity', 'target')
tokenized_datasets_final = tokenized_datasets_final.remove_columns(['review', 'aspect', 'start_position', 'end_position'])
tokenized_datasets_final.set_format("torch")

print(tokenized_datasets_train["train"].column_names)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch_size = 32

train_dataloader = DataLoader(
    tokenized_datasets_train["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
# eval_dataloader = DataLoader(
#     tokenized_datasets_train["test"], batch_size=batch_size, collate_fn=data_collator
# )
test_dataloader = DataLoader(
    tokenized_datasets_test["train"], batch_size=batch_size, collate_fn=data_collator
)

final_dataloader = DataLoader(
    tokenized_datasets_final["train"], batch_size=batch_size, collate_fn=data_collator
)

epoch_number = 10

model = AutoModelForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=3)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epoch_number * len(train_dataloader),)

for epoch in range(1,epoch_number+1):
    print(f"\t Epoch: {epoch}")
    train_loss,train_acc,train_f1 = train(model,train_dataloader,optimizer,train_pretrain=True)
    valid_loss,valid_acc,valid_f1 = evaluate(model,test_dataloader,train_pretrain=True)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Train f1: {train_f1*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f} |  val. f1: {valid_f1*100:.2f}%')
    print()

pred_BERT, probas_BERT = test(model,final_dataloader,train_pretrain=True)

print(probas_BERT)
print()
print(pred_BERT)
