import transformers
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score
import torch
import numpy as np
import random
import time

train_size = 40
learning_rate = 5e-5
epoch_num = 50

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device: {}".format(device))
checkpoint = "bert-base-cased" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5).to(device)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", num_train_epochs=epoch_num, learning_rate=learning_rate)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs > 0.5)] = 1
    return {'f1':f1_score(y_true=labels, y_pred=predictions, average='micro')}





def main():
    df_examples = pd.read_csv('examples/df/price500.csv', delimiter=';')
    df_labels =[]
    df_text = []
    for i in range(len(df_examples)):
        df_labels.append([x for x in df_examples.iloc[i,2:]])
        df_text.append(df_examples.iloc[i,1])
    df_test = pd.DataFrame({"text": df_text, "label": df_labels})
    df_train = df_test.iloc[random.sample(range(len(df_test)),train_size),:]
    print(df_train)
    train = Dataset.from_pandas(df_train)
    test = Dataset.from_pandas(df_test) 
    dataset = DatasetDict()
    dataset['train'] = train
    dataset['test'] = test

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    with open('results/NN/train_{}_lr_{}.txt'.format(train_size,learning_rate), 'w+') as f:
      for obj in trainer.state.log_history:
        f.write(str(obj)+"\n")
      

if __name__ == "__main__":
    start_time = time.time()
    main()
    print('---------- {:.1f} minutes ----------'.format((time.time() - start_time) / 60))
    print()
