import pandas as pd
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.scheme import IOB2
from transformers import AutoModelForTokenClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from transformers import EarlyStoppingCallback


data = pd.read_csv(r"Building Code Dataset.csv")

texts = []
labels = []
for i in range(len(data['processed_content'])):
    text = data['processed_content'][i]
    label = data['label'][i].split()
    out_label = []
    words = text.split()
    for j in range(len(label)):
        out = [words[j], label[j]] 
        out_label.append(out)
    texts.append(text)
    labels.append(out_label)

label_map = {'O': 0, 'B-object': 1, 'I-quality': 2, 'B-quality': 3, 'I-object': 4, 'B-value': 5, 'I-value': 6, 'B-property': 7, 'I-property': 8, 'B-OP': 9, 'I-OP':10}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_map))

class TokenClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map):
        self.tokenizer = tokenizer
        self.label_map = label_map
        
        # Tokenize each text individually
        self.encodings = [tokenizer(text, truncation=True, padding='max_length', max_length=512, return_offsets_mapping=True) for text in texts]
        self.labels = [self.encode_tags(label, encoding) for label, encoding in zip(labels, self.encodings)]

    def encode_tags(self, doc_labels, encoding):
        # Convert labels to the format expected by the model
        word_ids = encoding.word_ids()  # Get word IDs for all tokens
        doc_enc_labels = []
        # Filter out None values and find the maximum valid word_id
        valid_word_ids = [id for id in word_ids if id is not None]
        if not valid_word_ids:  # If no valid IDs, return an empty list
            return []
        max_word_id = max(valid_word_ids)

        # Initialize label for each word with a default value to avoid index errors
        label_for_word = [-100] * (max_word_id + 1)
        for index, (word, tag) in enumerate(doc_labels):
            if index <= max_word_id:
                label_for_word[index] = self.label_map.get(tag, -100)

        for token_index, word_id in enumerate(word_ids):
            if word_id is not None:
                doc_enc_labels.append(label_for_word[word_id])
            else:
                doc_enc_labels.append(-100)  # Padding for special tokens like [CLS], [SEP], etc.

        return doc_enc_labels

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)
    
def split_data(texts, labels, test_size=0.1, random_state=None):
    # Split both the texts and labels using the same indices
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state)
    return train_texts, train_labels, val_texts, val_labels

index_to_label = {v: k for k, v in label_map.items()}
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convert indices to labels using the reversed label map
    true_labels = [[index_to_label[i] for i in label_row if i != -100] for label_row in labels]
    pred_labels = [[index_to_label[pred] for pred, true in zip(pred_row, label_row) if true != -100] for pred_row, label_row in zip(predictions, labels)]

    # Calculate metrics using seqeval
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

train_texts, train_labels, val_texts, val_labels = split_data(texts, labels, test_size=0.1, random_state=42)

train_dataset = TokenClassificationDataset(train_texts, train_labels, tokenizer, label_map)
val_dataset = TokenClassificationDataset(val_texts, val_labels, tokenizer, label_map)

training_args = TrainingArguments(
    output_dir='./Trained Models/results_6',
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=30,
    report_to="tensorboard",
    weight_decay=0.05,
    logging_dir='./Trained Models/logs_6',
    save_strategy="epoch",
    metric_for_best_model="eval_f1",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

trainer.evaluate()

# def tokenize_input(text, tokenizer):
#     return tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")


# def predict_tags2(text, tokenizer, model, label_map):
#     # Tokenization
#     encoded_input = tokenize_input(text, tokenizer)

#     # Prepare for model prediction
#     input_ids = encoded_input['input_ids'].to(model.device)
#     attention_mask = encoded_input['attention_mask'].to(model.device)

#     # Prediction
#     with torch.no_grad():
#         output = model(input_ids, attention_mask=attention_mask)
#         logits = output.logits
#         predictions = torch.argmax(logits, dim=-1)

#     # Decode predictions
#     word_predictions = []
#     word_ids = encoded_input.word_ids(batch_index=0)
#     for i, word_id in enumerate(word_ids):
#         if word_id is not None and word_id != -1:  # Ignore special tokens and out-of-range
#             predicted_tag = list(label_map.keys())[list(label_map.values()).index(predictions[0, i].item())]
#             decoded_word = tokenizer.decode([input_ids[0, i]], skip_special_tokens=True, clean_up_tokenization_spaces=True)
#             word_predictions.append((decoded_word, predicted_tag))

#     return word_predictions

# input_text = "The height of panels should not exceed 10 m" 
# predicted_tags = predict_tags2(input_text, tokenizer, model, label_map)
# print(predicted_tags)