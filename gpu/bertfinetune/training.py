import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])  # Ensure text is a string
        label = self.labels[index]

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# Load and preprocess your sentiment dataset
rawdata = pd.read_excel('train_set.xlsx')

# Data cleaning and preprocessing
clean_texts = []
clean_labels = []
for text, label in zip(rawdata['post_content'], rawdata['sentiment_score']):
    if pd.isna(text) or pd.isna(label):
        print(f"Warning: Skipping entry with null text or label")
        continue
    try:
        clean_text = str(text).strip()
        clean_label = int(label)
        if clean_text and 0 <= clean_label <= 2:  # Assuming sentiment scores are 0, 1, or 2
            clean_texts.append(clean_text)
            clean_labels.append(clean_label)
        else:
            print(f"Warning: Skipping invalid entry. Text: '{clean_text[:20]}...', Label: {clean_label}")
    except ValueError:
        print(f"Warning: Invalid label {label} found. Skipping this entry.")

print(f"Total clean entries: {len(clean_texts)}")
print(f"Label distribution: {pd.Series(clean_labels).value_counts()}")

# Convert labels to tensor
labels = torch.tensor(clean_labels, dtype=torch.long)

# KcBERT 모델과 토크나이저 로드
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 원하는 최대 시퀀스 길이
max_length = 128

dataset = CustomDataset(clean_texts, labels, tokenizer, max_length)

# 데이터 로더 생성
batch_size = 32

# Train / Test set 분리
train, test = train_test_split(dataset, test_size=0.15, random_state=42)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 하이퍼파라미터 설정
learning_rate = 2e-5  # 학습률 약간 증가
epochs = 10  # 에폭 수 증가
batch_size = 16  # 배치 사이즈 감소
weight_decay = 0.01  # 가중치 감소 추가

# Train / Test set 분리 (변경 없음)
train, test = train_test_split(dataset, test_size=0.15, random_state=42)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 옵티마이저 및 손실 함수 설정
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# 학습률 스케줄러 추가
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# 모델 재학습
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # 학습률 스케줄러 스텝
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    # 모델 평가 (변경 없음)
    model.eval()
    val_total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_batch in valid_dataloader:
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['label'].to(device)

            val_outputs = model(val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
            val_loss = val_outputs.loss
            val_total_loss += val_loss.item()

            val_preds = val_outputs.logits.argmax(dim=1)
            correct += (val_preds == val_labels).sum().item()
            total += val_labels.size(0)

    val_avg_loss = val_total_loss / len(valid_dataloader)
    val_accuracy = correct / total
    print(f"Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")

# 모델 저장 (변경 없음)
model_save_path = "sentiment_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")