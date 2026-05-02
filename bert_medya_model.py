import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =========================
# 1. VERİYİ OKU
# =========================

df = pd.read_csv("haber_seti_skorlu.csv")

df["icerik"] = df["icerik"].fillna("").astype(str)
df["medya_tipi"] = df["medya_tipi"].str.lower().str.strip()

label_map = {
    "yandas": 0,
    "karsit": 1,
    "notr": 2
}

df["label"] = df["medya_tipi"].map(label_map)

print(df[["medya_tipi", "label"]].head())
print(df["label"].value_counts())

# =========================
# 2. EVENT BAZLI SPLIT
# =========================

eventler = df["olay_id"].unique()

train_events, test_events = train_test_split(
    eventler,
    test_size=0.2,
    random_state=42
)

train_df = df[df["olay_id"].isin(train_events)].reset_index(drop=True)
test_df = df[df["olay_id"].isin(test_events)].reset_index(drop=True)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

# =========================
# 3. TOKENIZER
# =========================

model_name = "dbmdz/bert-base-turkish-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

class HaberDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HaberDataset(
    train_df["icerik"].tolist(),
    train_df["label"].tolist()
)

test_dataset = HaberDataset(
    test_df["icerik"].tolist(),
    test_df["label"].tolist()
)

# =========================
# 4. BERT MODEL
# =========================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# =========================
# 5. TRAINING AYARLARI
# =========================

training_args = TrainingArguments(
    output_dir="./bert_medya_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# =========================
# 6. EĞİTİM
# =========================

trainer.train()

# =========================
# 7. TEST SONUCU
# =========================

predictions = trainer.predict(test_dataset)

y_pred = predictions.predictions.argmax(axis=1)
y_true = test_df["label"].tolist()

print("\nBERT Classification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["yandas", "karsit", "notr"]
    )
)

# =========================
# 8. MODELİ KAYDET
# =========================

model.save_pretrained("./bert_medya_model")
tokenizer.save_pretrained("./bert_medya_model")

print("\nModel kaydedildi: bert_medya_model")