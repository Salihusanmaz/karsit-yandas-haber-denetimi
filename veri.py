import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import re


# 1. CSV dosyasını oku
df = pd.read_csv("karsıt_yandas_haber_seti.csv")

# 2. Sütun adlarını ve metinleri temizle
df.columns = df.columns.str.strip()

df["medya_tipi"] = df["medya_tipi"].astype(str).str.lower().str.strip()
df["icerik"] = df["icerik"].fillna("").astype(str)
df["baslik"] = df["baslik"].fillna("").astype(str)

# 3. Veri kontrolü
# print("Sütunlar:")
# print(df.columns)

# print("\nVeri boyutu:")
# print(df.shape)

# print("\nHer olayda kaç haber var?")
# print(df.groupby("olay_id").size().value_counts())

# print("\nMedya tipleri:")
# print(df["medya_tipi"].unique())

# 4. Her olay için nötr haberi referans yap
referanslar = (
    df[df["medya_tipi"] == "notr"]
    .set_index("olay_id")["icerik"]
    .to_dict()
)

df["referans"] = df["olay_id"].map(referanslar)

# Eğer bazı olaylarda notr yoksa kendi içeriğini referans yap
df["referans"] = df["referans"].fillna(df["icerik"]).astype(str)

# 5. Similarity hesaplama
def calculate_similarity(row):
    text1 = str(row["referans"]).strip()
    text2 = str(row["icerik"]).strip()

    if text1 == "" or text2 == "":
        return 0

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])

    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return sim

df["similarity"] = df.apply(calculate_similarity, axis=1)

# 6. Değişiklik skoru
df["modification_score"] = 1 - df["similarity"]

# 7. Sonuçları göster
# print("\nİlk 10 sonuç:")
# print(df[["olay_id", "medya_tipi", "similarity", "modification_score"]].head(10))

# # 8. Yeni dosya olarak kaydet
# df.to_csv("haber_seti_skorlu.csv", index=False, encoding="utf-8-sig")

# print("\nİşlem tamamlandı. Yeni dosya oluşturuldu: haber_seti_skorlu.csv")


# 9. Provokatif kelimeler listesi
provokatif_kelimeler = [
    "skandal", "kriz", "şok", "rezalet", "kaos",
    "sert", "tehdit", "felaket", "gerilim", "panik",
    "öfke", "korku", "tepki", "eleştiri", "suçladı",
    "iddia", "iddialara göre", "tartışma", "gündem",
    "büyük tepki", "sert çıktı", "ağır sözler"
]

asiri_ifadeler = [
    "çok", "en", "asla", "kesinlikle", "mutlaka",
    "tamamen", "herkes", "hiç kimse"
]

def provokatif_skor(text):
    text = text.lower()
    
    skor = 0
    kelime_sayisi = len(text.split()) + 1

    # 1. Provokatif kelime yoğunluğu
    for kelime in provokatif_kelimeler:
        skor += text.count(kelime)

    # 2. Aşırı ifade kullanımı
    for kelime in asiri_ifadeler:
        skor += text.count(kelime) * 0.5

    # 3. Ünlem sayısı
    skor += text.count("!") * 2

    # 4. Büyük harf oranı (bağıran ton)
    buyuk = sum(1 for c in text if c.isupper())
    skor += buyuk / (len(text) + 1)

    # 5. Soru cümlesi (retorik olabilir)
    skor += text.count("?") * 0.5

    return skor / kelime_sayisi

df["provocative_score"] = df["icerik"].apply(provokatif_skor)

# 10. Yanıltıcılık label
def misleading_label(row):
    score = row["provocative_score"]
    sim = row["similarity"]

    if score > 0.12 and sim < 0.3:
        return 3
    elif score > 0.08 and sim < 0.5:
        return 2
    elif score > 0.04:
        return 1
    else:
        return 0

df["misleading_label"] = df.apply(misleading_label, axis=1)

# 11. Sonuç kontrol

# print("\nYeni özellikler:")
# print(df[["medya_tipi", "provocative_score", "misleading_label"]].head(10))

# print(df["provocative_score"].describe())
# print(df["misleading_label"].value_counts())

# =========================
# MODEL AŞAMASI
# =========================

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. EVENT BAZLI SPLIT
eventler = df["olay_id"].unique()

train_events, test_events = train_test_split(eventler, test_size=0.2, random_state=42)

train_df = df[df["olay_id"].isin(train_events)]
test_df = df[df["olay_id"].isin(test_events)]

print("\nTrain/Test boyutu:")
print(train_df.shape, test_df.shape)

# 2. TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df["icerik"])
X_test = vectorizer.transform(test_df["icerik"])

y_train = train_df["medya_tipi"]
y_test = test_df["medya_tipi"]

# 3. MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. SONUÇ
# accuracy = model.score(X_test, y_test)
# print("\nAccuracy:", accuracy)

# y_pred = model.predict(X_test)

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))


# X = metin
X_train = vectorizer.fit_transform(train_df["icerik"])
X_test = vectorizer.transform(test_df["icerik"])

# y = değişiklik skoru
y_train = train_df["modification_score"]
y_test = test_df["modification_score"]

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("\nMAE (Mean Absolute Error):", mae)

# MISLEADING CLASSIFICATION

y_train = train_df["misleading_label"]
y_test = test_df["misleading_label"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print("\nMisleading Classification:")
print(classification_report(y_test, y_pred))

df.to_csv("haber_seti_skorlu.csv", index=False, encoding="utf-8-sig")

print("Güncel dosya kaydedildi: haber_seti_skorlu.csv")
print(df.columns)