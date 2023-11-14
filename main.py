from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch
import json
import numpy as np

#dataをダウンロードしdatafreameに格納
with open("Q_A_embeddings.json", 'r',encoding="utf-8") as file:
    data = json.load(file)
df = pd.DataFrame(data)

# モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# 個々のテキストに対してembeddingを計算する関数
def embed_text(text):
    # テキストのトークン化とembeddingの計算
    batch_dict = tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch_dict)
    embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    return embedding[0].numpy()
#コサイン類似度の計算式
def cos_sim(v1, v2):#コサイン類似度
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


app = Flask(__name__)
messages = []



def get_reply(message):
    # ここにロジックを追加して、メッセージに対する返答を生成
    embed_input=embed_text(message)
    # Calculate cosine similarity for each row in the DataFrame
    df['similarity'] = df['embeddings'].apply(lambda x: cos_sim(x, embed_input))
    most_similar = df.loc[df['similarity'].idxmax()]

    return most_similar["A"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_message = request.form.get('message')
        messages.append({"text": user_message, "sender": "user"})
        app_reply = get_reply(user_message)
        messages.append({"text": app_reply, "sender": "app"})
    return render_template('chat.html', messages=messages)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
