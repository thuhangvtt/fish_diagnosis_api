from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os

model = SentenceTransformer("thuhang04/fish-diagnosis-model")
model = SentenceTransformer(model_path)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("input")
    trieu_chung_list = data.get("trieu_chung_list")

    if not input_text or not trieu_chung_list:
        return jsonify({"error": "Thiếu input hoặc danh sách triệu chứng"}), 400

    input_emb = model.encode(input_text, convert_to_tensor=True)
    results = []

    for tc in trieu_chung_list:
        emb = model.encode(tc, convert_to_tensor=True)
        score = util.cos_sim(input_emb, emb).item()
        results.append({"trieu_chung": tc, "similarity": round(score, 4)})

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return jsonify(results[:5])

if __name__ == "__main__":
    app.run()
