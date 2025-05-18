from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch

# Load model từ Hugging Face (tự động tải nếu chưa có)
model = SentenceTransformer("thuhang04/fish-diagnosis-model")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_text = data.get("input")
        symptom_list = data.get("trieu_chung_list")

        if not input_text or not symptom_list:
            return jsonify({"error": "Thiếu dữ liệu đầu vào"}), 400

        # Tính embedding
        input_embedding = model.encode(input_text, convert_to_tensor=True)
        symptom_embeddings = model.encode(symptom_list, convert_to_tensor=True)

        # Tính cosine similarity
        scores = util.pytorch_cos_sim(input_embedding, symptom_embeddings)[0]
        best_idx = torch.argmax(scores).item()

        result = {
            "trieu_chung": symptom_list[best_idx],
            "score": float(scores[best_idx])
        }
        return jsonify([result])

    except Exception as e:
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
