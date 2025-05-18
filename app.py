from flask import Flask, request, jsonify
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/thuhang04/fish-diagnosis-model"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("input")

    if not input_text:
        return jsonify({"error": "Thiếu input"}), 400

    payload = {
        "inputs": input_text
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        return jsonify(response.json())
    except Exception:
        return jsonify({"error": "Không thể phân tích phản hồi từ Hugging Face"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
