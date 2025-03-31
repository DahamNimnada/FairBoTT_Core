from flask import Flask, request, jsonify
from flask_cors import CORS
from fairbott.api.api_pipeline_test import detect_bias_from_text

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can call this API


@app.route('/api/detect-bias', methods=['POST'])
def detect_bias():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request'}), 400

    input_text = data['text']
    result = detect_bias_from_text(input_text)

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True)