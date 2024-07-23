from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Dummy function to process the video and extract rPPG signals
def process_video(file_path, method):
    # This is a dummy implementation. Replace with actual video processing logic.
    time = list(range(100))  # Example time data
    rppg = np.sin(np.linspace(0, 2 * np.pi, 100))  # Example rPPG signal
    return time, rppg.tolist()

@app.route('/upload-pfe', methods=['POST'])
def upload_pfe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            time, rppg = process_video(temp.name, 'PFE/TFA')
            os.remove(temp.name)
            return jsonify({'time': time, 'rppg': rppg})

@app.route('/upload-phys', methods=['POST'])
def upload_phys():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            time, rppg = process_video(temp.name, 'PhysNet')
            os.remove(temp.name)
            return jsonify({'time': time, 'rppg': rppg})

@app.route('/upload', methods=['POST'])
def upload_ssl():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            time, rppg = process_video(temp.name, 'SSL')
            os.remove(temp.name)
            return jsonify({'time': time, 'rppg': rppg})

if __name__ == '__main__':
    app.run(debug=True)
