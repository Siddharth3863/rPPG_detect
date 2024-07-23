from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class VideoDataset(Dataset):
    def __init__(self, video_path, video_length):
        self.video_path = video_path
        self.video_length = video_length
        self.frames = self._extract_frames()

    def _extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))  # Resize frame to 128x128
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        num_frames = frames.shape[0]
        if num_frames < self.video_length:
            padding = self.video_length - num_frames
            pad_frames = np.tile(frames[-1], (padding, 1, 1, 1))
            frames = np.concatenate((frames, pad_frames), axis=0)
        return frames

    def __len__(self):
        return 1  # Only one video

    def __getitem__(self, idx):
        frames = self.frames[:self.video_length]
        frames = np.transpose(frames, (3, 0, 1, 2))  # Convert to [channels, frames, height, width]
        frames = frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
        return torch.from_numpy(frames)

def load_model(model_path, device):
    model = PhysNet_padding_Encoder_Decoder_MAX()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode
    return model

def process_video(file_path, method, model, device):
    if method == 'PhysNet':
        video_length = 128  # Set the video length for processing
        dataset = VideoDataset(file_path, video_length)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_outputs = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)  # Move data to the device
                inputs = inputs.permute(0, 1, 2, 3, 4)  # Permute to [batch_size, channels, frames, height, width]
                outputs, _, _, _ = model(inputs)
                all_outputs.append(outputs)

        if len(all_outputs) == 0:
            return [], []

        all_outputs = torch.cat(all_outputs)
        all_outputs = (all_outputs - torch.mean(all_outputs)) / torch.std(all_outputs)
        time = list(range(len(all_outputs[0])))
        rppg = all_outputs[0].cpu().numpy().tolist()
        return time, rppg
    
    if method == 'SSL':
        video_length = 128  # Set the video length for processing
        dataset = VideoDataset(file_path, video_length)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_outputs = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)  # Move data to the device
                inputs = inputs.permute(0, 2, 1, 3, 4)  # Permute to [batch_size, channels, frames, height, width]
                outputs, _, _, _ = model(inputs)
                all_outputs.append(outputs)

        if len(all_outputs) == 0:
            return [], []

        all_outputs = torch.cat(all_outputs)
        all_outputs = (all_outputs - torch.mean(all_outputs)) / torch.std(all_outputs)
        time = list(range(len(all_outputs[0])))
        rppg = all_outputs[0].cpu().numpy().tolist()
        return time, rppg


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
            time, rppg = process_video(temp.name, 'PFE/TFA', model, device)
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
            time, rppg = process_video(temp.name, 'PhysNet', model, device)
            os.remove(temp.name)
            return jsonify({'time': time, 'rppg': rppg})

@app.route('/upload-ssl', methods=['POST'])
def upload_ssl():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            time, rppg = process_video(temp.name, 'SSL', model, device)
            os.remove(temp.name)
            return jsonify({'time': time, 'rppg': rppg})

if __name__ == '__main__':
    model_path1 = 'physnet_test_weights.pth'  # Path to the saved model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path1, device)
    app.run(debug=True,port = 5000)
