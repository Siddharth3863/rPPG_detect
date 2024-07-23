import os
import torch
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from net_full import Mynet
import matplotlib.pyplot as plt

class VideoDataset(Dataset):
    def __init__(self, video_path, video_length):
        self.video_path = video_path
        self.video_length = video_length
        self.frames = self._extract_frames()
        print(f"Total extracted frames: {len(self.frames)}")  # Debugging statement

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
    model = Mynet(base_filter=64, video_length=opt.video_length, num_expert=opt.num_expert)
    print(f"Model instantiated: {model}")  # Debugging statement

    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")  # Debugging statement
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

    model.to(device)  # Move the model to the device
    model.eval()  # Set the model to evaluation mode
    print("Model moved to device and set to evaluation mode.")  # Debugging statement
    return model

def test(model, dataloader, device):
    if model is None:
        print("Model is not loaded. Exiting the test function.")
        return None

    # model.eval()
    all_outputs = []
    # Assuming 'test_data_loader' is your DataLoader instance
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_data_loader):
            print(f"Batch {batch_idx}: Inputs shape = {inputs.shape}")
            inputs = inputs.to(device)  # Move data to the device
            inputs = inputs.permute(0, 2, 1, 3, 4)  # Permute to [batch_size, channels, frames, height, width]
            print(f"After permute: Inputs shape = {inputs.shape}")
            outputs= model(inputs,opt.video_length)
            print(f"Outputs shape = {outputs.shape}")
            all_outputs.append(outputs)

    if len(all_outputs) == 0:
        print("No outputs collected, check DataLoader and data processing.")
        return None

    all_outputs = torch.cat(all_outputs)
    return all_outputs

def plot_signals(outputs, plot_dir):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    for i in range(outputs.shape[0]):
        plt.figure(figsize=(10, 4))
        plt.plot(outputs[i].cpu().numpy(), label='Predicted rPPG')  # Move to CPU for plotting
        plt.xlabel('Frame')
        plt.ylabel('Signal Value')
        plt.title(f'Test Case {i+1}')
        plt.legend()
        plot_path = os.path.join(plot_dir, f'test_case_{i+1}.png')
        plt.savefig(plot_path)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
    parser.add_argument('--video_length', type=int, default=128, help='video length')
    parser.add_argument('--video_path', type=str, required=True, help='path to the input video file')
    parser.add_argument('--model_path', type=str, default='rppg_model_66.pth', help='path to the saved model weights')
    parser.add_argument('--plot_dir', type=str, default='test_plots', help='directory to save the plots')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU if available')
    parser.add_argument('--num_expert', type=int, default=9, help='Number of experts')

    opt = parser.parse_args()

    device = torch.device("cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    print('===> Loading video dataset')
    test_set = VideoDataset(opt.video_path, opt.video_length)
    print(f"Loaded {len(test_set)} samples.")  # Debugging statement
    test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batchSize, shuffle=False, drop_last=True)

    print('===> Loading the model')
    model = load_model(opt.model_path, device)

    print('===> Testing the model')
    outputs = test(model, test_data_loader, device)

    if outputs is not None:
        outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
        plot_signals(outputs, opt.plot_dir)
    else:
        print('No outputs to plot.')
