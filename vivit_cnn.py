import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import torch.nn as nn
import random
import sys
import pickle
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import h5py
from collections import defaultdict
from ultralytics import YOLO
import hashlib
import requests
import json
import uuid
import tempfile
import math


# Device settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 
DTYPE = torch.float16

logging.basicConfig(level=logging.DEBUG)

model_yolo = YOLO('yolov8x.pt')
model_yolo.to(device)

keypoints = {}
bboxes = {}

def normalize_batch_images(batch, low_percentile=1, high_percentile=99):
    """
    Нормализует батч изображений.
    
    :param batch: Тензор формы (B, C, H, W), где B - размер батча, C - количество каналов (обычно 3 для RGB)
    :param low_percentile: Нижний перцентиль для нормализации (по умолчанию 1)
    :param high_percentile: Верхний перцентиль для нормализации (по умолчанию 99)
    :return: Нормализованный батч изображений в формате numpy array
    """
    # Переводим в numpy, если это torch тензор
    if isinstance(batch, torch.Tensor):
        batch = batch.cpu().detach().numpy()
    
    # Создаем копию батча для нормализации
    normalized_batch = np.copy(batch)
    
    for i in range(batch.shape[0]):  # Итерация по изображениям в батче
        for c in range(batch.shape[1]):  # Итерация по каналам
            channel = batch[i, c]
            low = np.percentile(channel, low_percentile)
            high = np.percentile(channel, high_percentile)
            normalized_batch[i, c] = np.clip((channel - low) / (high - low), 0, 1)
    
    return normalized_batch
    
# Инициализация ControlNet, UNet и VAE моделей
def initialize_models(model_path="./controlnet-hands"):
    return None, None, None, None

    
def get_embeddings(images, vae, controlnet, text_encoder, tokenizer):
    images = images.to(device, dtype=DTYPE)

    result = normalize_batch_images(images)
    result = torch.from_numpy(result).to(device=device)
    
    return result

def preprocess_and_save_dataset(dataset, vae, controlnet, text_encoder, tokenizer, output_file, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#    vae.eval()
    
    # Получаем размерность эмбеддингов из первого батча
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_sequences, _ = sample_batch
        sample_frame = sample_sequences[:, 0, :, :, :].to(device, dtype=torch.float32)
        sample_embedding = get_embeddings(sample_frame, vae, controlnet, text_encoder, tokenizer)
        embedding_shape = sample_embedding.shape[1:]
    
    with h5py.File(output_file, 'w') as hf:
        embeddings_group = hf.create_group('embeddings')
        labels_group = hf.create_group('labels')
        
        embeddings_dataset = embeddings_group.create_dataset('data', shape=(0, sample_sequences.size(1), *embedding_shape), 
                                                             maxshape=(None, sample_sequences.size(1), *embedding_shape),
                                                             compression="gzip",
                                                             dtype='f')
        labels_dataset = labels_group.create_dataset('data', shape=(0,), maxshape=(None,), dtype='i')
        
        with torch.no_grad():
            for batch_sequences, batch_labels in tqdm(dataloader, desc="Preprocessing"):
                batch_embeddings = []
                for i in range(batch_sequences.size(1)):
                    frame = batch_sequences[:, i, :, :, :].to(device, dtype=torch.float32)
                    embedding = get_embeddings(frame, vae, controlnet, text_encoder, tokenizer)
                    batch_embeddings.append(embedding)
                
                batch_embeddings = torch.stack(batch_embeddings, dim=1).cpu().numpy()
                batch_labels = batch_labels.numpy()
                
                current_size = embeddings_dataset.shape[0]
                new_size = current_size + batch_embeddings.shape[0]
                embeddings_dataset.resize(new_size, axis=0)
                embeddings_dataset[-batch_embeddings.shape[0]:] = batch_embeddings
                
                labels_dataset.resize(new_size, axis=0)
                labels_dataset[-batch_labels.shape[0]:] = batch_labels
    
    print(f"Dataset saved to {output_file}")

class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = hf['embeddings/data'].shape[0]

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as hf:
            embedding = torch.from_numpy(hf['embeddings/data'][index])
            label = torch.tensor(hf['labels/data'][index], dtype=torch.long)
        return embedding, label

    def __len__(self):
        return self.length


class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.residual = nn.Linear(input_size, hidden_size * 2)
        self.ln = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        residual = self.residual(x)
        out, _ = self.lstm(x)
        out = out + residual
        out = self.ln(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class CNNBiLSTMWithAttention(nn.Module):
    def __init__(self, num_classes, sequence_length=32, lstm_hidden_size=256):
        super(CNNBiLSTMWithAttention, self).__init__()
        
        self.sequence_length = sequence_length
        
        self.cnn = nn.Sequential(
            nn.Conv3d(sequence_length, 32, kernel_size=(3, 3, 16), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 9), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.AdaptiveAvgPool3d((sequence_length, 20, 20))
        )
        
        self.cnn_output_size = 64 * 20 * 20
        
        self.bn1 = nn.BatchNorm1d(self.cnn_output_size)
        self.dropout1 = nn.Dropout(0.1)
        
        self.residual_lstm = ResidualLSTM(self.cnn_output_size, lstm_hidden_size)
        
        self.attention = AttentionLayer(lstm_hidden_size * 2)
        
        self.ln = nn.LayerNorm(lstm_hidden_size * 2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param.data)
                else:
                    nn.init.uniform_(param.data, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1, 3, 4).reshape(batch_size, self.sequence_length, -1)
        
        cnn_out = self.bn1(cnn_out.reshape(-1, self.cnn_output_size)).reshape(batch_size, self.sequence_length, -1)
#        cnn_out = self.dropout1(cnn_out)
        
        lstm_out = self.residual_lstm(cnn_out)
        
        context_vector, attention_weights = self.attention(lstm_out)
        
        context_vector = self.ln(context_vector)
        context_vector = self.dropout2(context_vector)
        output = self.fc(context_vector)
        
        return output


def get_bounding_box(keypoints):
    x_coords, y_coords = zip(*keypoints)
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

def draw_keypoints(image, keypoints, scores, color, base_radius=2, max_radius=10):
    for point, score in zip(keypoints, scores):
        x, y = map(int, point)
        radius = int(base_radius + (max_radius - base_radius) * score)
        cv2.circle(image, (x, y), radius, color, -1)

def draw_keypoints_with_data(image, original, keypoints, scores, color, base_radius=2, max_radius=20, x_min=0, y_min=0):
    for point, score in zip(keypoints, scores):
        x, y = map(int, point)
        radius = int(base_radius + (max_radius - base_radius) * score * 4)
        
        if x < 0 or y < 0 or x >= original.shape[1] or y >= original.shape[0]:
            continue
        left = max(0, x - radius)
        top = max(0, y - radius)
        right = min(original.shape[1], x + radius)
        bottom = min(original.shape[0], y + radius)
        
        if left >= right or top >= bottom:
            continue
        keypoint_area = original[top:bottom, left:right]
        
        new_left = max(0, x - radius - x_min)
        new_top = max(0, y - radius - y_min)
        
        # Вычисляем размер области назначения
        dest_height = min(image.shape[0] - new_top, 2*radius)
        dest_width = min(image.shape[1] - new_left, 2*radius)
        
        # Изменяем размер keypoint_area, чтобы соответствовать области назначения
        keypoint_area_resized = cv2.resize(keypoint_area, (dest_width, dest_height))
        
        # Убеждаемся, что область назначения не выходит за границы изображения
        dest_right = min(image.shape[1], new_left + dest_width)
        dest_bottom = min(image.shape[0], new_top + dest_height)
        
        # Обрезаем keypoint_area_resized, если необходимо
        keypoint_area_resized = keypoint_area_resized[:dest_bottom-new_top, :dest_right-new_left]
        
        # Копируем измененную область в изображение
        image[new_top:dest_bottom, new_left:dest_right] = keypoint_area_resized
        
        # Рисуем эллипс
        center = (new_left + (dest_right - new_left) // 2, new_top + (dest_bottom - new_top) // 2)
        axes = ((dest_right - new_left) // 2, (dest_bottom - new_top) // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, color, 1)

def draw_motion_arrows(image, current_keypoints, next_keypoints, color=(255, 0, 0), threshold=0.1):
    for current, next in zip(current_keypoints, next_keypoints):
        x1, y1 = map(int, current)
        x2, y2 = map(int, next)
        
        # Calculate displacement
        dx, dy = x2 - x1, y2 - y1
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Draw arrow only if displacement is above threshold
        if displacement > threshold and displacement < 100:
            cv2.arrowedLine(image, (x1, y1), (x2, y2), color, 1, tipLength=0.2)


def create_motion_intensity_map(image, current_keypoints, next_keypoints, color=(0, 0, 255), sigma=20, magnitude_scale=15, movement_threshold=0.01):
    h, w = image.shape[:2]
    motion_map = np.zeros((h, w), dtype=np.float32)

    current_keypoints = np.array(current_keypoints)
    next_keypoints = np.array(next_keypoints)

    # Проверка размерности и преобразование при необходимости
    if current_keypoints.ndim == 1:
        current_keypoints = current_keypoints.reshape(-1, 2)
    if next_keypoints.ndim == 1:
        next_keypoints = next_keypoints.reshape(-1, 2)

    # Нормализация координат
    current_keypoints[:, 0] = np.clip(current_keypoints[:, 0], 0, w-1)
    current_keypoints[:, 1] = np.clip(current_keypoints[:, 1], 0, h-1)
    next_keypoints[:, 0] = np.clip(next_keypoints[:, 0], 0, w-1)
    next_keypoints[:, 1] = np.clip(next_keypoints[:, 1], 0, h-1)

    for current, next in zip(current_keypoints, next_keypoints):
        x1, y1 = map(int, current)
        x2, y2 = map(int, next)
        dx, dy = x2 - x1, y2 - y1
        magnitude = np.sqrt(dx**2 + dy**2)

        # Применяем порог движения
        if magnitude < movement_threshold:
            continue

        # Применяем масштабирование к величине движения
        scaled_magnitude = magnitude * magnitude_scale

        # Создаем гауссово ядро
        y, x = np.ogrid[-sigma:sigma+1, -sigma:sigma+1]
        kernel = np.exp(-(x*x + y*y) / (2.*sigma*sigma))

        # Определяем область применения ядра
        x_min, x_max = max(0, x1-sigma), min(w, x1+sigma+1)
        y_min, y_max = max(0, y1-sigma), min(h, y1+sigma+1)
        kernel_x_min = max(0, sigma - (x1 - x_min))
        kernel_x_max = min(2*sigma+1, sigma + (x_max - x1))
        kernel_y_min = max(0, sigma - (y1 - y_min))
        kernel_y_max = min(2*sigma+1, sigma + (y_max - y1))

        # Применяем ядро к карте движения
        kernel_part = kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
        map_part = motion_map[y_min:y_max, x_min:x_max]
        if kernel_part.shape == map_part.shape:
            motion_map[y_min:y_max, x_min:x_max] += kernel_part * scaled_magnitude

    # Нормализация карты движения
    max_motion = np.max(motion_map)
    if max_motion > 0:
        motion_map = motion_map / max_motion

    # Создание цветной карты
    color_map = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(3):
        color_map[:,:,i] = motion_map * color[i] / 255.0

    return color_map

def combine_motion_maps(image, *motion_maps):
    combined_map = np.zeros_like(image, dtype=np.float32)
    for map in motion_maps:
        combined_map += map
    
    # Clip values to [0, 1] range
    combined_map = np.clip(combined_map, 0, 1)

    # Гамма-коррекция для улучшения визуализации
    gamma = 0.4
    combined_map = np.power(combined_map, gamma)

    # Наложение цветной карты на исходное изображение
    result = cv2.addWeighted(image, 0.7, (combined_map * 255).astype(np.uint8), 0.3, 0)

    return result

    
def resize_to_512(image):
    """Функция для изменения размера изображения до 512x512"""
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

def check_and_resize_images(images):
    """Проверяет размеры изображений и изменяет их при необходимости"""
    target_size = (512, 512)
    resized_images = []
    
    for img in images:
        if img.shape[:2] != target_size:
            resized_img = resize_to_512(img)
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    return resized_images

        
class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=32, augmentations_per_sample=3):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.augmentations_per_sample = augmentations_per_sample
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.video_sequences = self._load_sequences()

        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=0.6),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.6),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.6),
            A.Resize(256, 256),
            ToTensorV2(),
        ])

        self.notaug_transform = A.Compose([
            A.Resize(256, 256),
            ToTensorV2(),
        ])
        
    def _load_sequences(self):
        video_sequences = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            videos = set()
            for filename in os.listdir(class_dir):
                video_id = filename.split('_frame_')[0]
                videos.add(video_id)
            
            for video_id in videos:
                frames = sorted([f for f in os.listdir(class_dir) if f.startswith(video_id)])
                
                if len(frames) >= self.sequence_length:
                    step = len(frames) / self.sequence_length
                    selected_indices = [int(i * step) for i in range(self.sequence_length)]
                    sequence = [frames[i] for i in selected_indices]
                else:
                    sequence = frames + [frames[-1]] * (self.sequence_length - len(frames))
                
                video_sequences.append((class_name, video_id, sequence))
        
        return video_sequences

    def __len__(self):
        return len(self.video_sequences) * (self.augmentations_per_sample + 1)  # +1 для оригинальной последовательности

    def __getitem__(self, idx):
        original_idx = idx // (self.augmentations_per_sample + 1)
        aug_idx = idx % (self.augmentations_per_sample + 1)
        
        class_name, video_id, sequence = self.video_sequences[original_idx]
        
        images = []
        images_seq = []
        
        x1 = 0
        y1 = 0
        x2 = 640
        y2 = 640
        
        temp_dir = "c:\\Users\\Profi\\Downloads\\ContourletCNN\\prepared"
        
        for frame in sequence:
            img_path = os.path.join(self.root_dir, class_name, frame)
            temp_filename = frame
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            name, ext = os.path.splitext(frame)
            if os.path.exists(f"prepared4ch\\{name}_1{ext}"):
                image_original = cv2.imread(f"prepared4ch\\{name}_1{ext}")
                image_white = cv2.imread(f"prepared4ch\\{name}_2{ext}")
                image_with_data = cv2.imread(f"prepared4ch\\{name}_3{ext}")
                image_with_motion = cv2.imread(f"prepared4ch\\{name}_4{ext}")
                
                image_original = cv2.resize(image_original, (512, 512), interpolation=cv2.INTER_AREA)
                image_white = cv2.resize(image_white, (512, 512), interpolation=cv2.INTER_AREA)
                image_with_data = cv2.resize(image_with_data, (512, 512), interpolation=cv2.INTER_AREA)
                image_with_motion = cv2.resize(image_with_motion, (512, 512), interpolation=cv2.INTER_AREA)
                
                multi_channel_image = np.stack([
#                    image_original,
#                    image_white,
                    image_with_data,
                    image_with_motion
                ], axis=-1)

                # Преобразуем в тензор PyTorch
                # Меняем оси для соответствия формату PyTorch: (channels, height, width)
                multi_channel_tensor = torch.from_numpy(multi_channel_image).permute(3, 2, 0, 1)

                images_seq.append(multi_channel_tensor)
                continue
                
            if os.path.exists(temp_filepath):
                image = cv2.imread(temp_filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if img_path not in bboxes:
                    # Находим и вырезаем изображение человека
                    results = model_yolo(img_path, verbose=False)
                    person_detected = False
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            if box.cls == 0:
                                person_detected = True
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                break
                        if person_detected:
                            break
                else:
                    x1 = bboxes[img_path][0]
                    y1 = bboxes[img_path][1]
                    x2 = bboxes[img_path][2]
                    y2 = bboxes[img_path][3]
                    
                image = image[int(y1):int(y2), int(x1):int(x2)]
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
                
            images.append(image)
            bboxes[img_path] = [x1, y1, x2, y2]

        if len(images_seq) == self.sequence_length:
            image_sequence = torch.stack(images_seq)
            label = self.class_to_idx[class_name]
            return image_sequence, label
#        else:
#            print(f"{video_id} not in cache {len(images_seq)}, {self.sequence_length}")
            
        # Запрос ключевых точек через API
        keypoints = {}
        url = "http://localhost:5000/detect_hands"
        headers = {'Content-Type': 'application/json'}
                
        for i, (frame, image) in enumerate(zip(sequence, images)):
            key = hashlib.md5(image.tobytes()).hexdigest()
            
            if key not in keypoints:
                temp_filename = frame
                temp_filepath = os.path.join(temp_dir, temp_filename)
                
                # Сохраняем изображение во временный файл
                cv2.imwrite(temp_filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
                # Подготавливаем payload с путем к временному файлу
                payload = json.dumps({
                    "file_path": temp_dir,
                    "file_name": temp_filename
                })
                
                response = requests.request("POST", url, headers=headers, data=payload)
                keypoints[key] = response.json()
            
            # Обработка следующего кадра, если он существует
            if i + 1 < len(images):
                next_frame = sequence[i + 1]
                next_image = images[i + 1]
                next_key = hashlib.md5(next_image.tobytes()).hexdigest()
                
                if next_key not in keypoints:
                    temp_filename = next_frame
                    temp_filepath = os.path.join(temp_dir, temp_filename)
                    
                    # Сохраняем изображение во временный файл
                    cv2.imwrite(temp_filepath, cv2.cvtColor(next_image, cv2.COLOR_RGB2BGR))
                    
                    # Подготавливаем payload с путем к временному файлу
                    next_payload = json.dumps({
                        "file_path": temp_dir,
                        "file_name": temp_filename
                    })
                
                    next_response = requests.request("POST", url, headers=headers, data=next_payload)
                    keypoints[next_key] = next_response.json()
            
            else:
                next_key = key
            
            data = keypoints[key]  
            next_data = keypoints[next_key]
            
            # Get all keypoints
            all_keypoints = (
                data['body_detections']['keypoints'][0] +
                data['face_detections']['keypoints'][0]
            )

            for hand in data['hand_detections']:
                hand_box = next(iter(hand.values()))
                hand_keypoints = hand['keypoints'][0]
                all_keypoints.extend([[x + hand_box[0], y + hand_box[1]] for x, y in hand_keypoints])

            # Calculate bounding box
            x_min, y_min, x_max, y_max = get_bounding_box(all_keypoints)

            # Add padding
            original_image = image.copy()
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(original_image.shape[1], x_max + padding)
            y_max = min(original_image.shape[0], y_max + padding)

            image_original = original_image.copy()
            image_white = np.ones((int(y_max - y_min), int(x_max - x_min), 3), dtype=np.uint8) * 255
            image_with_data = np.ones((int(y_max - y_min), int(x_max - x_min), 3), dtype=np.uint8) * 255
            image_with_motion = np.ones((int(y_max - y_min), int(x_max - x_min), 3), dtype=np.uint8) * 255

            # Draw body keypoints
            body_keypoints = data['body_detections']['keypoints'][0]
            body_scores = data['body_detections']['keypoint_scores'][0]
            draw_keypoints(image_original, body_keypoints, body_scores, (0, 0, 255))  # Red in BGR
            draw_keypoints(image_white, [[x-x_min, y-y_min] for x, y in body_keypoints], body_scores, (0, 0, 255))
            draw_keypoints_with_data(image_with_data, original_image, body_keypoints, body_scores, (0, 0, 255), base_radius=2, max_radius=20, x_min=x_min, y_min=y_min)
#            draw_keypoints(image_with_motion, [[x-x_min, y-y_min] for x, y in body_keypoints], body_scores, (0, 0, 255))

            # Draw face keypoints
            face_keypoints = data['face_detections']['keypoints'][0]
            face_scores = data['face_detections']['keypoint_scores'][0]
            draw_keypoints(image_original, face_keypoints, face_scores, (255, 0, 0), base_radius=1, max_radius=5)  # Blue in BGR
            draw_keypoints(image_white, [[x-x_min, y-y_min] for x, y in face_keypoints], face_scores, (255, 0, 0), base_radius=1, max_radius=5)
            draw_keypoints_with_data(image_with_data, original_image, face_keypoints, face_scores, (255, 0, 0), base_radius=1, max_radius=5, x_min=x_min, y_min=y_min)
#            draw_keypoints(image_with_motion, [[x-x_min, y-y_min] for x, y in face_keypoints], face_scores, (255, 0, 0), base_radius=1, max_radius=5)

            # Draw hand keypoints
            for hand in data['hand_detections']:
                hand_box = next(iter(hand.values()))
                hand_keypoints = hand['keypoints'][0]
                hand_scores = hand['keypoint_scores'][0]
                
                adjusted_hand_keypoints = [
                    [x + hand_box[0], y + hand_box[1]] for x, y in hand_keypoints
                ]
                
                draw_keypoints(image_original, adjusted_hand_keypoints, hand_scores, (0, 255, 0), base_radius=1, max_radius=5)  # Green in BGR
                draw_keypoints(image_white, [[x-x_min, y-y_min] for x, y in adjusted_hand_keypoints], hand_scores, (0, 255, 0), base_radius=1, max_radius=5)
                draw_keypoints_with_data(image_with_data, original_image, adjusted_hand_keypoints, hand_scores, (0, 255, 0), base_radius=1, max_radius=5, x_min=x_min, y_min=y_min)
#                draw_keypoints(image_with_motion, [[x-x_min, y-y_min] for x, y in adjusted_hand_keypoints], hand_scores, (0, 255, 0), base_radius=1, max_radius=5)

            # Draw motion arrows
            if next_key != key:  # Only draw motion arrows if we have a different next frame
                # For body
                next_body_keypoints = next_data['body_detections']['keypoints'][0]
                body_motion_map = create_motion_intensity_map(image_with_motion, 
                                   [[x-x_min, y-y_min] for x, y in body_keypoints], 
                                   [[x-x_min, y-y_min] for x, y in next_body_keypoints], (0, 0, 255))
                
                # For face
                next_face_keypoints = next_data['face_detections']['keypoints'][0]
                face_motion_map = create_motion_intensity_map(image_with_motion, 
                                   [[x-x_min, y-y_min] for x, y in face_keypoints], 
                                   [[x-x_min, y-y_min] for x, y in next_face_keypoints], (255, 0, 0))
                
                # For hands
                current_hand_keypoints = []
                next_hand_keypoints = []
                
                for hand in data['hand_detections']:
                    hand_box = next(iter(hand.values()))
                    hand_keypoints = hand['keypoints'][0]
                    current_hand_keypoints.extend([[x + hand_box[0] - x_min, y + hand_box[1] - y_min] for x, y in hand_keypoints])
                
                for hand in next_data['hand_detections']:
                    hand_box = next(iter(hand.values()))
                    hand_keypoints = hand['keypoints'][0]
                    next_hand_keypoints.extend([[x + hand_box[0] - x_min, y + hand_box[1] - y_min] for x, y in hand_keypoints])
                
                # Draw motion arrows for hands, matching keypoints by index
                min_keypoints = min(len(current_hand_keypoints), len(next_hand_keypoints))
                hands_motion_map = create_motion_intensity_map(image_with_motion, 
                                   current_hand_keypoints[:min_keypoints], 
                                   next_hand_keypoints[:min_keypoints], (0, 255, 0), sigma=15, magnitude_scale=15)
                
                image_with_motion = combine_motion_maps(image_with_motion, body_motion_map, face_motion_map, hands_motion_map)
                
            # Save the results
#            sys.exit(0)
                
#            if aug_idx > 0:  # Применяем аугментацию ко всем кроме оригинальной последовательности
#                augmented = self.aug_transform(image=image)
#                images.append(augmented['image'])
#            else:
#                augmented = self.notaug_transform(image=image)
#                images.append(augmented['image'])

            images_to_check = [image_original, image_white, image_with_data, image_with_motion]
            resized_images = check_and_resize_images(images_to_check)

            image_original, image_white, image_with_data, image_with_motion = resized_images

            name, ext = os.path.splitext(frame)
            cv2.imwrite(f"prepared4ch\\{name}_1{ext}", image_original)
            cv2.imwrite(f"prepared4ch\\{name}_2{ext}", image_white)
            cv2.imwrite(f"prepared4ch\\{name}_3{ext}", image_with_data)
            cv2.imwrite(f"prepared4ch\\{name}_4{ext}", image_with_motion)

            image_original = cv2.resize(image_original, (512, 512), interpolation=cv2.INTER_AREA)
            image_white = cv2.resize(image_white, (512, 512), interpolation=cv2.INTER_AREA)
            image_with_data = cv2.resize(image_with_data, (512, 512), interpolation=cv2.INTER_AREA)
            image_with_motion = cv2.resize(image_with_motion, (512, 512), interpolation=cv2.INTER_AREA)
            
            multi_channel_image = np.stack([
#                image_original,
#                image_white,
                image_with_data,
                image_with_motion
            ], axis=-1)

            # Преобразуем в тензор PyTorch
            # Меняем оси для соответствия формату PyTorch: (channels, height, width)
            multi_channel_tensor = torch.from_numpy(multi_channel_image).permute(3, 2, 0, 1)

            images_seq.append(multi_channel_tensor)
            
        image_sequence = torch.stack(images_seq)
        
        label = self.class_to_idx[class_name]
        
        return image_sequence, label

def stratified_train_test_split(sequences, test_size=0.2, random_state=None):
    # Группируем последовательности по классам
    class_sequences = defaultdict(list)
    for seq in sequences:
        class_name = seq[0]
        class_sequences[class_name].append(seq)
    
    train_sequences = []
    val_sequences = []
    
    # Для каждого класса выполняем стратифицированное разделение
    for class_name, class_seqs in class_sequences.items():
        if len(class_seqs) == 1:
            # Если у класса только одна последовательность, добавляем её в обучающую выборку
            train_sequences.extend(class_seqs)
        else:
            train_seqs, val_seqs = train_test_split(class_seqs, test_size=test_size, random_state=random_state)
            train_sequences.extend(train_seqs)
            val_sequences.extend(val_seqs)
    
    return train_sequences, val_sequences
        
sequence_length = 22
num_epochs = 25

print("Starting data preparation #1")
dataset = VideoSequenceDataset(
    root_dir="c:\\Users\\Profi\\Downloads\\ContourletCNN\\data",
    sequence_length=sequence_length,
    augmentations_per_sample=0
)

# Получаем все последовательности
all_sequences = dataset.video_sequences

train_sequences, val_sequences = stratified_train_test_split(all_sequences, test_size=0.2, random_state=42)

print("Starting data preparation #2")
train_dataset = VideoSequenceDataset(
    root_dir="c:\\Users\\Profi\\Downloads\\ContourletCNN\\data",
    sequence_length=sequence_length,
    augmentations_per_sample=2
)
train_dataset.video_sequences = train_sequences

print("Starting data preparation #3")
val_dataset = VideoSequenceDataset(
    root_dir="c:\\Users\\Profi\\Downloads\\ContourletCNN\\data",
    sequence_length=sequence_length,
    augmentations_per_sample=0
)
val_dataset.video_sequences = val_sequences

vae, controlnet, text_encoder, tokenizer = initialize_models()

def preprocess_and_save_datasets(datasets, vae, controlnet, text_encoder, tokenizer, names):
    for dataset, name in zip(datasets, names):
        if not os.path.exists(f'preprocessed_{name}_dataset.h5'):
            preprocess_and_save_dataset(dataset, vae, controlnet, text_encoder, tokenizer, f'preprocessed_{name}_dataset.h5')
        print(f"Preprocessed and saved {name} dataset")

print("Starting data preparation #4")
preprocess_and_save_datasets([train_dataset, val_dataset], vae, controlnet, text_encoder, tokenizer, ['train', 'val'])

train_data = HDF5Dataset('preprocessed_train_dataset.h5')
val_data = HDF5Dataset('preprocessed_val_dataset.h5')

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

# Функция для вычисления точности
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

torch.autograd.set_detect_anomaly(True)

try:
    best_loss = 0
    model = CNNBiLSTMWithAttention(num_classes=42, sequence_length=sequence_length, lstm_hidden_size=1024).to(device)
    criterion = nn.CrossEntropyLoss().to(dtype=DTYPE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    current_lr = 0.001
    
    # Цикл обучения
    step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for embeddings, labels in train_pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)
            current_shape = embeddings.size()
            embeddings = embeddings.reshape(current_shape[0], sequence_length, -1, 512, 512)
            embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            try:
                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_accuracy = compute_accuracy(outputs, labels)
                correct_predictions += batch_accuracy * labels.size(0)
                total_samples += labels.size(0)
                
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print(loss.item())
                    logging.error(f"Step {step}: Loss is NaN or Inf")
                    raise ValueError("Training diverged")
                
                step += 1
                
                train_pbar.set_postfix({
                    'loss': f"{train_loss / (step % len(train_loader) or len(train_loader)):.4f}",
                    'acc': f"{correct_predictions / total_samples:.4f}"
                })
                
            except RuntimeError as e:
                logging.error(f"Error during training: {e}")
                logging.error(f"Input shape: {embeddings.shape}")
                logging.error(f"Input dtype: {embeddings.dtype}")
                raise

        train_loss /= len(train_loader)
        train_accuracy = correct_predictions / total_samples
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for embeddings, labels in val_pbar:
                embeddings, labels = embeddings.to(device), labels.to(device)
                current_shape = embeddings.size()
                embeddings = embeddings.reshape(current_shape[0], sequence_length, -1, 512, 512)
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                batch_accuracy = compute_accuracy(outputs, labels)
                correct_predictions += batch_accuracy * labels.size(0)
                total_samples += labels.size(0)
                
                val_pbar.set_postfix({
                    'loss': f"{val_loss / (val_pbar.n + 1):.4f}",
                    'acc': f"{correct_predictions / total_samples:.4f}"
                })
        
        val_loss /= len(val_loader)
        val_accuracy = correct_predictions / total_samples
        
        if val_accuracy > best_loss:
            best_loss = val_accuracy
            torch.save(model.state_dict(), 'bilstm_contourlet.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Learning Rate: {current_lr:.4f}")
    
    torch.save(model.state_dict(), 'bilstm_contourlet_final.pth')
    print(f"Best val acc: {best_loss}")

except Exception as e:
    print(f"An error occurred: {e}")
    # Дополнительная информация о состоянии GPU
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("CUDA not available")