import csv
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
from transliterate import translit
from ultralytics import YOLO
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')  # или другая версия модели
model.to(device)

def read_annotations(csv_file):
    annotations = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            annotations[row['attachment_id']] = {
                'text': row['text'],
                'height': int(row['height']),
                'width': int(row['width']),
                'length': float(row['length'])
            }
    return annotations

def process_mediapipe_data(json_file, annotations):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = {}
    for video_id, frames in data.items():
        if video_id in annotations:
            processed_data[video_id] = frames
    
    return processed_data

def extract_hand_image(frame, landmarks, height, width, padding=0.01, output_size=(640, 640)):
    hand_image_resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
    return hand_image_resized
                
    # Изменяем размер входного изображения до размера, ожидаемого YOLOv8
    input_size = (640, 640)  # Стандартный размер входа для YOLOv8
    frame_resized = cv2.resize(frame, input_size)
    
    # Преобразуем numpy array в PyTorch тензор и перемещаем на GPU, если доступно
    frame_torch = torch.from_numpy(frame_resized).float().to(device)
    frame_torch = frame_torch / 255.0
    # Изменяем форму тензора для соответствия ожиданиям модели (BCHW)
    frame_torch = frame_torch.permute(2, 0, 1).unsqueeze(0)
    
    results = model(frame_torch, verbose=False)  # Запуск детекции YOLOv8
    
    person_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls == 0:  # 0 соответствует классу "person" в COCO
                person_detected = True
                # Преобразуем координаты обратно к исходному размеру
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = x1 * width / input_size[0], x2 * width / input_size[0]
                y1, y2 = y1 * height / input_size[1], y2 * height / input_size[1]
                
                cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
                hand_image_resized = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_AREA)
                return hand_image_resized

    if not person_detected:
        print("Человек не обнаружен на изображении")
        return None

def safe_label(label):
    return translit(label, 'ru', reversed=True)
    
def process_video(video_path, video_id, frames_data, annotations, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    height = 640
    width = 640
    label = annotations[video_id]['text']
    
    if len(label) > 0:
        safed_label = safe_label(label)
        
        output_subdir = os.path.join(output_dir, safed_label)
        os.makedirs(output_subdir, exist_ok=True)

        for frame_index, frame_data in enumerate(tqdm(frames_data, desc=f"Processing {video_id}")):
            ret, frame = cap.read()
            if not ret:
                break
       
            # Обрабатываем каждую руку в данном кадре
            for hand_landmarks in frame_data['hand 1']:
                hand_image = extract_hand_image(frame, hand_landmarks, height, width)
                if hand_image is not None:
                    output_path = os.path.join(output_subdir, f"{video_id}_frame_{frame_index:03d}_hand_1.jpg")
                    cv2.imwrite(output_path, hand_image)

    cap.release()

def main():
    annotations_file = 'annotations_test.csv'
    mediapipe_file = 'slovo_mediapipe.json'
    videos_dir = 'test'
    output_dir = 'output'

    annotations = read_annotations(annotations_file)
    processed_data = process_mediapipe_data(mediapipe_file, annotations)

    for video_id, frames_data in processed_data.items():
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        process_video(video_path, video_id, frames_data, annotations, output_dir)

if __name__ == "__main__":
    main()