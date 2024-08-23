import os
import cv2
from functools import lru_cache

from yolo import YOLO

import numpy as np
from flask import Flask, request, jsonify
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples

app = Flask(__name__)

# Инициализация моделей
yolo_model = None  # Инициализируйте вашу YOLO модель здесь
mmpose_model_hand = init_model('td-hm_res50_8xb32-210e_onehand10k-256x256.py', 
                          'res50_onehand10k_256x256-739c8639_20210330.pth', 
                          device='cuda')

mmpose_model_body = init_model('td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py', 
                          'td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth', 
                          device='cuda')

mmpose_model_face = init_model('rtmpose-m_8xb256-120e_face6-256x256.py', 
                          'rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth', 
                          device='cuda')

yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
                          
def run_yolo_detection(image):
    # Возвращает список обнаруженных рук (координаты bounding box)
    # Пример: [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]

    width, height, inference_time, results = yolo.inference(image)

    res = []
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        res.append((x, y, x+w, y+h))

    print(results)
    return res

@lru_cache(maxsize=None)
def process_image_halves(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Разделение изображения на левую и правую половины
    left_half = img[:, :width//2]
    right_half = img[:, width//2:]

    detections = run_yolo_detection(img)
    
    # Обработка левой половины
    left_detections = run_yolo_detection(left_half)
    
    # Обработка правой половины
    right_detections = run_yolo_detection(right_half)
    
    # Корректировка координат для правой половины
    right_detections = [(x + width//2, y, w + width//2, h) for x, y, w, h in right_detections]
    
    # Объединение результатов
    all_detections = left_detections + right_detections + detections
    
    return all_detections

@lru_cache(maxsize=None)
def process_face_image(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    
    # Выполнение inference
    batch_results = inference_topdown(mmpose_model_face, img)
    results = merge_data_samples(batch_results)   
    # Извлечение нужных данных из результатов
    pred_instances = results.pred_instances
    
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    bboxes = pred_instances.bboxes
    bbox_scores = pred_instances.bbox_scores
    
    # Преобразование в список для удобства сериализации
    keypoints_list = keypoints.tolist()
    keypoint_scores_list = keypoint_scores.tolist()
    bboxes_list = bboxes.tolist()
    bbox_scores_list = bbox_scores.tolist()
    
    return {
        'keypoints': keypoints_list,
        'keypoint_scores': keypoint_scores_list,
    }

@lru_cache(maxsize=None)    
def process_body_image(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    
    # Выполнение inference
    batch_results = inference_topdown(mmpose_model_body, img)
    results = merge_data_samples(batch_results)   
    # Извлечение нужных данных из результатов
    pred_instances = results.pred_instances
    
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    bboxes = pred_instances.bboxes
    bbox_scores = pred_instances.bbox_scores
    
    # Преобразование в список для удобства сериализации
    keypoints_list = keypoints.tolist()
    keypoint_scores_list = keypoint_scores.tolist()
    bboxes_list = bboxes.tolist()
    bbox_scores_list = bbox_scores.tolist()
    
    return {
        'keypoints': keypoints_list,
        'keypoint_scores': keypoint_scores_list,
    }

@lru_cache(maxsize=None)    
def process_hand_image(image_path):
    # Загрузка изображения
    img = cv2.imread(image_path)
    
    # Выполнение inference
    batch_results = inference_topdown(mmpose_model_hand, img)
    results = merge_data_samples(batch_results)
    # Извлечение нужных данных из результатов
    pred_instances = results.pred_instances
    
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    bboxes = pred_instances.bboxes
    bbox_scores = pred_instances.bbox_scores
    
    # Преобразование в список для удобства сериализации
    keypoints_list = keypoints.tolist()
    keypoint_scores_list = keypoint_scores.tolist()
    bboxes_list = bboxes.tolist()
    bbox_scores_list = bbox_scores.tolist()
    
    return {
        'keypoints': keypoints_list,
        'keypoint_scores': keypoint_scores_list,
    }

@app.route('/detect_hands', methods=['POST'])
def detect_hands():
    data = request.json
    if not data or 'file_path' not in data or 'file_name' not in data:
        return jsonify({'error': 'Missing file_path or file_name in request'}), 400
    
    file_path = data['file_path']
    file_name = data['file_name']
    
    full_path = os.path.join(file_path, file_name)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'File not found'}), 404
    
    body_data = process_body_image(full_path)
    face_data = process_face_image(full_path)
    # Шаг 1: Запуск YOLO для обнаружения рук
    hand_bboxes = process_image_halves(full_path)
    
    results = []
    for i, bbox in enumerate(hand_bboxes):
        # Вырезаем изображение руки
        x1, y1, x2, y2 = bbox
        hand_img = cv2.imread(full_path)[y1:y2, x1:x2]
        if hand_img is not None and hand_img.size > 0:
            hand_filename = f"hand_{i}.jpg"
            hand_filepath = os.path.join(file_path, hand_filename)
            cv2.imwrite(hand_filepath, hand_img)
            
            # Шаг 2: Обработка каждого изображения руки с помощью MMPose
            hand_results = process_hand_image(hand_filepath)
            
            results.append({
                f"hand_{i}": [x1, y1, x2, y2],
                'keypoints': hand_results['keypoints'],
                'keypoint_scores': hand_results['keypoint_scores']
            })
        
            # Удаление временного файла руки
            os.remove(hand_filepath)
    
    return jsonify({
        'body_detections': body_data,
        'face_detections': face_data,
        'hand_detections': results
    })

if __name__ == '__main__':
    app.run(debug=True)