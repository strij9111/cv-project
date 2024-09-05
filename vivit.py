import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import torch.nn as nn
import random
import sys
import os
from transformers import ViTImageProcessor, ViTForImageClassification, pipeline
import logging
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import h5py
import torchvision.models as models
from collections import defaultdict

# Device settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 
DTYPE = torch.float16

logging.basicConfig(level=logging.DEBUG)

# Создание модели
model = models.resnext50_32x4d(weights=None)
model.to(device)
# Замена выходного слоя
model.fc = torch.nn.Identity()

# Загрузка checkpoint
checkpoint = torch.load('ResNext50.pth')
pretrained_dict = checkpoint['MODEL_STATE']

# Исключение слоёв fc из загружаемых параметров
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}

# Загрузка обновлённого словаря состояний в модель
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Загрузка модели и процессора
model_name = "dima806/hand_gestures_image_detection"
processor = ViTImageProcessor.from_pretrained(model_name)
model_vit = ViTForImageClassification.from_pretrained(model_name, output_hidden_states=True)
model_vit.to(device)


def split_and_get_embeddings_stack(images):
    device = images.device
    batch_size, channels, height, width = images.shape
    
    # Разделение изображений на левую и правую половины
    left_images = images[:, :, :, :width//2]
    right_images = images[:, :, :, width//2:]

    def get_embeddings_resnet(imgs):
        imgs = imgs.to(device)
    
        return model(imgs)
        
    def get_embeddings(imgs):
        imgs = imgs.to(device)
#        imgs = imgs / 255.0
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_vit(**inputs)
        embeddings = outputs.hidden_states[-1]
        mean_embeddings = torch.mean(embeddings, dim=1)
    
        return mean_embeddings
    
    # Получение эмбеддингов для левой и правой половин
    all_img = get_embeddings(images)
    all_img_resnet = get_embeddings_resnet(images)
    left_embeddings = get_embeddings(left_images)
    right_embeddings = get_embeddings(right_images)
    
    left_embeddings_resnet = get_embeddings_resnet(left_images)
    right_embeddings_resnet = get_embeddings_resnet(right_images)

    left_chunk1 = left_embeddings_resnet[:, :768]

    # Второй чанк
    left_chunk2 = left_embeddings_resnet[:, 768:768*2]

    # Третий чанк (оставшиеся элементы)
    left_chunk3 = left_embeddings_resnet[:, 768*2:]

    # Определяем, сколько нужно добавить нулей
    padding_size = 768 - left_chunk3.shape[1]

    # Если padding_size больше 0, дополняем нулями
    if padding_size > 0:
        left_chunk3 = torch.cat([left_chunk3, torch.zeros((batch_size, padding_size)).to(device)], dim=1)

    right_chunk1 = right_embeddings_resnet[:, :768]

    # Второй чанк
    right_chunk2 = right_embeddings_resnet[:, 768:768*2]

    # Третий чанк (оставшиеся элементы)
    right_chunk3 = right_embeddings_resnet[:, 768*2:]

    # Определяем, сколько нужно добавить нулей
    padding_size = 768 - right_chunk3.shape[1]

    # Если padding_size больше 0, дополняем нулями
    if padding_size > 0:
        right_chunk3 = torch.cat([right_chunk3, torch.zeros((batch_size, padding_size)).to(device)], dim=1)

    all_chunk1 = all_img_resnet[:, :768]

    # Второй чанк
    all_chunk2 = all_img_resnet[:, 768:768*2]

    # Третий чанк (оставшиеся элементы)
    all_chunk3 = all_img_resnet[:, 768*2:]

    # Определяем, сколько нужно добавить нулей
    padding_size = 768 - all_chunk3.shape[1]

    # Если padding_size больше 0, дополняем нулями
    if padding_size > 0:
        all_chunk3 = torch.cat([all_chunk3, torch.zeros((batch_size, padding_size)).to(device)], dim=1)
       
    # Создаем список тензоров, каждый из которых содержит левый и правый эмбеддинги для одного изображения
    embeddings_stack = torch.stack([left_embeddings, right_embeddings, left_chunk1, left_chunk2, left_chunk3, right_chunk1, right_chunk2, right_chunk3, all_img, all_chunk1, all_chunk2, all_chunk3], dim=1)

    return embeddings_stack
    
def get_embeddings(images):
    return split_and_get_embeddings_stack(images)

def preprocess_and_save_dataset(dataset, output_file, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Получаем размерность эмбеддингов из первого батча
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        sample_sequences, _ = sample_batch
        sample_frame = sample_sequences[:, 0, :, :, :].to(device, dtype=torch.float32)
        sample_embedding = get_embeddings(sample_frame)
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
                    embedding = get_embeddings(frame)
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, bidirectional=True, batch_first=True, dropout=0.3)
        self.residual = nn.Linear(input_size, hidden_size * 2)
        
    def forward(self, x):
        # x shape: [batch_size * num_channels, sequence_length, input_size]
        residual = self.residual(x)
        out, _ = self.lstm(x)
        return out + residual
        # output shape: [batch_size * num_channels, sequence_length, hidden_size * 2]

class SequenceAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SequenceAttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size * num_channels, sequence_length, hidden_size]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights
        # context_vector shape: [batch_size * num_channels, hidden_size]
        # attention_weights shape: [batch_size * num_channels, sequence_length, 1]

class ChannelAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_channels):
        super(ChannelAttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, channel_vectors):
        # channel_vectors shape: [batch_size, num_channels, hidden_size]
        attention_weights = F.softmax(self.attention(channel_vectors).squeeze(-1), dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * channel_vectors, dim=1)
        return context_vector, attention_weights
        # context_vector shape: [batch_size, hidden_size]
        # attention_weights shape: [batch_size, num_channels]

class CNNBiLSTMWithAttention(nn.Module):
    def __init__(self, num_classes, input_size=768, sequence_length=32, num_channels=12, lstm_hidden_size=1024):
        super(CNNBiLSTMWithAttention, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size
        
        self.dropout1 = nn.Dropout(0.3)
        
        self.residual_lstm = ResidualLSTM(input_size, lstm_hidden_size)
        
        self.sequence_attention = SequenceAttentionLayer(lstm_hidden_size * 2)
        self.channel_attention = ChannelAttentionLayer(lstm_hidden_size * 2, num_channels)
        
        self.ln = nn.LayerNorm(lstm_hidden_size * 2)
        self.dropout2 = nn.Dropout(0.3)
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
        # x shape: [batch_size, num_channels, sequence_length, input_size]
        # e.g., [4, 12, 32, 768]
        batch_size, num_channels, seq_len, features = x.size()
        
        # Reshape to [batch_size * num_channels, sequence_length, input_size]
        x = x.view(batch_size * num_channels, seq_len, features)
        
        lstm_out = self.residual_lstm(x)
        # lstm_out shape: [batch_size * num_channels, sequence_length, lstm_hidden_size * 2]
        
        sequence_context, sequence_attention = self.sequence_attention(lstm_out)
        # sequence_context shape: [batch_size * num_channels, lstm_hidden_size * 2]
        
        # Reshape sequence_context back to [batch_size, num_channels, lstm_hidden_size * 2]
        channel_vectors = sequence_context.view(batch_size, num_channels, -1)
        
        # Apply channel attention
        context_vector, channel_attention = self.channel_attention(channel_vectors)
        # context_vector shape: [batch_size, lstm_hidden_size * 2]
        
        context_vector = self.ln(context_vector)
        context_vector = self.dropout2(context_vector)
        
        output = self.fc(context_vector)
        # output shape: [batch_size, num_classes]
        
        return output, channel_attention, sequence_attention

class VideoSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=32, augmentations_per_sample=3):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.augmentations_per_sample = augmentations_per_sample
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.video_sequences = self._load_sequences()

        self.aug_transform = A.Compose([
            A.HorizontalFlip(p=1.0),
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
            A.Resize(640, 640),
            ToTensorV2(),
        ])

        self.notaug_transform = A.Compose([
            A.Resize(640, 640),
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
        try:
            for frame in sequence:
                img_path = os.path.join(self.root_dir, class_name, frame)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if aug_idx > 0:  # Применяем аугментацию ко всем кроме оригинальной последовательности
                    augmented = self.aug_transform(image=image)
                    images.append(augmented['image'])
                else:
                    augmented = self.notaug_transform(image=image)
                    images.append(augmented['image'])
        except:
            print(img_path)
            sys.exit(0)
            
        image_sequence = torch.stack(images)
        
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
            # Разделяем последовательности класса на обучающую и валидационную выборки
            train_seqs, val_seqs = train_test_split(class_seqs, test_size=test_size, random_state=random_state)
            train_sequences.extend(train_seqs)
            val_sequences.extend(val_seqs)
    
    return train_sequences, val_sequences
        
sequence_length = 32
num_epochs = 100

dataset = VideoSequenceDataset(
    root_dir="data",
    sequence_length=sequence_length,
    augmentations_per_sample=0
)

# Получаем все последовательности
all_sequences = dataset.video_sequences

train_sequences, val_sequences = stratified_train_test_split(all_sequences, test_size=0.2, random_state=42)

train_dataset = VideoSequenceDataset(
    root_dir="data",
    sequence_length=sequence_length,
    augmentations_per_sample=0
)
train_dataset.video_sequences = train_sequences

val_dataset = VideoSequenceDataset(
    root_dir="data",
    sequence_length=sequence_length,
    augmentations_per_sample=0
)
val_dataset.video_sequences = val_sequences


def preprocess_and_save_datasets(datasets, names):
    for dataset, name in zip(datasets, names):
        if not os.path.exists(f'preprocessed_{name}_dataset.h5'):
            preprocess_and_save_dataset(dataset, f'preprocessed_{name}_dataset.h5')
        print(f"Preprocessed and saved {name} dataset")

preprocess_and_save_datasets([train_dataset, val_dataset], ['train', 'val'])

train_data = HDF5Dataset('preprocessed_train_dataset.h5')
val_data = HDF5Dataset('preprocessed_val_dataset.h5')

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Функция для вычисления точности
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

torch.autograd.set_detect_anomaly(True)

def compute_attention_stats(attention_weights):
    # Находим индексы максимального и минимального значений для каждого элемента в батче
    max_indices = torch.argmax(attention_weights, dim=1)
    min_indices = torch.argmin(attention_weights, dim=1)
    
    # Получаем значения
    max_values = torch.gather(attention_weights, 1, max_indices.unsqueeze(1)).squeeze(1)
    min_values = torch.gather(attention_weights, 1, min_indices.unsqueeze(1)).squeeze(1)
    
    return {
        'max_values': max_values,
        'min_values': min_values,
        'max_indices': max_indices,
        'min_indices': min_indices
    }
        
        

best_acc = 0
model = CNNBiLSTMWithAttention(num_classes=986, sequence_length=sequence_length, lstm_hidden_size=512).to(device)
criterion = nn.CrossEntropyLoss().to(dtype=DTYPE)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
#    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader))
# Цикл обучения
step = 0
attention_stats = []
attention_stats_seq = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for embeddings, labels in train_pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        try:
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                print(f"NaN or Inf detected in input data at step {step}")
                continue
            optimizer.zero_grad()
            outputs, _, _ = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
#                scheduler.step()
            
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
    
    current_lr = optimizer.param_groups[0]['lr']

    # Валидация
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for embeddings, labels in val_pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            outputs, sequence_attention, channel_attention = model(embeddings)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            batch_accuracy = compute_accuracy(outputs, labels)
            correct_predictions += batch_accuracy * labels.size(0)
            total_samples += labels.size(0)
            
            attention_stats.append(compute_attention_stats(channel_attention))
            attention_stats_seq.append(compute_attention_stats(sequence_attention))
            
            val_pbar.set_postfix({
                'loss': f"{val_loss / (val_pbar.n + 1):.4f}",
                'acc': f"{correct_predictions / total_samples:.4f}"
            })

        avg_max_value = torch.mean(torch.cat([stats['max_values'] for stats in attention_stats]))
        avg_min_value = torch.mean(torch.cat([stats['min_values'] for stats in attention_stats]))
        avg_max_index = torch.mean(torch.cat([stats['max_indices'].float() for stats in attention_stats]))
        avg_min_index = torch.mean(torch.cat([stats['min_indices'].float() for stats in attention_stats]))

        print(f"Average most attended channel: value {avg_max_value:.6f} at index {avg_max_index:.2f}")
        
        avg_max_value = torch.mean(torch.cat([stats['max_values'] for stats in attention_stats_seq]))
        avg_min_value = torch.mean(torch.cat([stats['min_values'] for stats in attention_stats_seq]))
        avg_max_index = torch.mean(torch.cat([stats['max_indices'].float() for stats in attention_stats_seq]))
        avg_min_index = torch.mean(torch.cat([stats['min_indices'].float() for stats in attention_stats_seq]))

        print(f"Average most attended sequence: value {avg_max_value:.6f} at index {avg_max_index:.2f}")

    
    val_loss /= len(val_loader)
    val_accuracy = correct_predictions / total_samples
    
    if best_acc <= val_accuracy:
        best_acc = val_accuracy
        torch.save(model.state_dict(), 'bilstm_contourlet.pth')
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Learning Rate: {current_lr:.4f}")

torch.save(model.state_dict(), 'bilstm_contourlet_final.pth')
print(f"Best val acc: {best_acc}")

avg_max_value = torch.mean(torch.cat([stats['max_values'] for stats in attention_stats]))
avg_min_value = torch.mean(torch.cat([stats['min_values'] for stats in attention_stats]))
avg_max_index = torch.mean(torch.cat([stats['max_indices'].float() for stats in attention_stats]))
avg_min_index = torch.mean(torch.cat([stats['min_indices'].float() for stats in attention_stats]))

print(f"Average most attended channel: value {avg_max_value:.6f} at index {avg_max_index:.2f}")

avg_max_value = torch.mean(torch.cat([stats['max_values'] for stats in attention_stats_seq]))
avg_min_value = torch.mean(torch.cat([stats['min_values'] for stats in attention_stats_seq]))
avg_max_index = torch.mean(torch.cat([stats['max_indices'].float() for stats in attention_stats_seq]))
avg_min_index = torch.mean(torch.cat([stats['min_indices'].float() for stats in attention_stats_seq]))

print(f"Average most attended sequence: value {avg_max_value:.6f} at index {avg_max_index:.2f}")
    