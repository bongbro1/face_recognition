import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import shutil
import random
from flask_socketio import SocketIO

# Khởi tạo socketio
socketio = SocketIO()

def split_data(src_path):
    class_names = [d for d in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, d))]
    
    if class_names:
        for item in class_names:
            data_dir = os.path.join(src_path, item)  # Đường dẫn đến thư mục dữ liệu gốc
            train_dir = f'data/train/{item}'  # Thư mục để lưu dữ liệu train
            test_dir = f'data/test/{item}'  # Thư mục để lưu dữ liệu test

            # Xóa thư mục nếu đã tồn tại
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(test_dir, ignore_errors=True)

            # Tạo thư mục train và test
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Lấy danh sách tất cả các tệp ảnh .jpg
            files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]
            random.shuffle(files)

            # Chia dữ liệu vào thư mục train và test
            split_index = int(len(files) * 0.8)  # 80% cho train
            train_files = files[:split_index]
            test_files = files[split_index:]

            for file in train_files:
                shutil.copy(os.path.join(data_dir, file), train_dir)
            for file in test_files:
                shutil.copy(os.path.join(data_dir, file), test_dir)

        print("Dữ liệu đã được chia thành công vào thư mục train và test.")
    else:
        print("Dữ liệu tệp nguồn trống không thể chia.")
    
    return bool(class_names)

# Callback để phát tiến độ qua Socket.IO
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Tính toán tiến độ
        progress = int((epoch + 1) * 100 / self.total_epochs)
        socketio.emit('progress', {'percent': progress})  # Phát tiến độ sau mỗi epoch
        print(f'Epoch {epoch+1}/{self.total_epochs} - Progress: {progress}%')

def create_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Lớp tích chập đầu tiên với số bộ lọc giảm xuống
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))  # Dropout để giảm overfitting

    # Lớp tích chập thứ hai
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Lớp tích chập thứ ba
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Flatten để chuyển sang lớp fully connected
    model.add(Flatten())
    
    # Lớp fully connected
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Lớp đầu ra
    model.add(Dense(num_classes, activation='softmax'))

    # Compile mô hình với Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_data_dir, test_data_dir, input_shape=(150, 150, 3), epochs=50, socketio=None):
    if not split_data('data/src'):
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )

    # Sử dụng mô hình CNN được định nghĩa trước
    model = create_cnn_model(input_shape=input_shape, num_classes=train_generator.num_classes)

    # Khởi tạo các callback cho giảm learning rate và phát tiến độ
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    progress_callback = ProgressCallback(total_epochs=epochs)  # Callback để gửi tiến độ

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}:')
        # Giả lập tiến độ từng epoch
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=1,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size,
        )
        
        # Tính tiến độ và gửi về client
        progress = (epoch + 1) * 100 / epochs
        if socketio:  # Kiểm tra socketio có tồn tại không
            socketio.emit('progress', {'percent': progress})

    # Phát tín hiệu hoàn thành
    if socketio:
        socketio.emit('progress', {'percent': 100})

    loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Lưu mô hình
    model.save('cnn_model_improved.keras')
    socketio.emit('progress', {'percent': 100})