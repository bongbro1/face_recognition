from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, TIMESTAMP, DECIMAL, BLOB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.orm import sessionmaker
import cv2
import os, re, sqlite3
import numpy as np
import os
import shutil
import random

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False)
    created_at = Column(TIMESTAMP, default=lambda: datetime.utcnow())
    updated_at = Column(TIMESTAMP, default=lambda: datetime.utcnow(), onupdate=lambda: datetime.utcnow())
    face_encoding = Column(String(1000), nullable=True)

class FaceRecord(Base):
    __tablename__ = 'face_records'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    recorded_at = Column(TIMESTAMP, default=lambda: datetime.utcnow())
    prediction_result = Column(String(100)) # Tên của người đó
    prediction_percent = Column(DECIMAL(5, 2))

def adjust_brightness(image, factor):
    # Điều chỉnh độ sáng bằng cách nhân hệ số vào cường độ ảnh, sau đó cắt lại giá trị
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def resize_image(image, scale):
    # Phóng to hoặc thu nhỏ ảnh theo tỷ lệ
    height, width = image.shape[:2]
    return cv2.resize(image, (int(width * scale), int(height * scale)))

def generate_images(input_path, output_path, n):
    # Đọc ảnh
    image = cv2.imread(input_path)

    # Kiểm tra xem ảnh có được đọc thành công không
    if image is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
        return

    # Danh sách chứa các biến thể của ảnh
    transformations = [
        ('original', image),
        ('flip_horizontal', cv2.flip(image, 1)),
        ('flip_vertical', cv2.flip(image, 0))
    ]

    # Kiểm tra nếu n yêu cầu nhỏ hơn số biến thể sẵn có
    if n <= len(transformations):
        selected_transformations = transformations[:n]
    else:
        # Thêm biến thể xoay ngẫu nhiên
        angles = random.sample(range(-80, 81), min(n - len(transformations), 161))
        for angle in angles:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h))
            transformations.append((f'rotate_{angle}', rotated_image))

        # Thêm biến thể phóng to và thu nhỏ
        scales = [1.2, 0.8]  # Phóng to 20%, thu nhỏ 20%
        for scale in scales:
            resized_image = resize_image(image, scale)
            transformations.append((f'resize_{scale}', resized_image))

        # Thêm biến thể tăng và giảm độ sáng
        brightness_factors = [0.5, 1.5]  # Tăng sáng 50%, giảm sáng 50%
        for factor in brightness_factors:
            bright_image = adjust_brightness(image, factor)
            transformations.append((f'brightness_{factor}', bright_image))

        # Chọn đúng n ảnh biến thể đầu tiên
        selected_transformations = transformations[:n]

    # Tạo và lưu các ảnh biến thể
    for i, (name, img) in enumerate(selected_transformations):
        output_file = os.path.join(output_path, f'{name}_{i}.jpg')
        cv2.imwrite(output_file, img)

    print(f'Tạo thành công {len(selected_transformations)} ảnh.')

def get_names_from_ids(data_path, db_path):
    list_id = []
    pattern = re.compile(r'user_(\d+)')

    for d in os.listdir(data_path):
        dir_path = os.path.join(data_path, d)
        if os.path.isdir(dir_path):
            match = pattern.match(d)
            if match:
                list_id.append(match.group(1))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    user_dict = {}
    for user_id in list_id:
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if result:
            user_dict[user_id] = result[0]

    conn.close()
    return user_dict

# Tạo engine và session
engine = create_engine('sqlite:///users.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
