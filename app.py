from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.utils import secure_filename
from database import get_session
from models import User, FaceRecord, generate_images, get_names_from_ids
from cnn_face_recognition import train_model
import numpy as np
import os, uuid, base64
import json, time
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)




data_path = 'data/src'
db_path = 'users.db'
if any(os.path.isdir(os.path.join(data_path, d)) for d in os.listdir(data_path)):
        class_names = get_names_from_ids(data_path, db_path)
else:
    class_names = []
model = load_model('cnn_model_improved.keras')

@app.route('/api/getUserCount', methods=['GET'])
def get_user_count():
    session = get_session()
    count = session.query(User).count()
    return jsonify({'count': count})

@app.route('/get_users_attendance', methods=['GET'])
def get_users_attendance():
    session = get_session()
    users = (
        session.query(FaceRecord)
        .order_by(FaceRecord.recorded_at.desc())
        .limit(20)
        .all()
    )
    users_info = [
        {
            'user_id': user.id,
            'recorded_at': user.recorded_at.strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_result': user.prediction_result,
            'prediction_percent': user.prediction_percent
        }
        for user in users
    ]
    return jsonify(users_info)

@app.route('/')
def index():
    return render_template('index.html')

# Lấy ngày hiện tại không tính thời gian
start_of_today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
end_of_today = start_of_today + timedelta(days=1)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Thay đổi kích thước hình ảnh
    img_array = image.img_to_array(img) / 255.0  # Chia pixel cho 255
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều cho batch

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)
    predicted_percent = predictions[0][predicted_index[0]] * 100

    session = get_session()
    

    # Kiểm tra xem bản ghi đã tồn tại cho user_id cụ thể trong ngày hôm nay chưa
    existing_record = (
        session.query(FaceRecord)
        .filter(FaceRecord.user_id == int(predicted_index[0]))
        .filter(FaceRecord.recorded_at >= start_of_today, FaceRecord.recorded_at < end_of_today)
        .order_by(FaceRecord.prediction_percent.desc())  # Sắp xếp theo tỷ lệ dự đoán giảm dần
        .first()  # Lấy bản ghi đầu tiên
    )
    flag = False
    # Nếu không có bản ghi nào hoặc nếu tỷ lệ dự đoán mới lớn hơn tỷ lệ tối đa
    if existing_record is None or round(predicted_percent, 2) > existing_record.prediction_percent:
        # Tạo đối tượng FaceRecord chỉ khi user_id chưa tồn tại hoặc có tỷ lệ mới cao hơn
        face_record = FaceRecord(
            user_id=int(predicted_index[0]),  # Gán user_id từ predicted_index
            prediction_result=class_names[f'{int(predicted_index[0]) + 1}'],
            prediction_percent=round(predicted_percent, 2)  # Làm tròn phần trăm
        )

        # Thêm vào session và lưu vào cơ sở dữ liệu
        session.add(face_record)
        session.commit()
        flag = True
    if flag and existing_record:
        session.delete(existing_record)  # Xóa bản ghi cũ
        session.commit()
    

    return predicted_index, predicted_percent

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files.get('image')

    if img_file and img_file.content_type.startswith('image/'):
        # Tạo tên tệp ngẫu nhiên
        random_filename = f"{uuid.uuid4()}.jpg"
        img_path = os.path.join('data/image/image_temp', random_filename)
        
        # Đảm bảo thư mục tồn tại
        if not os.path.exists('data/image/image_temp'):
            os.makedirs('data/image/image_temp')
        
        img_file.save(img_path)
        img_path = img_path.replace('\\', '/')

        # Dự đoán
        predicted_index, predicted_percent = predict_image(img_path)

        # Xóa tệp sau khi sử dụng
        os.remove(img_path)
        
        # Lấy tên lớp dự đoán
        result = class_names[f'{int(predicted_index[0])+1}']

        # Định dạng phần trăm dự đoán
        percent_result = f"{predicted_percent:.2f}".replace('.', ',') + '%'

        return jsonify({'predicted_class': result, 'prediction_percent': percent_result})
    else:
        return jsonify({'error': 'Invalid image file'}), 400


#========================================================================================

app.config['UPLOAD_FOLDER'] = 'data/image/avatar' 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Kiểm tra định dạng file hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/add_user', methods=['GET'])
def show_add_user_form():
    return render_template('add_user.html')  # Hiển thị form thêm người dùng

@app.route('/add_user', methods=['POST'])
def add_user():
    session = get_session()
    data = request.form

    name = data.get('name')
    email = data.get('email')
    face_encoding_data = data.get('face_encoding')

    # Tạo mới một user
    new_user = User(
        name=name,
        email=email,
        face_encoding=None
    )
    session.add(new_user)
    session.commit()

    # Đặt tên file là id của user vừa tạo
    user_id = new_user.id
    
    # Đường dẫn lưu ảnh trong thư mục 'data/src/user_<user_id>'
    user_directory = os.path.join('data', 'src', f'user_{user_id}')
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(user_directory, exist_ok=True)
    if face_encoding_data:
        # Chuyển đổi chuỗi JSON về danh sách các ảnh đã chụp
        captured_images = json.loads(face_encoding_data)

        for index, img_data in enumerate(captured_images):
            # Tạo tên file cho mỗi ảnh
            filename = f"user_{user_id}_{index}.jpg"
            file_path = os.path.join(user_directory, filename)

            # Xử lý chuỗi base64
            img_data = img_data.split(",")[1]  # Bỏ phần header
            try:
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(img_data))
            except Exception as e:
                print(f"Lỗi khi lưu ảnh {filename}: {e}")

    user_directory = user_directory.replace('\\', '/')
    new_user.face_encoding = user_directory  # Lưu đường dẫn thư mục chứa ảnh
    session.commit()

    session.close()
    return jsonify({'message': 'Đã lưu đối tượng!'}), 201


@app.route('/train_model', methods=['GET'])
def train_model_route():
    # Khởi động hàm huấn luyện trong background
    socketio.start_background_task(train_model, 'data/train', 'data/test', socketio=socketio)
    return jsonify({'message': 'Started training model!'}), 202  # Trả về mã 202 để chỉ ra rằng quá trình đã được bắt đầu


# @app.route('/train_model', methods=['GET'])
# def train_model_route():
#     # Khởi động hàm huấn luyện trong background
#     socketio.start_background_task(train_model_app, 'data/train', 'data/test')
#     return jsonify({'message': 'Started training model!'}), 202  # Trả về mã 202 để chỉ ra rằng quá trình đã được bắt đầu

# def train_model_app(train_data_path, test_data_path):
#     total_epochs = 50  # Giả sử bạn có 50 epoch
#     for epoch in range(total_epochs):
#         time.sleep(5)  # Giả lập thời gian huấn luyện mỗi epoch

#         # Tính toán tiến độ
#         progress = (epoch + 1) * 100 / total_epochs
#         socketio.emit('progress', {'percent': progress})  # Gửi tiến độ về client

#     # Khi hoàn thành, phát ra tiến độ 100%
#     socketio.emit('progress', {'percent': 100})
#     print("Training completed.")


if __name__ == '__main__':
    socketio.run(app, debug=True)
