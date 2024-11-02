from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  # Import Base từ models.py

# Địa chỉ của cơ sở dữ liệu
DATABASE_URL = 'sqlite:///users.db'

# Tạo engine cho cơ sở dữ liệu
engine = create_engine(DATABASE_URL)

# Tạo tất cả các bảng trong cơ sở dữ liệu
Base.metadata.create_all(engine)

# Tạo một session factory
Session = sessionmaker(bind=engine)

# Hàm khởi tạo và trả về một session mới
def get_session():
    session = Session()
    return session
