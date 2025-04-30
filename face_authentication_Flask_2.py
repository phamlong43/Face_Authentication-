import cv2
import dlib
import numpy as np
import os
import time
import sys

# Hiển thị thông tin phiên bản
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Dlib version: {dlib.__version__ if 'dlib' in sys.modules else 'Not imported'}")
print(f"Numpy version: {np.__version__}")

# Biến toàn cục
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Đường dẫn tới các file model
SHAPE_PREDICTOR_PATH = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
FACE_RECOG_MODEL_PATH = os.path.join(current_dir, 'dlib_face_recognition_resnet_model_v1.dat')
DB_FILE = os.path.join(current_dir, "face_db.npz")

# Kiểm tra các file model
if not os.path.exists(SHAPE_PREDICTOR_PATH):
    print(f"Lỗi: Không tìm thấy file {SHAPE_PREDICTOR_PATH}")
    print("Vui lòng tải file shape_predictor_68_face_landmarks.dat và đặt vào thư mục chạy chương trình")
else:
    print(f"Đã tìm thấy file {SHAPE_PREDICTOR_PATH}")

if not os.path.exists(FACE_RECOG_MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file {FACE_RECOG_MODEL_PATH}")
    print("Vui lòng tải file dlib_face_recognition_resnet_model_v1.dat và đặt vào thư mục chạy chương trình")
else:
    print(f"Đã tìm thấy file {FACE_RECOG_MODEL_PATH}")

# Tải model nhận diện khuôn mặt
try:
    print("Đang tải model...")
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
    print("Đã tải thành công các model nhận diện khuôn mặt")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    detector = None
    sp = None
    face_encoder = None

# Khởi tạo cơ sở dữ liệu khuôn mặt
embeddings = []
labels = []

# Kiểm tra xem cơ sở dữ liệu đã tồn tại không
if os.path.exists(DB_FILE):
    try:
        data = np.load(DB_FILE, allow_pickle=True)
        embeddings = list(data["embeddings"])
        labels = list(data["labels"])
        print(f"Đã tải cơ sở dữ liệu với {len(labels)} khuôn mặt")
    except Exception as e:
        print(f"Lỗi khi tải cơ sở dữ liệu: {e}")

# Hàm chuyển đổi dlib shape thành numpy array
def shape_to_np(shape, dtype="int"):
    """
    Chuyển đổi dlib shape thành numpy array
    """
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    """
    Tiền xử lý ảnh để tăng khả năng phát hiện khuôn mặt
    """
    # Kiểm tra ảnh
    if image is None or image.size == 0:
        print("Lỗi: Ảnh không hợp lệ")
        return None
        
    # Đảm bảo image là uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Tạo bản sao để tránh thay đổi ảnh gốc
    processed = image.copy()
    
    try:
        # Tách các kênh màu
        b, g, r = cv2.split(processed)
        # Cân bằng histogram cho từng kênh
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        # Ghép các kênh lại
        processed = cv2.merge((b, g, r))
    except Exception as e:
        print(f"Lỗi khi cân bằng histogram: {e}")
    
    try:
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
    except Exception as e:
        print(f"Lỗi khi áp dụng bộ lọc làm mịn: {e}")
    
    try:
        # Tăng độ tương phản
        alpha = 1.3  # Hệ số tương phản (>1 tăng, <1 giảm)
        beta = 10    # Hệ số độ sáng (>0 tăng, <0 giảm)
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
    except Exception as e:
        print(f"Lỗi khi điều chỉnh độ sáng và độ tương phản: {e}")
    
    return processed

# Hàm phát hiện khuôn mặt
def detect_faces(image, upsample_times=1, adjust_threshold=True):
    """
    Phát hiện khuôn mặt với các tham số khác nhau
    """
    if detector is None:
        return []
        
    # Đảm bảo image là BGR (định dạng OpenCV)
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Lỗi: Ảnh không phải định dạng BGR")
        return []
    
    # Chuyển sang RGB cho dlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Thử với các tham số khác nhau
    faces = []
    try:
        # Thử với upsample_times khác nhau
        for upsample in range(upsample_times, 0, -1):
            # Nếu adjust_threshold, thử với ngưỡng thấp hơn
            if adjust_threshold:
                # Điều chỉnh ngưỡng phát hiện (0 là mặc định, giảm để phát hiện nhiều khuôn mặt hơn)
                faces = detector(rgb_image, upsample, -0.2)
            else:
                faces = detector(rgb_image, upsample)
                
            if len(faces) > 0:
                break
    except Exception as e:
        print(f"Lỗi khi phát hiện khuôn mặt: {e}")
    
    return faces

# Hàm so sánh hai embedding
def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    # Ngưỡng 0.6 có thể điều chỉnh để thay đổi độ chính xác
    threshold = 0.6
    return dist, dist < threshold

# Hàm xác minh khuôn mặt trên frame
def verify_faces_on_frame(frame):
    """
    Xác minh khuôn mặt trên frame và trả về frame đã xử lý
    """
    if detector is None or sp is None or face_encoder is None:
        cv2.putText(frame, "Model chưa được tải", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame
    
    # Kiểm tra frame
    if frame is None or frame.size == 0:
        print("Lỗi: Frame không hợp lệ")
        return frame
    
    # Đảm bảo frame là uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
        
    # Kiểm tra nếu không có dữ liệu khuôn mặt
    if len(embeddings) == 0:
        cv2.putText(frame, "Chưa có dữ liệu khuôn mặt", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    # Tiền xử lý frame
    processed_frame = preprocess_image(frame)
    if processed_frame is None:
        return frame
    
    # Phát hiện khuôn mặt với các tham số cải tiến
    faces = detect_faces(processed_frame, upsample_times=1, adjust_threshold=True)
    
    # Debug: Hiển thị số khuôn mặt phát hiện được
    print(f"Số khuôn mặt phát hiện được: {len(faces)}")
    
    # Nếu không tìm thấy khuôn mặt, thử lại với frame gốc
    if len(faces) == 0:
        print("Thử phát hiện khuôn mặt với frame gốc...")
        faces = detect_faces(frame, upsample_times=1, adjust_threshold=True)
        print(f"Với frame gốc: Số khuôn mặt phát hiện được: {len(faces)}")

    if len(faces) == 0:
        cv2.putText(frame, "Không phát hiện khuôn mặt (thử di chuyển gần hơn)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    # Hiển thị đường viền nhận diện
    for face in faces:
        try:
            # Chuyển sang RGB cho face_encoder
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Lấy landmarks
            landmarks = sp(rgb_frame, face)
            
            # Tính toán embedding
            embedding = face_encoder.compute_face_descriptor(rgb_frame, landmarks)
            embedding = np.array(embedding)

            matched_name = "Unknown"
            max_score = 0.0

            for name, reg_emb in zip(labels, embeddings):
                dist, matched = compare_embeddings(reg_emb, embedding)
                score = max(0, 1 - dist) * 100
                if matched and score > max_score:
                    matched_name = name
                    max_score = score

            # Vẽ bounding box và tên lên ảnh gốc (BGR)
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            text = f"{matched_name} ({max_score:.2f}%)"
            cv2.putText(frame, text, (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
            # Vẽ landmarks để kiểm tra
            shape = shape_to_np(landmarks)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                
        except Exception as e:
            print(f"Lỗi khi xử lý khuôn mặt: {e}")
    
    return frame

# Hiển thị thông tin chỉ dẫn
def draw_info(frame):
    cv2.putText(frame, "ESC: Thoat", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"So khuon mat da dang ky: {len(labels)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

# Khởi tạo camera
def init_camera():
    """
    Khởi tạo camera với nhiều lựa chọn backend và xử lý lỗi
    """
    # Danh sách các backend camera để thử theo thứ tự ưu tiên
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "Video4Linux2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Default")
    ]
    
    # Thử nhiều camera index (0-3)
    camera_indexes = list(range(4))
    
    # Thử các cách kết hợp backend và index
    for idx in camera_indexes:
        # Cách 1: Thử trước không có backend chỉ định
        try:
            print(f"Thử kết nối camera index {idx} không chỉ định backend")
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                time.sleep(0.5)  # Đợi camera khởi động
                
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print(f"Kết nối thành công với camera index {idx}")
                    
                    # Chỉ thiết lập kích thước sau khi đã kết nối thành công
                    try:
                        # Lấy kích thước hiện tại để kiểm tra
                        current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        print(f"Kích thước camera hiện tại: {current_width}x{current_height}")
                        
                        # Chỉ thay đổi kích thước nếu cần thiết
                        if current_width != 640 or current_height != 480:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            print("Đã điều chỉnh kích thước camera")
                    except Exception as e:
                        print(f"Không thể thiết lập kích thước camera: {e}")
                    
                    return cap
                else:
                    print(f"Mở camera thành công nhưng không đọc được frame")
                    cap.release()
            else:
                print(f"Không thể mở camera với index {idx}")
        except Exception as e:
            print(f"Lỗi khi thử kết nối camera index {idx}: {e}")
        
        # Cách 2: Thử với các backend cụ thể
        for backend, backend_name in backends:
            try:
                print(f"Đang thử kết nối camera với backend {backend_name} và index {idx}")
                cap = cv2.VideoCapture(idx, backend)
                
                if cap.isOpened():
                    # Đợi camera khởi động
                    time.sleep(0.5)
                    
                    # Đọc vài frame để đảm bảo camera hoạt động
                    success = False
                    for i in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            print(f"Đọc thành công frame {i+1}/5")
                            success = True
                            break
                        time.sleep(0.1)
                    
                    if success:
                        print(f"Đã kết nối thành công với camera (backend: {backend_name}, index: {idx})")
                        print(f"Thông tin frame - Shape: {test_frame.shape}, Type: {test_frame.dtype}")
                        
                        # Chỉ thiết lập các tham số sau khi kết nối thành công
                        try:
                            # Lấy thông tin hiện tại
                            current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            current_fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            print(f"Cấu hình camera hiện tại: {current_width}x{current_height} @ {current_fps}fps")
                            
                            # Thiết lập cấu hình mới
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                        except Exception as e:
                            print(f"Không thể thiết lập tham số camera, nhưng vẫn tiếp tục: {e}")
                        
                        return cap
                    else:
                        print(f"Mở camera thành công nhưng không đọc được frame, thử cách khác")
                        cap.release()
                else:
                    print(f"Không thể mở camera với backend {backend_name} và index {idx}")
            except Exception as e:
                print(f"Lỗi khi thử kết nối với backend {backend_name} và index {idx}: {e}")
    
    print("Không thể kết nối với bất kỳ camera nào")
    return None

def main():
    print("Khởi động ứng dụng nhận diện khuôn mặt...")
    
    # Khởi tạo camera
    cap = init_camera()
    
    if cap is None:
        print("Không thể khởi tạo camera, thoát chương trình.")
        return
    
    print("Camera đã được mở thành công. Nhấn 'ESC' để thoát.")
    
    try:
        while True:
            # Đọc một frame từ camera
            ret, frame = cap.read()
            
            # Kiểm tra xem frame có được đọc đúng không
            if not ret or frame is None or frame.size == 0:
                print("Không thể nhận frame. Thoát...")
                break
            
            # Lật frame để hiển thị như gương
            frame = cv2.flip(frame, 1)
            
            # Nhận diện khuôn mặt
            if detector is not None and sp is not None and face_encoder is not None:
                frame = verify_faces_on_frame(frame)
            
            # Hiển thị thông tin
            frame = draw_info(frame)
            
            # Hiển thị frame
            cv2.imshow('Face Recognition', frame)
            
            # Nhấn phím ESC để thoát
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                print("Nhấn ESC, thoát chương trình")
                break
    
    except Exception as e:
        print(f"Lỗi trong vòng lặp chính: {e}")
    
    finally:
        # Giải phóng camera và đóng các cửa sổ
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Đã giải phóng tài nguyên và đóng chương trình")

if __name__ == "__main__":
    main()