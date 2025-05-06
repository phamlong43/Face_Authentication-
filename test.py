import dlib
import cv2
import numpy as np
import os
import threading
import queue

# Tạo khóa để bảo vệ các thao tác ghi vào cơ sở dữ liệu
db_lock = threading.Lock()

# Tạo hàng đợi để xử lý các khung hình tuần tự
frame_queue = queue.Queue()

# Tải model phát hiện khuôn mặt và đặc trưng khuôn mặt
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# File lưu cơ sở dữ liệu khuôn mặt
DB_FILE = "face_db.npz"
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# Lấy embedding từ ảnh
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    embs = []
    for face in faces:
        landmarks = sp(gray, face)
        embedding = face_encoder.compute_face_descriptor(image, landmarks)
        embs.append(np.array(embedding))
    return embs

# So sánh hai embedding
def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist, dist < 0.4

# Lưu cơ sở dữ liệu
def save_db():
    with db_lock:  # Đảm bảo chỉ có một luồng ghi vào cơ sở dữ liệu
        np.savez(DB_FILE, embeddings=embeddings, labels=labels)

# Đăng ký khuôn mặt
def register_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return

    face = faces[0]
    landmarks = sp(gray, face)
    embedding = face_encoder.compute_face_descriptor(frame, landmarks)
    embedding = np.array(embedding)

    for reg_emb in embeddings:
        _, matched = compare_embeddings(reg_emb, embedding)
        if matched:
            return

    name = input("Nhập tên người dùng: ").strip()
    if name in labels:
        return

    with db_lock:
        embeddings.append(embedding)
        labels.append(name)
        save_db()

# Xác thực khuôn mặt và hiển thị kết quả trên khung hình
def verify_faces_on_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = sp(gray, face)
        embedding = face_encoder.compute_face_descriptor(frame, landmarks)
        embedding = np.array(embedding)

        matched_name = "Unknown"
        max_score = 0.0

        for name, reg_emb in zip(labels, embeddings):
            dist, matched = compare_embeddings(reg_emb, embedding)
            score = max(0, 1 - dist) * 100
            if matched and score > max_score:
                matched_name = name
                max_score = score

        # Vẽ bounding box và tên
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Kiểm tra giả mạo bằng cách kiểm tra Landmark
def detect_spoofing_with_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return False

    face = faces[0]
    landmarks = sp(gray, face)

    # Tính toán khoảng cách giữa các điểm landmark và kiểm tra tính hợp lý
    points = [(p.x, p.y) for p in landmarks.parts()]
    dist_eye_nose = np.linalg.norm(np.array(points[36]) - np.array(points[30]))  # Khoảng cách giữa mắt trái và mũi
    dist_eye_mouth = np.linalg.norm(np.array(points[36]) - np.array(points[48]))  # Khoảng cách giữa mắt trái và miệng

    # Kiểm tra khoảng cách có hợp lý không
    if dist_eye_nose < 15 or dist_eye_mouth < 20:
        return True

    # Kiểm tra sự phân bố các điểm landmark
    mean_x = np.mean([point[0] for point in points])
    mean_y = np.mean([point[1] for point in points])

    variance_x = np.var([point[0] for point in points])
    variance_y = np.var([point[1] for point in points])

    if variance_x < 100 or variance_y < 100:
        return True

    return False

# Main
def main():
    cap = cv2.VideoCapture(0)
    mode = "idle"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kiểm tra giả mạo trước khi xử lý khuôn mặt
        if detect_spoofing_with_landmarks(frame):
            cv2.putText(frame, "Warning: Spoofing detected!", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Xử lý theo chế độ
            if mode == "register":
                frame_queue.put((register_face, frame))
                mode = "idle"
            else:
                frame_queue.put((verify_faces_on_frame, frame))

        cv2.putText(frame, "'r': Đăng ký | 'q': Thoát", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            mode = "register"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm xử lý công việc từ hàng đợi
def process_queue():
    while True:
        func, frame = frame_queue.get()
        if func:
            func(frame)
        frame_queue.task_done()

if __name__ == "__main__":
    # Bắt đầu luồng xử lý công việc
    threading.Thread(target=process_queue, daemon=True).start()
    main()
