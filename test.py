# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np
import os

# Tải mô hình phát hiện khuôn mặt và trích xuất đặc trưng
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Tên file chứa cơ sở dữ liệu khuôn mặt
DB_FILE = "face_db.npz"
embeddings = []
labels = []

# Load cơ sở dữ liệu nếu có
if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# Hàm trích xuất đặc trưng khuôn mặt
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
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

# Đăng ký khuôn mặt
def register_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("Khong phat hien khuon mat.")
        return

    face = faces[0]
    landmarks = sp(gray, face)
    embedding = face_encoder.compute_face_descriptor(frame, landmarks)
    embedding = np.array(embedding)

    for reg_emb in embeddings:
        _, matched = compare_embeddings(reg_emb, embedding)
        if matched:
            print("Khuon mat da ton tai.")
            return

    name = input("Nhap ten nguoi dung: ").strip()
    if name in labels:
        print("Nguoi dung da ton tai.")
        return

    embeddings.append(embedding)
    labels.append(name)
    save_db()
    print(f"Dang ky thanh cong cho {name}")

# Xác thực khuôn mặt và hiển thị kết quả
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

        # Vẽ khung và hiển thị tên
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Chế độ đăng ký
def run_register_mode():
    cap = cv2.VideoCapture(0)
    print("Chế độ DANG KY. Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        register_face(frame)
        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chế độ xác thực
def run_verification_mode():
    cap = cv2.VideoCapture(0)
    print("Chế độ XAC THUC. Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        verify_faces_on_frame(frame)
        cv2.imshow("Verify Face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Menu chính
def main():
    print("1: Dang ky khuon mat")
    print("2: Xac thuc khuon mat")
    choice = input("Chon che do (1 hoac 2): ").strip()

    if choice == '1':
        run_register_mode()
    elif choice == '2':
        run_verification_mode()
    else:
        print("Lua chon khong hop le!")

if __name__ == "__main__":
    main()
