# -*- coding: utf-8 -*-
import dlib
import cv2
import numpy as np
import os

# Tai model phat hien khuon mat va dac trung khuon mat
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# File luu co so du lieu khuon mat
DB_FILE = "face_db.npz"
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# Lay embedding tu anh
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    embs = []
    for face in faces:
        landmarks = sp(gray, face)
        embedding = face_encoder.compute_face_descriptor(image, landmarks)
        embs.append(np.array(embedding))
    return embs

# So sanh 2 embedding
def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist, dist < 0.6

# Luu csdl
def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

# Dang ky khuon mat
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

    name = input("Nhap ten nguoi dung: ").strip()
    if name in labels:
        return

    embeddings.append(embedding)
    labels.append(name)
    save_db()

# Xac thuc khuon mat va hien thi ket qua tren khung hinh
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

        # Ve bounding box va ten
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Main
def main():
    cap = cv2.VideoCapture(0)
    mode = "idle"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Xu ly theo mode
        if mode == "register":
            register_face(frame)
            mode = "idle"
        else:
            verify_faces_on_frame(frame)

        cv2.putText(frame, "'r': Dang ky | 'q': Thoat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            mode = "register"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
