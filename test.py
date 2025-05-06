import dlib
import cv2
import numpy as np
import os
import threading
import queue

# Load model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

DB_FILE = "face_db.npz"
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist, dist < 0.4

def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

def register_face(image, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("⚠ Không phát hiện khuôn mặt.")
        return

    face = faces[0]
    landmarks = sp(gray, face)
    embedding = face_encoder.compute_face_descriptor(image, landmarks)
    embedding = np.array(embedding)

    for reg_emb in embeddings:
        _, matched = compare_embeddings(reg_emb, embedding)
        if matched:
            print("⚠ Khuôn mặt đã tồn tại.")
            return

    embeddings.append(embedding)
    labels.append(name)
    save_db()
    print(f"✅ Đăng ký thành công: {name}")

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

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Luồng nhập tên từ terminal
def input_thread(input_queue):
    while True:
        name = input("Nhập tên để đăng ký (hoặc để trống để hủy): ").strip()
        input_queue.put(name)

def main():
    cap = cv2.VideoCapture(0)
    input_queue = queue.Queue()
    last_frame = None

    # Bắt đầu thread để nhập tên
    threading.Thread(target=input_thread, args=(input_queue,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        last_frame = frame.copy()
        verify_faces_on_frame(frame)

        cv2.putText(frame, "'r': Đăng ký | 'q': Thoát", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("📸 Đang chụp khuôn mặt...")
            print("💬 Nhập tên ở terminal:")
            # Chờ tên nhập từ thread
            while input_queue.empty():
                cv2.waitKey(1)
            name = input_queue.get()
            if name:
                register_face(last_frame, name)
            else:
                print("❌ Đăng ký bị hủy.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
