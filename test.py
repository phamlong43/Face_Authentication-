import dlib
import cv2
import numpy as np
import os
import threading

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

        # Draw box and name
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        text = f"{matched_name} ({max_score:.2f}%)"
        cv2.putText(frame, text, (face.left(), face.top() - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Tạo các biến trạng thái đăng ký
is_registering = False
input_name = ""
register_thread_running = False

def handle_key_input(key):
    global input_name, is_registering, register_thread_running

    if key == 13:  # Enter
        if input_name and not register_thread_running:
            print(f"🔧 Đăng ký: {input_name}")
            register_thread_running = True
            threading.Thread(target=register_worker, args=(last_frame.copy(), input_name), daemon=True).start()
            is_registering = False
            input_name = ""
    elif key == 8:  # Backspace
        input_name = input_name[:-1]
    elif 32 <= key <= 126:  # ASCII printable
        input_name += chr(key)

def register_worker(frame, name):
    global register_thread_running
    register_face(frame, name)
    register_thread_running = False

# Main
last_frame = None
def main():
    global is_registering, input_name, last_frame

    cap = cv2.VideoCapture(0)
    print("Nhấn 'r' để đăng ký, 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()

        verify_faces_on_frame(frame)

        # Giao diện nhập tên
        if is_registering:
            cv2.putText(frame, "Nhap ten: " + input_name, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "'r': Dang ky | 'q': Thoat", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if is_registering:
            handle_key_input(key)
        else:
            if key == ord('r'):
                is_registering = True
                input_name = ""
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

