import dlib
import cv2
import numpy as np
import os
import threading

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

# Global variables for the video capture frame
frame = None
frame_lock = threading.Lock()

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
    return dist, dist < 0.4

# Luu csdl
def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

# Dang ky khuon mat
def register_face():
    global frame
    while True:
        with frame_lock:
            if frame is None:
                continue  # Skip if the frame is not ready yet

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            continue

        face = faces[0]
        landmarks = sp(gray, face)
        embedding = face_encoder.compute_face_descriptor(frame, landmarks)
        embedding = np.array(embedding)

        # Check if this face is already registered
        for reg_emb in embeddings:
            _, matched = compare_embeddings(reg_emb, embedding)
            if matched:
                print("Face already registered.")
                continue

        # Prompt user for a name and register
        name = input("Nhap ten nguoi dung: ").strip()
        if name in labels:
            print(f"Name '{name}' already exists.")
            continue

        embeddings.append(embedding)
        labels.append(name)
        save_db()
        print(f"Face for {name} registered successfully.")

# Xac thuc khuon mat va hien thi ket qua tren khung hinh
def verify_faces_on_frame():
    global frame
    while True:
        with frame_lock:
            if frame is None:
                continue  # Skip if the frame is not ready yet

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

            # Draw bounding box and name on the frame
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            text = f"{matched_name} ({max_score:.2f}%)"
            cv2.putText(frame, text, (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Video capture thread
def video_capture_thread():
    global frame
    cap = cv2.VideoCapture(0)
    while True:
        ret, captured_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = captured_frame
    cap.release()

# Main function to start threads for face registration and verification
def main():
    # Start the video capture thread
    capture_thread = threading.Thread(target=video_capture_thread)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the face registration thread
    register_thread = threading.Thread(target=register_face)
    register_thread.daemon = True
    register_thread.start()

    # Start the face verification thread
    verify_thread = threading.Thread(target=verify_faces_on_frame)
    verify_thread.daemon = True
    verify_thread.start()

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
