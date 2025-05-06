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

def register_process(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        print("[!] Khong phat hien khuon mat.")
        return

    face = faces[0]
    landmarks = sp(gray, face)
    embedding = face_encoder.compute_face_descriptor(face_img, landmarks)
    embedding = np.array(embedding)

    for reg_emb in embeddings:
        _, matched = compare_embeddings(reg_emb, embedding)
        if matched:
            print("[!] Khuon mat da ton tai.")
            return

    name = input("[?] Nhap ten de dang ky: ").strip()
    if not name:
        print("[-] Dang ky bi huy.")
        return
    if name in labels:
        print(f"[!] Ten '{name}' da ton tai.")
        return

    embeddings.append(embedding)
    labels.append(name)
    save_db()
    print(f"[+] Dang ky thanh cong cho {name}")

def register_interactive(cap):
    while True:
        print("[*] Dang chup khuon mat...")
        ret, frame = cap.read()
        if not ret:
            print("[!] Loi camera.")
            return

        # Hien thi anh chup len
        preview = frame.copy()
        cv2.putText(preview, "Nhan 'y' de xac nhan | 'c' de chup lai | 'q' de huy", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Xac nhan dang ky", preview)

        # Doi nguoi dung bam phim
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('y'):
                cv2.destroyWindow("Xac nhan dang ky")
                register_process(frame)
                return
            elif k == ord('c'):
                cv2.destroyWindow("Xac nhan dang ky")
                break  # chup lai
            elif k == ord('q'):
                cv2.destroyWindow("Xac nhan dang ky")
                print("[-] Dang ky bi huy.")
                return

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

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        verify_faces_on_frame(frame)

        cv2.putText(frame, "'r': Dang ky | 'q': Thoat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Nhan dien khuon mat", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            register_interactive(cap)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
