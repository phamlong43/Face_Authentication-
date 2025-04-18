import cv2
import numpy as np
import tensorflow.lite as tflite
from scipy.spatial.distance import cosine
import os
import threading

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

interpreter = tflite.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

DB_FILE = "face_db.npz"
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embedding"])
    labels = list(data["label"])

def extract_embedding(face_img):
    img = cv2.resize(face_img, (160, 160)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])[0]
    return emb / np.linalg.norm(emb)

def compare_embeddings(emb1, emb2, threshold=0.2):  
    return cosine(emb1, emb2) < threshold

def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

def handle_register(face_img):
    new_emb = extract_embedding(face_img)

    for emb in embeddings:
        if compare_embeddings(emb, new_emb):
            print("[!] Khuon mat da ton tai!")
            return

    name = input("Nhap ten nguoi dung: ").strip()

    if name in labels:
        print(f"[!] Ten '{name}' da ton tai!")
        return

    embeddings.append(new_emb)
    labels.append(name)
    save_db()
    print(f"[+] Dang ky thanh cong cho {name}")

def main():
    cap = cv2.VideoCapture(0)
    mode = "idle"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60:
                continue

            face = small_frame[y:y+h, x:x+w]
            cv2.rectangle(small_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if mode == "verify":
                emb_live = extract_embedding(face)
                found = False
                for name, emb_reg in zip(labels, embeddings):
                    if compare_embeddings(emb_live, emb_reg):
                        cv2.putText(small_frame, f"? {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        found = True
                        break
                if not found:
                    cv2.putText(small_frame, "? Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            elif mode == "register":
                threading.Thread(target=handle_register, args=(face.copy(),)).start()
                mode = "idle"

        cv2.putText(small_frame, "'r': Dang ky | 'v': Xac thuc | 'q': Thoat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Face Verification", small_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            mode = "register"
        elif key == ord('v'):
            mode = "verify"
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
