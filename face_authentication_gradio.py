import cv2
import numpy as np
import tensorflow.lite as tflite
from scipy.spatial.distance import cosine
import gradio as gr
import os

# Load FaceNet Lite model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
interpreter = tflite.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load database
DB_FILE = "face_db.npz"
embeddings = []
labels = []

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

# Extract embedding
def extract_embedding(face_img):
    img = cv2.resize(face_img, (160, 160)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])[0]
    return emb / np.linalg.norm(emb)

# So sánh cosine
def compare_embeddings(emb1, emb2, threshold=0.4):  
    return cosine(emb1, emb2) < threshold

# Lưu database
def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

# Đăng ký khuôn mặt
def register_face(image, name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Không tìm thấy khuôn mặt!"

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]

    new_emb = extract_embedding(face)
    for emb in embeddings:
        if compare_embeddings(emb, new_emb):
            return "[!] Khuôn mặt đã tồn tại!"

    if name.strip() == "" or name in labels:
        return f"[!] Tên không hợp lệ hoặc đã tồn tại!"

    embeddings.append(new_emb)
    labels.append(name.strip())
    save_db()
    return f"[+] Đăng ký thành công cho: {name.strip()}"

# Xác thực khuôn mặt
def verify_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Không tìm thấy khuôn mặt!", image

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    emb = extract_embedding(face)

    for name, db_emb in zip(labels, embeddings):
        if compare_embeddings(emb, db_emb):
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(image, f"✅ {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            return f"Xác thực: {name}", image

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.putText(image, f"❌ Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return "Không nhận diện được khuôn mặt!", image

# Giao diện Gradio
register_ui = gr.Interface(
    fn=register_face,
    inputs=[gr.Image(type="numpy"), gr.Text(label="Tên người dùng")],
    outputs="text",
    title="Đăng ký khuôn mặt"
)

verify_ui = gr.Interface(
    fn=verify_face,
    inputs=gr.Image(type="numpy"),
    outputs=["text", "image"],
    title="Xác thực khuôn mặt"
)

# Combine tabs
gr.TabbedInterface([register_ui, verify_ui], ["Đăng ký", "Xác thực"]).launch()
