import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from flask import Flask, render_template_string, Response, request, redirect, url_for
import os
import time

app = Flask(__name__)

# Load model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
interpreter = tf.lite.Interpreter(model_path=r"D:\PhamLong\code\Project\face_authentication\facenet.tflite")
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

# --- Tiện ích ---
def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

def compare_embeddings(emb1, emb2, threshold=0.2):
    distance = cosine(emb1, emb2)
    return distance < threshold

def extract_embedding(face_img):
    img = cv2.resize(face_img, (160, 160)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])[0]
    return emb / np.linalg.norm(emb)

# --- Kiểm tra chuyển động thật ---
def quick_motion_check_and_capture():
    cap = cv2.VideoCapture(0)
    frames = []
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, None
        frames.append(cv2.resize(frame, (160, 160)))
        cv2.waitKey(300)  # Chờ 300ms
    cap.release()

    diffs = [np.linalg.norm(frames[i].astype("float32") - frames[i+1].astype("float32")) for i in range(2)]
    avg_diff = np.mean(diffs)
    return (avg_diff > 15), frames[-1]

# --- Đăng ký khuôn mặt ---
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
        return "[!] Tên không hợp lệ hoặc đã tồn tại!"

    embeddings.append(new_emb)
    labels.append(name.strip())
    save_db()
    return f"[+] Đăng ký thành công cho: {name.strip()}"

# --- Xác thực ---
def verify_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "Không tìm thấy khuôn mặt!", image

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        emb = extract_embedding(face)
        matched = False
        for name, db_emb in zip(labels, embeddings):
            if compare_embeddings(emb, db_emb):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"✅ {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                matched = True
                break
        if not matched:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, f"❌ Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return "Xác thực hoàn tất.", image

# --- Webcam feed ---
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, frame = verify_faces(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string("""
    <html><body>
        <h1>Face Recognition</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480" style="border:1px solid black;"><br>
        <form action="/register" method="post">
            <label>Enter Name:</label>
            <input type="text" name="name" required>
            <button type="submit">Register</button>
        </form>
        <h2>Registered Faces</h2>
        <ul>
        {% for name in labels %}
            <li>{{ name }} 
                <a href="{{ url_for('delete', name=name) }}">Delete</a> | 
                <a href="{{ url_for('edit', name=name) }}">Edit</a>
            </li>
        {% endfor %}
        </ul>
    </body></html>
    """, labels=labels)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get("name")
    is_real, face_frame = quick_motion_check_and_capture()
    if not is_real:
        return render_template_string("<h1>⚠️ Không phát hiện chuyển động - nghi ngờ ảnh tĩnh!</h1><a href='/'>Trở lại</a>")
    message = register_face(face_frame, name)
    return render_template_string("<h1>{{ message }}</h1><a href='/'>Trở lại</a>", message=message)

@app.route('/delete/<name>')
def delete(name):
    if name in labels:
        index = labels.index(name)
        del labels[index]
        del embeddings[index]
        save_db()
    return redirect(url_for('index'))

@app.route('/edit/<name>', methods=['GET', 'POST'])
def edit(name):
    if request.method == 'POST':
        new_name = request.form.get("new_name")
        if new_name and new_name not in labels:
            index = labels.index(name)
            labels[index] = new_name
            save_db()
            return redirect(url_for('index'))
    return render_template_string("""
        <h1>Edit Name</h1>
        <form action="" method="post">
            <label for="new_name">New Name:</label>
            <input type="text" name="new_name" required>
            <button type="submit">Save</button>
        </form>
        <a href="/">Back</a>
    """)

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
