import dlib
import cv2
import numpy as np
import os
from itertools import combinations

# Load model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

DB_FILE = "face_db.npz"
embeddings = []
labels = []
THRESHOLD = 0.5

if os.path.exists(DB_FILE):
    data = np.load(DB_FILE, allow_pickle=True)
    embeddings = list(data["embeddings"])
    labels = list(data["labels"])

def suggest_optimal_threshold():
    global THRESHOLD
    if len(embeddings) < 2:
        return

    same_dists = []
    diff_dists = []

    for (i1, emb1), (i2, emb2) in combinations(enumerate(embeddings), 2):
        dist = np.linalg.norm(emb1 - emb2)
        if labels[i1] == labels[i2]:
            same_dists.append(dist)
        else:
            diff_dists.append(dist)

    if not diff_dists:
        return

    thresholds = np.linspace(0.2, 5.0, 250)
    best_threshold = 0.5
    best_acc = 0

    for t in thresholds:
        if t >= 0.5:
            continue
        tp = np.sum(np.array(same_dists) <= t)
        fn = np.sum(np.array(same_dists) > t)
        tn = np.sum(np.array(diff_dists) > t)
        fp = np.sum(np.array(diff_dists) <= t)
        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    THRESHOLD = best_threshold
def compare_embeddings(embedding1, embedding2):
    dist = np.linalg.norm(embedding1 - embedding2)
    return dist, dist < THRESHOLD

def save_db():
    np.savez(DB_FILE, embeddings=embeddings, labels=labels)

def compute_embedding(image, face_rect):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = sp(gray, face_rect)
    embedding = face_encoder.compute_face_descriptor(image, landmarks)
    return np.array(embedding)

def is_face_centered(face, frame_shape, threshold_ratio=0.2):
    face_center_x = (face.left() + face.right()) // 2
    face_center_y = (face.top() + face.bottom()) // 2
    frame_center_x = frame_shape[1] // 2
    frame_center_y = frame_shape[0] // 2
    center_diff = np.linalg.norm([face_center_x - frame_center_x, face_center_y - frame_center_y])
    return center_diff < threshold_ratio * min(frame_shape[0], frame_shape[1])

def get_pose_direction(landmarks):
    nose = landmarks.part(30)
    chin = landmarks.part(8)
    forehead = landmarks.part(27)
    left_cheek = landmarks.part(2)  
    right_cheek = landmarks.part(14) 

    face_height = chin.y - forehead.y
    nose_chin_dist = chin.y - nose.y
    vertical_ratio = nose_chin_dist / face_height if face_height > 0 else 0

    face_width = right_cheek.x - left_cheek.x
    nose_left_dist = nose.x - left_cheek.x
    horizontal_ratio = nose_left_dist / face_width if face_width > 0 else 0

    if vertical_ratio < 0.65:
        return "looking down"
    elif vertical_ratio > 0.8:
        return "looking up"
    if horizontal_ratio > 0.6:
        return "looking left"
    elif horizontal_ratio < 0.4:
        return "looking right"
    return "frontal"

def register_multi_pose(cap):
    required_poses = {
        "frontal": "Nhin thang vao camera",
        "looking left": "Nghieng trai",
        "looking right": "Nghieng phai",
        "looking up": "Ngan len",
        "looking down": "Cui xuong"
    }

    captured_embeddings = []
    name = input("Nhap ten nguoi dung: ").strip()
    if not name:
        print("[-] Ten khong hop le.")
        return
    if name in labels:
        print("[!] Ten da ton tai.")
        return

    print("[*] Huong dan nguoi dung thuc hien tung goc...")
    for pose, message in required_poses.items():
        print(f"[{pose.upper()}] {message}")
        pose_captured = False

        while not pose_captured:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            display_frame = frame.copy()

            if len(faces) == 1:
                face = faces[0]
                if is_face_centered(face, frame.shape):
                    landmarks = sp(gray, face)
                    detected_pose = get_pose_direction(landmarks)

                    cv2.rectangle(display_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{message} - Nhan 'c' de chup", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    if detected_pose == pose:
                        cv2.putText(display_frame, f"Goc {pose} dung", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, f"Goc hien tai: {detected_pose}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    cv2.imshow("Dang ky", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c') and detected_pose == pose:
                        emb = compute_embedding(frame, face)

                        if pose == "frontal":
                            for saved_emb in embeddings:
                                dist = np.linalg.norm(emb - saved_emb)
                                if dist < THRESHOLD:
                                    print("[!] Khuon mat da ton tai trong he thong.")
                                    cv2.destroyWindow("Dang ky")
                                    return

                        captured_embeddings.append(emb)
                        print(f"[+] Da chup goc {pose}")
                        pose_captured = True
                else:
                    cv2.putText(display_frame, "Can giua khung hinh!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow("Dang ky", display_frame)
                    cv2.waitKey(1)
            else:
                cv2.putText(display_frame, "Can co 1 khuon mat trong khung!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Dang ky", display_frame)
                cv2.waitKey(1)

    if len(captured_embeddings) == len(required_poses):
        avg_embedding = np.mean(captured_embeddings, axis=0)
        embeddings.append(avg_embedding)
        labels.append(name)
        save_db()
        print(f"[+] Dang ky hoan tat cho {name}")
    else:
        print("[!] Dang ky chua hoan tat.")

    cv2.destroyWindow("Dang ky")

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
    suggest_optimal_threshold()
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
            register_multi_pose(cap)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
