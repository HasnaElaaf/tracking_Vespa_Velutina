import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import random
from collections import deque
import math
import sys

# ---------- Utils couleurs / dessin ----------
def generate_unique_color(existing_colors):
    while True:
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        if not existing_colors:
            return color
        if all(sum(abs(c1 - c2) for c1, c2 in zip(color, ec)) > 100 for ec in existing_colors.values()):
            return color

def draw_geographic_angle_reference(frame):
    h, w = frame.shape[:2]
    origin = (w - 50, 40)
    radius = 20
    cv2.circle(frame, origin, radius, (255, 255, 255), 1)
    cv2.line(frame, origin, (origin[0], origin[1] - radius), (0, 0, 255), 2)      # N
    cv2.line(frame, origin, (origin[0] + radius, origin[1]), (0, 255, 0), 2)      # E
    cv2.line(frame, origin, (origin[0], origin[1] + radius), (255, 0, 0), 2)      # S
    cv2.line(frame, origin, (origin[0] - radius, origin[1]), (0, 255, 255), 2)    # W
    cv2.putText(frame, "N", (origin[0] - 5, origin[1] - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame, "E", (origin[0] + radius + 5, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, "S", (origin[0] - 10, origin[1] + radius + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.putText(frame, "W", (origin[0] - radius - 20, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

def draw_direction_indicator(frame, position, angle_boussole, length=30):
    # 0° = Nord, 90° = Est
    angle_cv = (90 - angle_boussole) % 360
    angle_rad = math.radians(angle_cv)
    end_x = int(position[0] + length * math.cos(angle_rad))
    end_y = int(position[1] - length * math.sin(angle_rad))
    cv2.arrowedLine(frame, position, (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

# fonction de log (sera redéfinie après chemin de sortie)
def log_direction(track_id, angle):
    pass

# ---------- Ouverture vidéo multi-backends ----------
def try_open_video(path):
    if not os.path.exists(path):
        return None, f"Fichier vidéo introuvable: {path}"

    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, "default"

    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap, "ffmpeg"
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(path, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "gstreamer-file"
    except Exception:
        pass

    try:
        gst = f"filesrc location={path} ! decodebin ! videoconvert ! appsink"
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap, "gstreamer-pipeline"
    except Exception:
        pass

    return None, (
        "Impossible d'ouvrir la vidéo.\n"
        "➡️ Installe FFmpeg: sudo apt-get install -y ffmpeg\n"
        "➡️ Ou installe GStreamer + plugins.\n"
        "➡️ Ou convertis en AVI/MJPEG:\n"
        f"   ffmpeg -i {path} -vcodec mjpeg -q:v 5 -an {os.path.splitext(path)[0]}.avi"
    )

# ---------- Writer robuste (MP4 puis fallback AVI/MJPEG) ----------
def open_video_writer(width, height, fps):
    ts = time.strftime("%Y%m%d-%H%M%S")
    videos_dir = os.path.expanduser("~/Videos")
    os.makedirs(videos_dir, exist_ok=True)

    mp4_path = os.path.join(videos_dir, f"vespa_{ts}.mp4")
    avi_path = os.path.join(videos_dir, f"vespa_{ts}.avi")

    # 1) MP4 (mp4v)
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc_mp4, fps, (width, height))
    if out.isOpened():
        return out, mp4_path, "mp4v"

    # 2) Fallback AVI/MJPEG (super compatible sur Pi)
    fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(avi_path, fourcc_mjpg, fps, (width, height))
    if out.isOpened():
        return out, avi_path, "MJPG-AVI"

    return None, None, None

# ---------- Config ----------
VIDEO_PATH = "/home/helglalef/Downloads/vlc-record-2025-08-14-13h57m34s-video_12_1.mp4-.avi"
MODEL_PATH = "/home/helglalef/Desktop/vespcv/version11/best.pt"
MODEL_ONNX = "/home/helglalef/Desktop/vespcv/version11/best.onnx"

CLASS_NAME = "Vespa_velutina"
CONFIDENCE_THRESHOLD = 0.5
IMGSZ = 640
MAX_TRAJECTORY_LENGTH = 64
ANGLE_FILTER_WINDOW = 5
MAX_ASSOCIATION_DIST = 50

# ---------- Modèle ----------
if not os.path.exists(MODEL_ONNX):
    print("Conversion du modèle en ONNX...")
    os.makedirs(os.path.dirname(MODEL_ONNX), exist_ok=True)
    model_tmp = YOLO(MODEL_PATH)
    model_tmp.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
    print(f"Modèle ONNX généré : {MODEL_ONNX}")

model = YOLO(MODEL_ONNX, task='detect')
class_names = model.names
print(f"Modèle chargé | Classes : {class_names}")

# ---------- Ouverture vidéo ----------
cap, backend_info = try_open_video(VIDEO_PATH)
if cap is None:
    print(backend_info)
    sys.exit(1)
print(f"OK: vidéo ouverte via backend = {backend_info}")

# test de lecture
for i in range(3):
    ok, test_frame = cap.read()
    print(f"read() test {i+1} ->", ok, (None if not ok else test_frame.shape))
    if not ok:
        print("⚠️ cap.read() a échoué.")
        break
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ---------- Sortie vidéo (avec fallback) ----------
fps_src = cap.get(cv2.CAP_PROP_FPS)
if not fps_src or fps_src <= 1:
    fps_src = 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

out, output_path, writer_kind = open_video_writer(width, height, fps_src)
if out is None:
    print("❌ Impossible d'ouvrir l'écrivain vidéo (mp4v & MJPG).")
    print("   Essaie: sudo apt-get install -y ffmpeg   (pour mp4)")
    print("   Ou bascule manuellement en AVI/MJPEG.")
    sys.exit(1)

print(f"Écriture -> {output_path} (codec: {writer_kind})")

# définir le vrai logger maintenant que log_path est connu
log_path = os.path.splitext(output_path)[0] + "_angles.txt"
open(log_path, "w").close()
def log_direction(track_id, angle):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} | ID:{track_id} | Angle:{angle:.1f}°\n")

# ---------- Tracking ----------
tracks = {}
colors = {}
next_id = 0
frame_idx = 0
frames_written = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin / échec lecture frame")
        break

    frame_idx += 1
    t0 = time.time()

    # Letterbox
    h, w = frame.shape[:2]
    scale = IMGSZ / max(h, w)
    resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
    padded = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized

    # Détection
    results = model(padded, imgsz=IMGSZ, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    if results:
        r = results[0]
        boxes = r.boxes.cpu().numpy()
        for box in boxes:
            cls_i = int(box.cls)
            if class_names.get(cls_i, "") != CLASS_NAME:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append(((cx, cy), (x1, y1, x2, y2)))

    # Association simple
    unmatched_dets = list(range(len(detections)))
    updated_track_ids = []

    for tid, data in list(tracks.items()):
        if not data.get('path') or not unmatched_dets:
            continue
        last_pt = data['path'][-1]
        dists = []
        for di in unmatched_dets:
            c, _ = detections[di]
            d2 = (c[0]-last_pt[0])**2 + (c[1]-last_pt[1])**2
            dists.append((d2, di))
        if not dists:
            continue
        dists.sort(key=lambda x: x[0])
        dist2, di = dists[0]
        if dist2 <= (MAX_ASSOCIATION_DIST**2):
            c, bb = detections[di]
            tracks[tid]['path'].append(c)
            tracks[tid]['bbox'] = bb
            unmatched_dets.remove(di)
            updated_track_ids.append(tid)

    for di in unmatched_dets:
        c, bb = detections[di]
        tracks[next_id] = {
            'path': deque([c], maxlen=MAX_TRAJECTORY_LENGTH),
            'angles': deque(maxlen=ANGLE_FILTER_WINDOW),
            'bbox': bb
        }
        colors[next_id] = generate_unique_color(colors)
        updated_track_ids.append(next_id)
        next_id += 1

    # Dessin + angle
    for tid, data in list(tracks.items()):
        path = data['path']
        if len(path) > 1:
            pts = np.array(path, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=colors.get(tid, (255,255,255)), thickness=3)
            pt1, pt2 = path[-2], path[-1]
            dx = pt2[0] - pt1[0]
            dy = pt1[1] - pt2[1]
            angle = (90 - math.degrees(math.atan2(dy, dx))) % 360
            data['angles'].append(angle)
            smoothed_angle = float(np.mean(data['angles'])) if data['angles'] else angle
            draw_direction_indicator(frame, pt2, smoothed_angle)
            cv2.putText(frame, f"{smoothed_angle:.1f}", (pt2[0]+10, pt2[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            log_direction(tid, smoothed_angle)

        bb = data.get('bbox')
        if bb:
            x1,y1,x2,y2 = bb
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors.get(tid,(255,255,255)), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(tid,(255,255,255)), 2)

    # HUD
    active_count = len(updated_track_ids)
    cv2.putText(frame, f"Frelons: {active_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    fps_now = 1.0 / max(1e-3, time.time() - time.time() + t0)  # petite protection
    cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    draw_geographic_angle_reference(frame)

    # Écriture frame
    out.write(frame)
    frames_written += 1

    # Affichage local (désactive si headless)
    cv2.imshow("Tracking Frelons avec Boussole", frame)
    if cv2.waitKey(1) == ord('q'):
        break

    if frame_idx % 30 == 0:
        print(f"Frames traitées: {frame_idx} | écrites: {frames_written}")

cap.release()
out.release()
cv2.destroyAllWindows()

print("Terminé.")
print("Frames écrites   ->", frames_written)
print("Vidéo enregistrée->", output_path)
print("Log enregistré   ->", log_path)
if frames_written == 0:
    print("⚠️ Aucune frame écrite. Le fichier de sortie peut être vide.")
    print("   Vérifie le backend vidéo ou convertis la source en AVI/MJPEG.")
