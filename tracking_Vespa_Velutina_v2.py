#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLOv11
import time
import os
import random
import geojson
from collections import deque
import math
import folium
from folium.plugins import AntPath
import webbrowser
import json
import serial
import pynmea2
from branca.element import Template, MacroElement

# Calibration
REFERENCE_DISTANCE_METERS = 2.0
PIXEL_TO_DEG = 0.00001
REFERENCE_DISTANCE_PIXELS = None  # Initialisé ici
frame_shape = (480, 640)  # (height, width)

# Coordonnées par défaut qui seront écrasées par la suite par CURRENT_LAT/LON
MAP_CENTER_LAT = 48.121177  # Exemple: latitude du piège 1
MAP_CENTER_LON = -1.632677  # Exemple: longitude du piège 1

# Configuration des pièges
PIEGES = {
    "piege_1": {
        "lat": 48.121177, 
        "lon": -1.632677,
        "t_t": 200  # Temps aller-retour en secondes
    },
    "piege_2": {
        "lat": 48.119882,
        "lon": -1.636597, 
        "t_t": 100
    }
}

# Dossier de stockage
DATA_DIR = "tracking_data"
os.makedirs(DATA_DIR, exist_ok=True)  # Crée DATA_DIR AVANT toute utilisation

def get_piege_path(piege_id):
    """Retourne le chemin des fichiers pour un piège donné"""
    return os.path.join(DATA_DIR, f"piege_{piege_id}")

def save_piege_config(piege_id, lat, lon):
    config = {
        "MAP_CENTER_LAT": lat,
        "MAP_CENTER_LON": lon,
        "timestamp": time.time()
    }
    config_path = os.path.join(get_piege_path(piege_id), "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

# Création des dossiers et fichiers de configuration pour tous les pièges
for piege_id in PIEGES:
    piege_dir = os.path.join(DATA_DIR, f"piege_{piege_id}")
    os.makedirs(piege_dir, exist_ok=True)
    config_path = os.path.join(piege_dir, "config.json")
    if not os.path.exists(config_path):
        save_piege_config(piege_id, PIEGES[piege_id]["lat"], PIEGES[piege_id]["lon"])

# Piège actif (modifier cette variable pour changer de piège)
PIEGE_ACTIF = "piege_1"  # Nom correspondant à une clé du dictionnaire PIEGES

# Variables dérivées 
CURRENT_LAT = PIEGES[PIEGE_ACTIF]["lat"]
CURRENT_LON = PIEGES[PIEGE_ACTIF]["lon"]
CURRENT_T_T = PIEGES[PIEGE_ACTIF]["t_t"]

def load_piege_config(piege_id):
    config_path = os.path.join(get_piege_path(piege_id), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    else:
        # Retourne une configuration par défaut si le fichier n'existe pas
        return {
            "MAP_CENTER_LAT": PIEGES[piege_id]["lat"],
            "MAP_CENTER_LON": PIEGES[piege_id]["lon"],
            "timestamp": time.time()
        }

def calculate_pixel_to_deg_factor():
    global PIXEL_TO_DEG
    meter_to_deg = 1 / 111320
    if REFERENCE_DISTANCE_PIXELS is None or REFERENCE_DISTANCE_PIXELS == 0:
        print("Calibration non effectuée - Utilisation de la valeur par défaut")
        PIXEL_TO_DEG = 0.00001
    else:
        PIXEL_TO_DEG = (REFERENCE_DISTANCE_METERS * meter_to_deg) / REFERENCE_DISTANCE_PIXELS
        print(f"Facteur de calibration calculé : {PIXEL_TO_DEG:.10f} degrés/pixel")

calculate_pixel_to_deg_factor()

def generate_unique_color(existing_colors):
    while True:
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        if not existing_colors:
            return color
        if all(sum(abs(c1 - c2) for c1, c2 in zip(color, ec)) > 100 for ec in existing_colors.values()):
            return color

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def load_existing_trajectories(piege_id=PIEGE_ACTIF):  
    try:
        filename = os.path.join(get_piege_path(piege_id), "trajectories.geojson")
        if os.path.exists(filename):
            with open(filename) as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                    max_id = max([f['properties'].get('track_id', 0) for f in data['features']], default=0)
                    return data, max_id
        return {'type': 'FeatureCollection', 'features': []}, 0
    except Exception as e:
        print(f"Erreur de chargement des trajectoires: {e}")
        return {'type': 'FeatureCollection', 'features': []}, 0

def save_trajectories_to_geojson(trajectories, colors, frame_shape, piege_id=None):
    if piege_id is None:
        piege_id = PIEGE_ACTIF
    piege_dir = get_piege_path(piege_id)
    os.makedirs(piege_dir, exist_ok=True)
    filename = os.path.join(piege_dir, "trajectories.geojson")
    existing_data, _ = load_existing_trajectories(piege_id)
    # Initialisation de new_features
    new_features = []
    for track_id, data in trajectories.items():
        path = list(data['path'])
        if len(path) < 2:
            continue
        gps_path = []
        for x, y in path:
            lon = CURRENT_LON + (x - frame_shape[1]/2) * PIXEL_TO_DEG
            lat = CURRENT_LAT - (y - frame_shape[0]/2) * PIXEL_TO_DEG
            gps_path.append([lon, lat])
        if len(gps_path) < 2:
            continue
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": gps_path
            },
            "properties": {
                "track_id": track_id,
                "color": rgb_to_hex(colors[track_id]),
                "timestamps": data.get('timestamps', [time.time()] * len(gps_path)),
                "piege_id": piege_id,  # Nouveau
                "t_t": PIEGES[piege_id]["t_t"]  # Stocke le paramètre            
            }
        }
        new_features.append(feature)
    existing_ids = {f['properties']['track_id'] for f in existing_data['features']}
    features_to_keep = [f for f in existing_data['features'] if f['properties']['track_id'] not in trajectories]
    feature_collection = {
        "type": "FeatureCollection",
        "features": features_to_keep + new_features
    }
    with open(filename, 'w') as f:
        json.dump(feature_collection, f, indent=2)
                
        
def create_folium_map(all_features, t_n=45, S_min=1.8, S_max=5.4):
    """
    Crée une carte Folium interactive à partir d'un GeoJSON de trajectoires.
    - all_features : dict GeoJSON (ex: contenu de trajectories.geojson)
    - t_t, t_n, S_min, S_max : paramètres pour le calcul des distances
    Retourne le nom du fichier HTML généré.
    """
    try:
        m = folium.Map(
            location=[MAP_CENTER_LAT, MAP_CENTER_LON],
            zoom_start=19,
            tiles='OpenStreetMap'
        )
        if not isinstance(all_features, dict) or 'features' not in all_features:
            raise ValueError("Format GeoJSON invalide")
            
        color_palette = [
                '#FF0000',  # Rouge vif
                '#0000FF',  # Bleu foncé
                '#006400',  # Vert foncé
                '#8B008B',  # Magenta foncé
                '#FF8C00',  # Orange foncé
                '#8B0000',  # Rouge foncé
                ]
        for i, feature in enumerate(all_features['features']):
            try:
                coords = feature.get('geometry', {}).get('coordinates', [])
                if len(coords) < 2:
                    continue
                properties = feature.get('properties', {})
                piege_id = properties.get('piege_id', PIEGE_ACTIF)
                t_t = properties.get('t_t', PIEGES.get(piege_id, {}).get("t_t", 90))                 
                #track_id = properties.get('track_id', 'inconnu')
                
                color = properties.get('color', color_palette[i % len(color_palette)])
                
                # Conversion [lon, lat] → [lat, lon] pour Folium
                folium_coords = [[lat, lon] for [lon, lat] in coords]
                last_point = folium_coords[-1]
                
                # Cercles de distance
                D_min, D_max = calculer_distances(t_t, t_n, S_min, S_max)
                
                folium.Circle(
                    location=last_point,
                    radius=D_max,
                    color=color,
                    fill=True,
                    fill_opacity=0.3, #0.1
                    weight=2,  #1
                    popup=f"Zone maximale (D_max): {D_max:.1f}m"
                ).add_to(m)
                
                folium.Circle(
                    location=last_point,
                    radius=D_min,
                    color=color,
                    fill=True,
                    fill_opacity=0.1,
                    weight=2,
                    popup=f"Zone probable (D_min): {D_min:.1f}m"
                ).add_to(m)
                
                # Visualisation de direction (flèche extrapolée)
                if len(folium_coords) >= 3:
                    last_points = folium_coords[-3:]
                    start_dir = last_points[-2]  # Avant-dernier point
                    end_dir = last_points[-1]    # Dernier point (identique à last_point)
                    
                    dx = end_dir[1] - start_dir[1]  # delta longitude
                    dy = end_dir[0] - start_dir[0]  # delta latitude
                    angle_rad = math.atan2(dy, dx)
                    
                    from geopy.distance import geodesic
                    origin_point = (end_dir[0], end_dir[1])
                    
                    # Points extrapolés (direction principale)
                    extrapolated_point = geodesic(meters=D_max).destination(origin_point, math.degrees(angle_rad))
                    extrapolated_coords = [extrapolated_point.latitude, extrapolated_point.longitude]
                    
                    # Points d'erreur (±15°)
                    left_point = geodesic(meters=D_max).destination(origin_point, math.degrees(angle_rad) + 15)
                    left_coords = [left_point.latitude, left_point.longitude]
                    
                    right_point = geodesic(meters=D_max).destination(origin_point, math.degrees(angle_rad) - 15)
                    right_coords = [right_point.latitude, right_point.longitude]
                        
                    # Lignes partant du dernier point de la trajectoire (last_point)
                    folium.PolyLine([end_dir, extrapolated_coords], color=color, weight=3, dash_array='5,5').add_to(m)
                    folium.PolyLine([end_dir, left_coords], color=color, weight=2, dash_array='5,5', opacity=0.5).add_to(m)
                    folium.PolyLine([end_dir, right_coords], color=color, weight=2, dash_array='5,5', opacity=0.5).add_to(m)
                    
                    # Flèche
                    folium.RegularPolygonMarker(
                    location=extrapolated_coords,
                    fill_color=color,
                    number_of_sides=3,
                    radius=10,
                    rotation=math.degrees(angle_rad),
                    fill_opacity=0.8
                    ).add_to(m)  
                    
                    # Trajectoire principale
                    AntPath(
                    folium_coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    delay=1000,
                    pulse_color=color,
                    dash_array=[1, 0]
                    ).add_to(m)
                    
            except Exception as e:
                print(f"Erreur sur feature {track_id}: {e}")
                continue
        # Plugins communs
        folium.plugins.Fullscreen().add_to(m)
        folium.plugins.MiniMap().add_to(m)
        folium.plugins.MeasureControl().add_to(m)
        
        map_file = "combined_tracking_map.html"
        m.save(map_file)
        
        # CSS responsive
        with open(map_file, 'a') as f:
            f.write("""
            <style>
                html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
                .folium-map { height: 100vh !important; width: 100% !important; }
            </style>
            <script>
                function onMapClick(e) {
                var popup = L.popup()
                .setLatLng(e.latlng)
                .setContent("Coordonnées : " + e.latlng.lat.toFixed(6) + ", " + e.latlng.lng.toFixed(6))
                .openOn(map);
                console.log("Clique détecté à : ", e.latlng.lat, e.latlng.lng);
                }
                map.on('click', onMapClick);
            </script>
            """)
        return map_file
        
    except Exception as e:
        print(f"Erreur création carte: {str(e)}")
        return None

def draw_geographic_angle_reference(frame):
    h, w = frame.shape[:2]
    origin = (w - 50, 40)
    radius = 20
    cv2.circle(frame, origin, radius, (255, 255, 255), 1)
    cv2.line(frame, origin, (origin[0], origin[1] - radius), (0, 0, 255), 2)
    cv2.line(frame, origin, (origin[0] + radius, origin[1]), (0, 255, 0), 2)
    cv2.line(frame, origin, (origin[0], origin[1] + radius), (255, 0, 0), 2)
    cv2.line(frame, origin, (origin[0] - radius, origin[1]), (0, 255, 255), 2)
    cv2.putText(frame, "N", (origin[0] - 5, origin[1] - radius - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "E", (origin[0] + radius + 5, origin[1] + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "S", (origin[0] - 10, origin[1] + radius + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, "W", (origin[0] - radius - 20, origin[1] + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def log_direction(track_id, angle):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("direction_logs.txt", "a") as f:
        f.write(f"{timestamp} | ID:{track_id} | Angle:{angle:.1f}°\n")

def draw_direction_indicator(frame, position, angle_boussole, length=30):
    angle_cv = (90 - angle_boussole) % 360
    angle_rad = math.radians(angle_cv)
    end_x = int(position[0] + length * math.cos(angle_rad))
    end_y = int(position[1] - length * math.sin(angle_rad))
    cv2.arrowedLine(frame, position, (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)

def calculate_offset_point(start, end, angle_deg, distance):
    """Version améliorée pour conserver la longueur de la ligne principale."""
    angle_rad = math.radians(angle_deg)
    dx = end[1] - start[1]  # delta longitude
    dy = end[0] - start[0]  # delta latitude
    norm = math.sqrt(dx**2 + dy**2)
    if norm == 0:
        return end
    # Rotation du vecteur directeur
    rotated_dx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    rotated_dy = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
    # Normalisation et application de la distance
    rotated_norm = math.sqrt(rotated_dx**2 + rotated_dy**2)
    scaled_dx = rotated_dx / rotated_norm * distance
    scaled_dy = rotated_dy / rotated_norm * distance
    return (end[0] + scaled_dy, end[1] + scaled_dx)

def calculer_distances(t_t, t_n=45, S_min=1.8, S_max=5.4):
    D_min = ((t_t - t_n) / 2) * S_min
    D_max = ((t_t - t_n) / 2) * S_max
    return D_min, D_max

def calculate_angle(path):
    if len(path) < 2:
        return 0
    pts = np.array([(p[0], -p[1]) for p in path[-3:]])
    dx = pts[-1,0] - pts[0,0]
    dy = pts[-1,1] - pts[0,1]
    angle = (math.degrees(math.atan2(dy, dx)) - 90 + 360) % 360
    return angle

def draw_calibration_interface(frame):
    cv2.putText(frame, "CALIBRATION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Distance réelle: {REFERENCE_DISTANCE_METERS}m", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Clic gauche sur les 2 extrémités", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Appuyez sur 'c' pour valider", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, pt in enumerate(calibration_points):
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(calibration_points) == 2:
        cv2.line(frame, calibration_points[0], calibration_points[1], (0, 255, 255), 2)
        distance_px = math.sqrt((calibration_points[1][0]-calibration_points[0][0])**2 +
                      (calibration_points[1][1]-calibration_points[0][1])**2)
        cv2.putText(frame, f"Distance: {distance_px:.1f} px",
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def mouse_callback(event, x, y, flags, param):
    if calibration_mode and event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 2:
            calibration_points.append((x, y))
            print(f"Point {len(calibration_points)}: ({x}, {y})")

def changer_piege(nouveau_piege):
    global PIEGE_ACTIF, CURRENT_LAT, CURRENT_LON, CURRENT_T_T, trajectories, colors, id_counter
    if nouveau_piege in PIEGES:
        PIEGE_ACTIF = nouveau_piege
        CURRENT_LAT = PIEGES[PIEGE_ACTIF]["lat"]
        CURRENT_LON = PIEGES[PIEGE_ACTIF]["lon"]
        CURRENT_T_T = PIEGES[PIEGE_ACTIF]["t_t"]
        save_piege_config(PIEGE_ACTIF, CURRENT_LAT, CURRENT_LON)
        trajectories = {}
        colors = {}
        id_counter = 0
        print(f"\n[SYSTEM] Piège changé : {PIEGE_ACTIF}")
        print(f"• Position : {CURRENT_LAT}, {CURRENT_LON}")
        print(f"• Temps aller-retour : {CURRENT_T_T}s\n")
        return True
    else:
        print(f"\n[ERREUR] Piège '{nouveau_piege}' non configuré")
        print(f"Pièges disponibles : {list(PIEGES.keys())}\n")
        return False

def afficher_config_actuelle():
    print("\n[CONFIGURATION ACTUELLE]")
    print(f"Piège : {PIEGE_ACTIF}")
    print(f"Position : {CURRENT_LAT}, {CURRENT_LON}")
    print(f"Temps aller-retour (t_t) : {CURRENT_T_T}s")
    print(f"Dossier de données : {DATA_DIR}/{PIEGE_ACTIF}\n")     

# Configuration camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Configuration modèle
MODEL_PATH = "/home/helglalef/Desktop/Projects/vespcv/best.pt"
MODEL_ONNX = "/home/helglalef/Desktop/Projects/vespcv/best.onnx"
CLASS_NAME = "Vespa_velutina"
CONFIDENCE_THRESHOLD = 0.5
IMGSZ = 640
MAX_TRAJECTORY_LENGTH = 64
ANGLE_FILTER_WINDOW = 5

# Conversion automatique en ONNX
if not os.path.exists(MODEL_ONNX):
    print("Conversion du modèle en ONNX...")
    os.makedirs(os.path.dirname(MODEL_ONNX), exist_ok=True)
    model = YOLO(MODEL_PATH)
    model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
    print(f"Modèle ONNX généré : {MODEL_ONNX}")

# Chargement modèle
model = YOLO(MODEL_ONNX)
class_names = model.names
print(f"Modèle chargé | Classes : {class_names}")

# Structures de données
trajectories = {}
colors = {}
id_counter = 0
total_count = 0
calibration_points = []
calibration_mode = False

# Initialisation fichier log
with open("direction_logs.txt", "w") as f:
    f.write("")

try:
    existing_data, last_id = load_existing_trajectories()
    id_counter = last_id + 1
    cv2.namedWindow("Tracking Frelons avec Boussole")
    cv2.setMouseCallback("Tracking Frelons avec Boussole", mouse_callback)
    while True:
        start_time = time.time()
        try:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            frame_shape = (h, w)
            if calibration_mode:
                draw_calibration_interface(frame)
            else:
                scale = IMGSZ / max(h, w)
                resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
                padded = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
                padded[:resized.shape[0], :resized.shape[1]] = resized
                results = model(padded, imgsz=IMGSZ, conf=CONFIDENCE_THRESHOLD, verbose=False)
                current_detections = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        if class_names[int(box.cls)] == CLASS_NAME:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            current_detections.append(center)
                            if id_counter not in colors:
                                colors[id_counter] = generate_unique_color(colors)
                            color = colors[id_counter]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"ID:{id_counter}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if current_detections:
                    centers = np.array(current_detections)
                    for id_ in list(trajectories.keys()):
                        if id_ in trajectories and 'path' in trajectories[id_] and len(trajectories[id_]['path']) > 0:
                            last_pt = trajectories[id_]['path'][-1]
                            if len(centers) > 0:
                                distances = np.linalg.norm(centers - last_pt, axis=1)
                                min_idx = np.argmin(distances)
                                if distances[min_idx] < 50:
                                    trajectories[id_]['path'].append(tuple(centers[min_idx]))
                                    trajectories[id_]['timestamps'].append(time.time())
                                    centers = np.delete(centers, min_idx, 0)
                    for pt in centers:
                        trajectories[id_counter] = {
                            'path': deque([tuple(pt)], maxlen=MAX_TRAJECTORY_LENGTH),
                            'angles': deque(maxlen=ANGLE_FILTER_WINDOW),
                            'timestamps': [time.time()]
                        }
                        id_counter += 1
                for id_, data in trajectories.items():
                    path = list(data['path'])
                    if len(path) > 1:
                        pts = np.array(path, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False,
                                    color=colors.get(id_, (255, 255, 255)), thickness=3)
                        pt1 = path[-2]
                        pt2 = path[-1]
                        dx = pt2[0] - pt1[0]
                        dy = pt1[1] - pt2[1]
                        angle = (90 - math.degrees(math.atan2(dy, dx))) % 360
                        data['angles'].append(angle)
                        smoothed_angle = np.mean(data['angles']) if data['angles'] else angle
                        draw_direction_indicator(frame, pt2, smoothed_angle)
                        cv2.putText(frame, f"{smoothed_angle:.1f}", (pt2[0]+10, pt2[1]+10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        log_direction(id_, smoothed_angle)
                current_count = len(current_detections)
                if id_counter > total_count:
                    total_count = id_counter
                cv2.putText(frame, f"Frelons: {current_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                fps = 1 / max(0.001, time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                draw_geographic_angle_reference(frame)
            cv2.imshow("Tracking Frelons avec Boussole", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break       
            elif key == ord('m'):
                save_trajectories_to_geojson(trajectories, colors, frame_shape, piege_id=PIEGE_ACTIF)
                # Fusionner les features de tous les pièges
                all_features = {'type': 'FeatureCollection', 'features': []}
                for piege_id in PIEGES:
                    filename = os.path.join(get_piege_path(piege_id), "trajectories.geojson")
                    if os.path.exists(filename):
                        with open(filename) as f:
                            data = json.load(f)
                            if 'features' in data:
                                all_features['features'].extend(data['features'])
                if all_features['features']:
                    map_file = create_folium_map(all_features)
                    if map_file:
                        print(f"Carte générée: {os.path.abspath(map_file)}")
                        webbrowser.open(f'file://{os.path.abspath(map_file)}')
                    else:
                        print("Aucune trajectoire à afficher.")
            elif key == ord('c'):
                calibration_mode = not calibration_mode
                if calibration_mode:
                    calibration_points.clear()
                    print("\n=== CALIBRATION ===")
                    print(f"1. Placez un objet de référence de {REFERENCE_DISTANCE_METERS}m")
                    print("2. Cliquez sur les deux extrémités dans l'image")
                    print("3. Appuyez sur 'c' pour valider\n")
                else:
                    if len(calibration_points) == 2:
                        dx = calibration_points[1][0] - calibration_points[0][0]
                        dy = calibration_points[1][1] - calibration_points[0][1]
                        REFERENCE_DISTANCE_PIXELS = math.sqrt(dx**2 + dy**2)
                        calculate_pixel_to_deg_factor()
                        print("Calibration réussie!")
            elif key == ord('n'):
                print("\n[CHANGEMENT DE PIÈGE]")
                print("Pièges disponibles :")
                for nom, config in PIEGES.items():
                    print(f"- {nom} (t_t={config['t_t']}s)")
                nouveau = input("Entrez le nom du piège : ").strip()
                changer_piege(nouveau)
        except Exception as e:
            print(f"Erreur: {e}")
            continue
finally:
    if trajectories:
        save_trajectories_to_geojson(trajectories, colors, frame_shape, piege_id=PIEGE_ACTIF)
    picam2.stop()
    cv2.destroyAllWindows()
    print("Programme terminé")
