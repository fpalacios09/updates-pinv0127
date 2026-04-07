import serial
import serial.tools.list_ports
import time
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import matplotlib.path as mlpPath
import signal
import sys


ser = None  # Variable global para poder cerrarla desde cualquier parte

def find_cp210x_port():
    """Busca el puerto asociado al adaptador CP210x."""
    ports = serial.tools.list_ports.comports()

    for port in ports:
        print(port.description)

    for port in ports:
        if "CP210" in port.description:
            return port.device
    return None

def close_serial():
    global ser
    if ser and ser.is_open:
        print("\n[debug] Cerrando el puerto serial...")
        ser.close()
        print("Puerto cerrado.")

def signal_handler(sig, frame):
    print("\n[debug] Señal de cierre recibida (Ctrl+C o similar).")
    close_serial()
    sys.exit(0)

# Registra la señal para cerrar el puerto ante Ctrl+C o cierre forzado
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    port = find_cp210x_port()
    if not port:
        print("[debug] No se encontró ningún dispositivo CH341 conectado.")
        sys.exit(1)

    print(f"[debug] Conectando... : {port}")
    ser = serial.Serial(port, baudrate=115200, timeout=1)
    print(f"[debug] Conectado!: {port}")

    #-------------------------------------------------------------------------------------------------

    # Verificar CUDA
    print("")
    print('Disponibilidad de CUDA con torch: ', torch.cuda.is_available())
    print('Disponibilidad de CUDA con cv2: ', cv2.cuda.getCudaEnabledDeviceCount() > 0)
    print("")

    model = YOLO("yolov8n.pt")
    #video_path = '0'  # Cámara
    video_path = 'cars.mp4'
    car_class_id = 2

    tracker_config = {
        'tracker': "bytetrack.yaml",
        'show': False,
        'save': False,
        'save_txt': False,
        'imgsz': 640,
        'conf': 0.4,
        'iou': 0.5,
        'agnostic_nms': True,
        'device': '0',
        'stream': True
    }

    zone = np.array([
        [205, 185],
        [185, 585],
        [1052, 594],
        [1015, 181],
        [219, 176]
    ])

    interval = 5

    def get_bboxes(det):
        if len(det) == 0:
            return None
        xyxy = det.xyxy[0].cpu().numpy()
        class_id = int(det.cls[0].cpu().numpy())
        if class_id == car_class_id:
            return xyxy.astype(int)
        return None

    def get_center(bbox):
        if bbox is None:
            return None, None
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def is_valid_detection(xc, yc):
        return mlpPath.Path(zone).contains_point((xc, yc))

    def get_id(det):
        if det.id is None:
            return None
        return int(det.id[0].cpu().numpy())

    tracked_ids = []
    last_sent = time.time()

    for result in model.track(source=video_path, **tracker_config):
        frame = result.orig_img
        cv2.polylines(frame, pts=[zone], isClosed=True, color=(255, 0, 0), thickness=2)

        detections = 0
        frame_ids = []

        for det in result.boxes:
            bboxes = get_bboxes(det)
            if bboxes is not None:
                xc, yc = get_center(bboxes)
                if is_valid_detection(xc, yc):
                    detections += 1
                    obj_id = get_id(det)
                    if obj_id is not None and obj_id not in frame_ids:
                        frame_ids.append(obj_id)

                cv2.rectangle(frame, (int(bboxes[0]), int(bboxes[1])), (int(bboxes[2]), int(bboxes[3])), (0, 255, 0), 2)
                cv2.circle(frame, center=(xc, yc), radius=5, color=(0, 0, 255), thickness=2)

        print("Cars detected:", detections)
        print("Tracked IDs in this frame:", frame_ids)

        for obj_id in frame_ids:
            if obj_id not in tracked_ids:
                tracked_ids.append(obj_id)

        print("Total unique vehicles detected:", len(tracked_ids))
        print("")

        if time.time() - last_sent > interval:
            if ser and ser.is_open:
                mensaje = len(tracked_ids)
                ser.write(f"{mensaje}\n".encode())
                print(f"[debug] Enviado por serial: {mensaje}")
            last_sent = time.time()

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #-------------------------------------------------------------------------------------------------

except Exception as e:
    print(f"[debug] Error: {e}")

finally:
    close_serial()
    cv2.destroyAllWindows()
