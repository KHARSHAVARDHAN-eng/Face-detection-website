import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LivenessFeaturesService:
    """
    Service to track eye blinking, head pose, and face movement continuity
    silently in the backend using the MediaPipe Tasks Face Landmarker API.
    Uses Intersection over Union (IoU) box matching to track faces across frames.
    """
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        model_path = os.path.join(backend_dir, "models", "face_landmarker.task")
        
        print(f"[LIVENESS] Initializing MediaPipe Face Landmarker from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe Face Landmarker task model not found at {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Stateful tracking sessions
        # key: track_id (int) -> value: dict containing tracking history and states
        self.sessions = {}
        self.next_track_id = 1
        self.session_timeout = 3.0 # seconds
        
        # Standard 3D facial coordinate points for solvePnP
        # Coordinates correspond to standard anthropometric points on a canonical face model
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (index 1)
            (0.0, -330.0, -65.0),        # Chin (index 152)
            (-225.0, 170.0, -135.0),     # Left eye left corner (index 33)
            (225.0, 170.0, -135.0),      # Right eye right corner (index 263)
            (-150.0, -150.0, -125.0),    # Left mouth corner (index 61)
            (150.0, -150.0, -125.0)      # Right mouth corner (index 291)
        ], dtype=np.float32)

    def _calculate_iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        denom = float(boxAArea + boxBArea - interArea)
        if denom == 0:
            return 0.0
        return interArea / denom

    def _calculate_ear(self, landmarks, eye_indices, w, h):
        """Calculates the Eye Aspect Ratio (EAR) for blink detection."""
        p1 = np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h])
        p2 = np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h])
        p3 = np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h])
        p4 = np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h])
        p5 = np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h])
        p6 = np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h])
        
        dist_v1 = np.linalg.norm(p2 - p6)
        dist_v2 = np.linalg.norm(p3 - p5)
        dist_h = np.linalg.norm(p1 - p4)
        
        if dist_h == 0:
            return 0.0
        return (dist_v1 + dist_v2) / (2.0 * dist_h)

    def _prune_sessions(self, current_time):
        """Cleans up inactive sessions to avoid memory leakage."""
        dead_ids = [tid for tid, s in self.sessions.items() if (current_time - s["last_seen"]) > self.session_timeout]
        for tid in dead_ids:
            del self.sessions[tid]

    def track_and_update(self, img_bgr, bbox_coords) -> dict:
        """
        Runs MediaPipe Face Mesh on frame, associates landmarks to bounding box, 
        and updates eye blinking, head pose, and static-face flags.
        
        bbox_coords: [x1, y1, x2, y2]
        """
        current_time = time.time()
        self._prune_sessions(current_time)
        
        h_img, w_img = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect(mp_image)
        
        default_tracking_info = {
            "blink_detected": False,
            "head_movements": {"left": False, "right": False, "up": False, "down": False},
            "is_static": False,
            "track_id": None
        }
        
        if not results.face_landmarks:
            # No face mesh landmarks found
            return default_tracking_info
            
        best_match_id = None
        best_iou = -1.0
        matched_landmarks = None
        mesh_box = None
        
        # Process each detected face mesh and find the one that overlaps best with the query bounding box
        for face_lms in results.face_landmarks:
            # Compute bounding box of this mesh
            xs = [lm.x * w_img for lm in face_lms]
            ys = [lm.y * h_img for lm in face_lms]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            candidate_box = [x1, y1, x2, y2]
            
            iou = self._calculate_iou(bbox_coords, candidate_box)
            if iou > best_iou:
                best_iou = iou
                matched_landmarks = face_lms
                mesh_box = candidate_box
                
        # We need a minimal overlap to associate with the requested face
        if best_iou < 0.20 or matched_landmarks is None:
            return default_tracking_info
            
        # Match with stateful tracking sessions
        for tid, sess in self.sessions.items():
            sess_iou = self._calculate_iou(mesh_box, sess["last_box"])
            if sess_iou > 0.35:
                best_match_id = tid
                break
                
        # If no matching session is found, instantiate a new one
        if best_match_id is None:
            best_match_id = self.next_track_id
            self.next_track_id += 1
            self.sessions[best_match_id] = {
                "track_id": best_match_id,
                "last_box": mesh_box,
                "last_seen": current_time,
                "blink_detected": False,
                "blink_in_progress": False,
                "head_movements": {"left": False, "right": False, "up": False, "down": False},
                "landmark_history": [],
                "static_frames": 0,
                "is_static": False,
                "frame_count": 0
            }
            print(f"[LIVENESS] Created new tracking session #{best_match_id} for box {mesh_box}")
            
        session = self.sessions[best_match_id]
        session["last_box"] = mesh_box
        session["last_seen"] = current_time
        session["frame_count"] += 1
        
        # 1. BLINK DETECTION (Eye Aspect Ratio - EAR)
        # Left eye landmarks: corners (33, 133), top eyelids (160, 158), bottom (144, 153)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        # Right eye landmarks: corners (362, 263), top eyelids (385, 387), bottom (380, 373)
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_ear = self._calculate_ear(matched_landmarks, left_eye_indices, w_img, h_img)
        right_ear = self._calculate_ear(matched_landmarks, right_eye_indices, w_img, h_img)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink transitions
        if avg_ear < 0.20:
            session["blink_in_progress"] = True
        elif avg_ear >= 0.23 and session["blink_in_progress"]:
            session["blink_detected"] = True
            session["blink_in_progress"] = False
            print(f"[LIVENESS] Blink detected for session #{best_match_id}!")
            
        # 2. HEAD POSE TRACKING (solvePnP)
        # Map 2D image points from face mesh landmarks
        image_points = np.array([
            (matched_landmarks[1].x * w_img, matched_landmarks[1].y * h_img),    # Nose tip
            (matched_landmarks[152].x * w_img, matched_landmarks[152].y * h_img),# Chin
            (matched_landmarks[33].x * w_img, matched_landmarks[33].y * h_img),  # Left eye left corner
            (matched_landmarks[263].x * w_img, matched_landmarks[263].y * h_img),# Right eye right corner
            (matched_landmarks[61].x * w_img, matched_landmarks[61].y * h_img),  # Left mouth corner
            (matched_landmarks[291].x * w_img, matched_landmarks[291].y * h_img) # Right mouth corner
        ], dtype=np.float32)
        
        camera_matrix = np.array([
            [w_img, 0, w_img / 2],
            [0, w_img, h_img / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            # Euler angles
            sy = np.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rmat[2,1] , rmat[2,2])
                y = np.arctan2(-rmat[2,0], sy)
                z = np.arctan2(rmat[1,0], rmat[0,0])
            else:
                x = np.arctan2(-rmat[1,2], rmat[1,1])
                y = np.arctan2(-rmat[2,0], sy)
                z = 0
                
            pitch = x * 180.0 / np.pi
            yaw = y * 180.0 / np.pi
            
            # Detect movement thresholds
            if yaw < -12.0:
                session["head_movements"]["left"] = True
            elif yaw > 12.0:
                session["head_movements"]["right"] = True
                
            if pitch > 10.0:
                session["head_movements"]["up"] = True
            elif pitch < -10.0:
                session["head_movements"]["down"] = True
                
        # 3. TEMPORAL LIVENESS / STATIC FACE CHECK
        # Record nose tip position in history to check for absolute lack of movement (printed photo / fake identity)
        nose_tip = (matched_landmarks[1].x, matched_landmarks[1].y)
        session["landmark_history"].append(nose_tip)
        if len(session["landmark_history"]) > 15:
            session["landmark_history"].pop(0)
            
        # We need a minimum of 5 frames of history to evaluate standard deviation
        if len(session["landmark_history"]) >= 5:
            std_x = np.std([pt[0] for pt in session["landmark_history"]])
            std_y = np.std([pt[1] for pt in session["landmark_history"]])
            std_val = std_x + std_y
            
            # If standard deviation of nose position is extremely low (meaning frame is frozen/static face)
            if std_val < 0.0015:
                session["static_frames"] += 1
            else:
                session["static_frames"] = max(0, session["static_frames"] - 1)
                
            # If face is static for 8+ frames, flag as a static spoof attempt
            if session["static_frames"] >= 8:
                session["is_static"] = True
            else:
                session["is_static"] = False
                
        return {
            "blink_detected": session["blink_detected"],
            "head_movements": session["head_movements"].copy(),
            "is_static": session["is_static"],
            "track_id": best_match_id
        }

# Singleton instance
_liveness_features_service = None

def get_liveness_features_service():
    global _liveness_features_service
    if _liveness_features_service is None:
        _liveness_features_service = LivenessFeaturesService()
    return _liveness_features_service
