import os
import cv2
import numpy as np
import onnxruntime as ort

class CDCNService:
    """
    CDCN Inference Service using ONNX Runtime.
    Provides real-time liveness checking on cropped facial regions.
    Falls back gracefully to MiniFASNet if the CDCN ONNX weights are not present.
    """
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        self.model_path = os.path.join(backend_dir, "models", "cdcn.onnx")
        self.fallback_model_path = os.path.join(backend_dir, "models", "minifasnet_v2.onnx")
        
        self.session = None
        self.is_cdcn = False
        
        if os.path.exists(self.model_path):
            print(f"\n[CDCN] Loading CDCN model from: {self.model_path}")
            self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.is_cdcn = True
            print("[CDCN] CDCN model loaded successfully.")
        elif os.path.exists(self.fallback_model_path):
            print(f"\n[CDCN] CDCN model not found at {self.model_path}. Falling back to MiniFASNet at {self.fallback_model_path} with CDCN schema.")
            self.session = ort.InferenceSession(self.fallback_model_path, providers=["CPUExecutionProvider"])
            self.is_cdcn = False
            print("[CDCN] Fallback MiniFASNet model loaded successfully.\n")
        else:
            raise FileNotFoundError("No liveness models found. Please place minifasnet_v2.onnx or cdcn.onnx in the backend/models/ directory.")
            
        self.input_name = self.session.get_inputs()[0].name

    def _get_new_box(self, src_w: int, src_h: int, bbox, scale: float):
        """
        Calculates expanded bounding box with scale factor and fits it inside source boundaries.
        bbox format: (x, y, w, h)
        """
        x, y, box_w, box_h = bbox
        
        # Ensure scale doesn't exceed image limits
        scale = min((src_h - 1) / max(1, box_h), min((src_w - 1) / max(1, box_w), scale))
        
        new_width = box_w * scale
        new_height = box_h * scale
        
        center_x, center_y = box_w / 2.0 + x, box_h / 2.0 + y
        left_top_x = center_x - new_width / 2.0
        left_top_y = center_y - new_height / 2.0
        right_bottom_x = center_x + new_width / 2.0
        right_bottom_y = center_y + new_height / 2.0
        
        # Clamp to image boundaries
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1
            
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale: float, out_w: int, out_h: int):
        """
        Crops face area from the original image according to bounding box expansion scale.
        """
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)
        img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
        if img.size == 0:
            return np.zeros((out_h, out_w, 3), dtype=np.uint8)
        dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img

    def predict_liveness(self, img_bgr, bbox_coords) -> tuple:
        """
        Runs CDCN (or fallback MiniFASNet) inference on cropped face region.
        
        Returns:
            (liveness_label: str ("real"|"spoof"), real_score: float, spoof_score: float)
        """
        if img_bgr is None or img_bgr.size == 0:
            return "spoof", 0.0, 1.0
            
        try:
            # Parse bounding box format
            if len(bbox_coords) == 4:
                x1, y1, x2, y2 = bbox_coords
                # Ensure correct sorting of coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                x = x1
                y = y1
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                bbox_xywh = (x, y, w, h)
            else:
                return "spoof", 0.0, 1.0

            if self.is_cdcn:
                # 1. CDCN Model Inference (Image size: 256x256, Crop scale ~1.6, Normalized input)
                face_crop = self.crop(img_bgr, bbox_xywh, 1.6, 256, 256)
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Scale to [0.0, 1.0] and normalize with ImageNet stats
                face_normalized = face_crop_rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                face_normalized = (face_normalized - mean) / std
                
                # Transpose HWC -> CHW and expand to batch [1, 3, 256, 256]
                face_crop_chw = face_normalized.transpose((2, 0, 1))
                input_data = np.expand_dims(face_crop_chw, axis=0)
                
                # Run inference -> output is depth map of shape [1, 1, 32, 32]
                outputs = self.session.run(None, {self.input_name: input_data})
                depth_map = outputs[0][0]
                
                # Liveness score is determined by the mean depth map intensity
                depth_mean = float(np.mean(depth_map))
                
                # Map depth_mean to a [0.0, 1.0] probability using sigmoid scaling
                real_score = 1.0 / (1.0 + np.exp(-depth_mean * 2.0))
                spoof_score = 1.0 - real_score
                liveness_label = "real" if real_score >= 0.50 else "spoof"
                
                print(f"[CDCN] Depth map mean: {depth_mean:.4f} | Real score: {real_score:.4f} | Label: {liveness_label}")
                return liveness_label, real_score, spoof_score
            else:
                # 2. Fallback MiniFASNet Inference (Image size: 80x80, Crop scale 2.7, Raw values [0.0, 255.0])
                face_crop = self.crop(img_bgr, bbox_xywh, 2.7, 80, 80)
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_crop_chw = face_crop_rgb.transpose((2, 0, 1))
                input_data = np.expand_dims(face_crop_chw.astype(np.float32), axis=0)
                
                outputs = self.session.run(None, {self.input_name: input_data})
                logits = outputs[0][0]
                
                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                
                # Class 1 is real/live person, 0 and 2 are spoof/proxy classes
                real_score = float(probs[1])
                spoof_score = 1.0 - real_score
                liveness_label = "real" if real_score >= 0.85 else "spoof"
                
                print(f"[CDCN Fallback MiniFASNet] Real score: {real_score:.4f} | Label: {liveness_label}")
                return liveness_label, real_score, spoof_score

        except Exception as e:
            print(f"[CDCN] Inference error: {e}")
            return "spoof", 0.0, 1.0

# Singleton instance
_cdcn_service = None

def get_cdcn_service():
    global _cdcn_service
    if _cdcn_service is None:
        _cdcn_service = CDCNService()
    return _cdcn_service
