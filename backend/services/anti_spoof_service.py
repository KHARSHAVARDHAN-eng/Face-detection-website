import os
import cv2
import numpy as np
import onnxruntime as ort

class AntiSpoofService:
    def __init__(self):
        # Resolve absolute path to the ONNX model in models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(current_dir), "models", "minifasnet_v2.onnx")
        
        print(f"[ANTISPOOF] Loading liveness model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Liveness detection model not found at: {model_path}")
            
        # Initialize ONNX runtime session on CPU
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print("[ANTISPOOF] Liveness model loaded successfully.")

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
        dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img

    def check_liveness(self, img_bgr, bbox_coords) -> tuple:
        """
        Runs MiniFASNetV2 inference to verify if the face is a live person or spoof.
        bbox_coords: can be [x, y, w, h] or [x1, y1, x2, y2].
        
        Returns:
            (is_real: bool, confidence: float)
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, 0.0
            
        try:
            # Parse bounding box format
            # If coordinates represent [x1, y1, x2, y2]
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
                return False, 0.0

            # Crop the face with a scale of 2.7 (specifically for MiniFASNetV2)
            face_crop = self.crop(img_bgr, bbox_xywh, 2.7, 80, 80)
            
            # Preprocess: BGR to RGB
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Transpose HWC -> CHW
            face_crop_chw = face_crop_rgb.transpose((2, 0, 1))
            
            # Convert to float32 (values in range [0.0, 255.0])
            face_crop_float = face_crop_chw.astype(np.float32)
            
            # Expand dims to [1, 3, 80, 80]
            input_data = np.expand_dims(face_crop_float, axis=0)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_data})
            logits = outputs[0][0] # shape [3]
            
            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            class_id = int(np.argmax(probs))
            is_real = (class_id == 1) # Class 1 is real/live person, 0/2 are fakes
            confidence = float(probs[class_id])
            
            print(f"[ANTISPOOF] Prediction class: {class_id} (is_real: {is_real}), confidence: {confidence:.4f}, probabilities: {probs}")
            return is_real, confidence
            
        except Exception as e:
            print(f"[ANTISPOOF] Error during check_liveness: {e}")
            return False, 0.0

# Singleton instance
anti_spoof_service = None

def get_anti_spoof_service():
    global anti_spoof_service
    if anti_spoof_service is None:
        anti_spoof_service = AntiSpoofService()
    return anti_spoof_service
