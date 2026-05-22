import os
import sys
import numpy as np
import cv2

# Add backend directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from services.cdcn_service import get_cdcn_service
from services.liveness_features_service import get_liveness_features_service

def main():
    print("=" * 60)
    print("VERIFYING CDCN & TEMPORAL LIVENESS PIPELINE")
    print("=" * 60)
    
    print("Initializing services...")
    cdcn = get_cdcn_service()
    liveness_tracker = get_liveness_features_service()
    
    # Create a dummy BGR image
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img, (640, 360), 100, (200, 200, 200), -1) # Draw a face-like circle
    
    # Bounding box coords: [x1, y1, x2, y2]
    box = [540, 260, 740, 460]
    
    print("\n--- Test 1: CDCN Liveness Inference ---")
    try:
        label, real_score, spoof_score = cdcn.predict_liveness(img, box)
        print(f"CDCN Output: label='{label}', real_score={real_score:.4f}, spoof_score={spoof_score:.4f}")
        assert label in ["real", "spoof"], "Label must be real or spoof"
        assert 0.0 <= real_score <= 1.0, "Real score must be in [0.0, 1.0]"
        assert 0.0 <= spoof_score <= 1.0, "Spoof score must be in [0.0, 1.0]"
        print(">> CDCN Inference Test PASSED!")
    except Exception as e:
        print(f">> CDCN Inference Test FAILED: {e}")
        sys.exit(1)
        
    print("\n--- Test 2: Stateful MediaPipe Liveness Tracker ---")
    try:
        tracking_info = liveness_tracker.track_and_update(img, box)
        print(f"Tracker Output: {tracking_info}")
        assert "blink_detected" in tracking_info
        assert "head_movements" in tracking_info
        assert "is_static" in tracking_info
        print(">> Liveness Tracker Test PASSED!")
    except Exception as e:
        print(f">> Liveness Tracker Test FAILED: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("ALL SERVICES INTEGRATED & INFERENCE FLOW FUNCTIONAL!")
    print("=" * 60)

if __name__ == "__main__":
    main()
