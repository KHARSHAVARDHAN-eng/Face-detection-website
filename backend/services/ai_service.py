import numpy as np
import cv2

from deepface import DeepFace


class FaceRecognitionError(Exception):
    pass


# COSINE SIMILARITY
def cosine_similarity(a, b):

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    # NORMALIZE
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm > 0:
        a = a / a_norm

    if b_norm > 0:
        b = b / b_norm

    similarity = np.dot(a, b)

    return float(similarity)


def preprocess_face(img):
    """
    Applies LAB-space CLAHE, adaptive gamma correction, bilateral filtering,
    unsharp mask sharpening, and specular reflection inpainting to optimize
    face images for ArcFace representation under real-world conditions.
    """
    if img is None or img.size == 0:
        return img

    try:
        # 1. Resize to a consistent scale
        img = cv2.resize(img, (224, 224))

        # 2. Specular reflection & glare correction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, glare_mask = cv2.threshold(gray, 242, 255, cv2.THRESH_BINARY)
        if np.sum(glare_mask) > 0:
            kernel_glare = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            glare_mask = cv2.dilate(glare_mask, kernel_glare)
            img = cv2.inpaint(img, glare_mask, 3, cv2.INPAINT_TELEA)

        # 3. CLAHE contrast normalization in LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l_chan)
        img = cv2.cvtColor(cv2.merge((cl, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

        # 4. Adaptive Gamma Correction (luminance scaling)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_lum = np.mean(gray)
        target = 120.0
        if mean_lum < 90:
            gamma = max(0.5, mean_lum / target)
        elif mean_lum > 160:
            gamma = min(1.8, mean_lum / target)
        else:
            gamma = 1.0

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        # 5. Denoising via bilateral filter (preserves facial features, smooths noise)
        img = cv2.bilateralFilter(img, d=7, sigmaColor=50, sigmaSpace=50)

        # 6. Mild Sharpening via Unsharp Masking
        blur = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)

        return img
    except Exception as e:
        print("Error in preprocess_face:", e)
        return img


def estimate_image_quality_and_threshold(img) -> float:
    """
    Estimates blur, low light, and screen glare of a face crop to adaptively
    return a similarity threshold, reducing false negatives in real-world scenarios.
    """
    if img is None or img.size == 0:
        return 0.30

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Blur evaluation (Laplacian variance)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_val < 120.0

        # 2. Lighting evaluation
        mean_brightness = np.mean(gray)
        is_low_light = mean_brightness < 75.0
        is_overexposed = mean_brightness > 210.0

        # 3. Specular highlights/glare density
        _, glare = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(glare == 255) / gray.size
        is_screen_or_glare = glare_ratio > 0.02 or is_overexposed

        threshold = 0.30

        if is_screen_or_glare:
            threshold -= 0.04
            print(f"[ADAPTIVE] Glare/Screen detected (ratio: {glare_ratio:.3f}). Reducing threshold.")
        if is_blurry:
            threshold -= 0.02
            print(f"[ADAPTIVE] Blur detected (var: {blur_val:.1f}). Reducing threshold.")
        if is_low_light:
            threshold -= 0.02
            print(f"[ADAPTIVE] Low light detected (mean: {mean_brightness:.1f}). Reducing threshold.")

        threshold = max(0.22, min(0.35, threshold))
        print(f"[ADAPTIVE] Final threshold: {threshold:.3f} (Blur: {blur_val:.1f}, Brightness: {mean_brightness:.1f})")
        return threshold
    except Exception as e:
        print("Error in estimate_image_quality_and_threshold:", e)
        return 0.30


# EXTRACT REAL ARCFACE EMBEDDING
def extract_face_embedding(image_bytes: bytes) -> list:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise FaceRecognitionError("No valid face detected")

        # Extract, align, and crop the face using MTCNN
        extracted = DeepFace.extract_faces(
            img_path=img,
            detector_backend="mtcnn",
            enforce_detection=False,
            align=True
        )
        if not extracted or len(extracted) == 0:
            raise FaceRecognitionError("No valid face detected")

        face_raw = extracted[0]["face"]
        if face_raw.max() <= 1.0:
            face_raw = (face_raw * 255).astype(np.uint8)

        # Apply preprocessing pipeline
        preprocessed = preprocess_face(face_raw)

        # Extract ArcFace embedding directly on the preprocessed face crop
        embedding_objs = DeepFace.represent(
            img_path=preprocessed,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False
        )

        if not embedding_objs or len(embedding_objs) == 0:
            raise FaceRecognitionError("No valid face detected")

        embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        print(f"Embedding extracted successfully ({len(embedding)} dimensions)")
        return embedding.tolist()

    except Exception as e:
        print("Embedding extraction error:", e)
        raise FaceRecognitionError("No valid face detected")


def extract_multiple_face_embeddings(image_bytes: bytes):
    """
    Decodes the image bytes, runs DeepFace.represent to extract all faces,
    preprocesses each, and returns a list of embedding representations, boxes, and crops.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return [], 0, 0
            
        h_img, w_img = img.shape[:2]
        
        # Extract all aligned faces using MTCNN
        extracted_faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="mtcnn",
            enforce_detection=False,
            align=True
        )
        
        results = []
        if not extracted_faces:
            return [], w_img, h_img
            
        for obj in extracted_faces:
            area = obj.get("facial_area", {})
            if not area:
                continue
                
            x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            
            face_raw = obj["face"]
            if face_raw.max() <= 1.0:
                face_raw = (face_raw * 255).astype(np.uint8)

            # Preprocess the aligned face crop
            preprocessed = preprocess_face(face_raw)
            
            # Generate embedding representation
            embedding_objs = DeepFace.represent(
                img_path=preprocessed,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )
            
            if not embedding_objs or len(embedding_objs) == 0:
                continue
                
            emb = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            results.append({
                "embedding": emb.tolist(),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "face_crop": face_raw
            })
            
        return results, w_img, h_img
        
    except Exception as e:
        print("Error in extract_multiple_face_embeddings:", e)
        return [], 0, 0


# GLOBAL BEST MATCH SEARCH
def compare_faces(
    target_embedding,
    database_embeddings,
    threshold=0.70
):

    """
    database_embeddings format:

    [
        ("harsha", embedding1),
        ("harsha", embedding2),
        ("roshan", embedding1),
    ]
    """

    if len(database_embeddings) == 0:

        print("Database empty")

        return None

    best_name = None
    best_similarity = -1.0

    print("\n===== FACE MATCHING =====")

    for name, emb_vector in database_embeddings:

        try:

            similarity = cosine_similarity(
                target_embedding,
                emb_vector
            )

            print(
                f"User: {name} | "
                f"Similarity: {similarity:.4f}"
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_name = name

        except Exception as e:

            print(
                f"Comparison failed for {name}: {e}"
            )

    print(
        f"\nBEST MATCH: {best_name}"
    )

    print(
        f"BEST SIMILARITY: "
        f"{best_similarity:.4f}"
    )

    print(
        f"THRESHOLD: {threshold}"
    )

    # VALID MATCH
    if best_similarity >= threshold:

        print(
            f"FINAL RESULT: MATCHED -> {best_name}"
        )

        return best_name

    # INVALID PERSON
    print(
        "FINAL RESULT: INVALID PERSON"
    )

    return None 