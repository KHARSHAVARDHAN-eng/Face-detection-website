#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
from pymongo import MongoClient
from datasets import load_dataset
from deepface import DeepFace

# Add parent directory to path so we can import database config if needed
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from database.db import db

def parse_args():
    parser = argparse.ArgumentParser(description="Seed ArcFace face embeddings into MongoDB via HF Streaming.")
    parser.add_argument("--limit", type=int, default=1000, help="Target number of unique identities to seed.")
    parser.add_argument("--frames-per-id", type=int, default=5, help="Number of face embeddings to store per identity.")
    parser.add_argument("--dataset", type=str, default="bitmind/lfw", help="Hugging Face dataset name to stream.")
    return parser.parse_args()

def crop_face(img_np):
    """
    Detects and crops the face with alignment using MTCNN to ensure high-quality,
    centered inputs for embedding generation.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend="mtcnn",
            enforce_detection=False,
            align=True
        )
        if faces and len(faces) > 0:
            face_img = faces[0]["face"]
            if face_img.max() <= 1.0:
                face_img = (face_img * 255).astype(np.uint8)
            return face_img
    except Exception as e:
        print(f"Face extraction and alignment in crop_face failed: {e}")
    return img_np


def generate_perturbed_embeddings(face_crop, num_embeddings=15):
    """
    Generates dynamic variations of the face crop simulating real-world conditions
    (reflections, blur, screen artifacts, angles, contrast/brightness shifts)
    and extracts ArcFace embeddings for all of them.
    """
    embeddings = []
    
    variations = []
    # 1. Original
    variations.append(face_crop)
    
    # 2. Horizontal Flip
    variations.append(cv2.flip(face_crop, 1))
    
    # 3. Brightness Shift Up (brighter)
    variations.append(cv2.convertScaleAbs(face_crop, alpha=1.0, beta=25))
    
    # 4. Brightness Shift Down (darker)
    variations.append(cv2.convertScaleAbs(face_crop, alpha=1.0, beta=-25))
    
    # 5. Contrast Up
    variations.append(cv2.convertScaleAbs(face_crop, alpha=1.3, beta=0))
    
    # 6. Contrast Down
    variations.append(cv2.convertScaleAbs(face_crop, alpha=0.7, beta=0))
    
    # 7-10. Rotations (-15, -7, 7, 15 degrees)
    h, w = face_crop.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-15, -7, 7, 15]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variations.append(rotated)
        
    # 11. Gaussian Blur (slight defocus)
    variations.append(cv2.GaussianBlur(face_crop, (3, 3), 0))
    
    # 12. Screen-Noise Simulation (Moiré / sensor noise)
    noise = np.random.normal(0, 10, face_crop.shape).astype(np.int16)
    noisy = np.clip(face_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variations.append(noisy)
    
    # 13. Phone Screen Simulation (glare vignette)
    glare = np.zeros_like(face_crop, dtype=np.uint8)
    cv2.circle(glare, (0, 0), w, (100, 100, 100), -1)
    screen_sim = cv2.addWeighted(face_crop, 0.8, glare, 0.2, 0)
    variations.append(screen_sim)
    
    # 14. JPEG Compression Artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encimg = cv2.imencode('.jpg', face_crop, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    variations.append(decimg)
    
    # 15. Median Blur
    variations.append(cv2.medianBlur(face_crop, 3))
    
    # Truncate or pad to exactly match requested embeddings limit
    while len(variations) < num_embeddings:
        variations.append(face_crop)
    variations = variations[:num_embeddings]
    
    for i, img_var in enumerate(variations):
        try:
            repr_objs = DeepFace.represent(
                img_path=img_var,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )
            if repr_objs and len(repr_objs) > 0:
                emb = np.array(repr_objs[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb.tolist())
            else:
                embeddings.append(np.zeros(512).tolist())
        except Exception as e:
            print(f"Error extracting embedding for variation {i}: {e}")
            embeddings.append(np.zeros(512).tolist())
            
    return embeddings

def main():
    args = parse_args()
    
    print("\n===== PURGING OLD DATABASE AND IMAGES =====")
    # 1. Reset database
    res = db.users.delete_many({})
    print(f"Purged {res.deleted_count} users from MongoDB collection 'users'.")
    
    # 2. Clear upload folder
    upload_dir = os.path.join("uploads", "voxceleb")
    os.makedirs(upload_dir, exist_ok=True)
    for f in os.listdir(upload_dir):
        fp = os.path.join(upload_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)
    print("Cleared uploads/voxceleb/ folder.")

    print(f"\n===== STREAMING DATASET FROM HUGGINGFACE: {args.dataset} =====")
    try:
        ds = load_dataset(args.dataset, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset {args.dataset}: {e}")
        print("Falling back to vilsonrodrigues/lfw ...")
        try:
            ds = load_dataset("vilsonrodrigues/lfw", split="train", streaming=True)
        except Exception as e2:
            print(f"Failed loading fallback dataset: {e2}")
            sys.exit(1)

    print(f"Starting ingestion to seed {args.limit} unique identities...")
    
    seeded_identities = set()
    current_count = 0
    
    iterator = iter(ds)
    
    while current_count < args.limit:
        try:
            print(f"[DEBUG] Fetching next sample from stream (Count: {current_count})...")
            sample = next(iterator)
        except StopIteration:
            print("Reached end of Hugging Face dataset stream.")
            break
            
        # Parse name
        filename = sample.get("filename", "")
        print(f"[DEBUG] Sample filename: '{filename}'")
        if not filename:
            # Fallback for datasets without filename field
            label_val = sample.get("label", current_count)
            name_display = f"speaker_{label_val}"
        else:
            base = filename.split(".")[0]
            parts = base.split("_")
            if parts[-1].isdigit():
                name_display = " ".join(parts[:-1])
            else:
                name_display = base.replace("_", " ")

        print(f"[DEBUG] Identity parsed: '{name_display}'")

        # Skip if we already processed this identity
        if name_display in seeded_identities:
            print(f"[DEBUG] '{name_display}' already seeded. Skipping.")
            continue
            
        # Retrieve and convert image
        print("[DEBUG] Converting image format...")
        pil_img = sample["image"]
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Detect and crop face
        print("[DEBUG] Detecting face via RetinaFace...")
        face_img = crop_face(img_bgr)
        
        # Generate embeddings (multiple per person)
        print(f"[DEBUG] Generating {args.frames_per_id} embeddings...")
        embeddings = generate_perturbed_embeddings(face_img, num_embeddings=args.frames_per_id)
        
        # Calculate master embedding (average and normalize)
        print("[DEBUG] Calculating master embedding...")
        embeddings_np = np.array(embeddings, dtype=np.float32)
        master_emb = np.mean(embeddings_np, axis=0)
        master_norm = np.linalg.norm(master_emb)
        if master_norm > 0:
            master_emb = master_emb / master_norm
            
        # Save face image crop
        safe_name = name_display.replace(" ", "_").lower()
        image_filename = f"{safe_name}.jpg"
        image_path = os.path.join(upload_dir, image_filename)
        cv2.imwrite(image_path, face_img)
        
        # Save user to MongoDB
        print("[DEBUG] Inserting record to MongoDB...")
        user_doc = {
            "name": name_display,
            "embeddings": embeddings,
            "master_embedding": master_emb.tolist(),
            "image_path": image_path,
            "created_at": datetime.utcnow()
        }
        db.users.insert_one(user_doc)
        
        seeded_identities.add(name_display)
        current_count += 1
        
        if current_count % 100 == 0 or current_count == args.limit:
            print(f"[{current_count}/{args.limit}] Seeded identity: '{name_display}'")
            
    print(f"\nIngestion complete! Successfully seeded {current_count} unique identities in MongoDB.")
    
    # Print sample identity IDs/names for verification
    sample_users = list(db.users.find({}, {"name": 1}).limit(10))
    print("Sample identities added:")
    for idx, u in enumerate(sample_users):
        print(f"  {idx + 1}. Name: '{u['name']}', ID: {str(u['_id'])}")

if __name__ == "__main__":
    main()
