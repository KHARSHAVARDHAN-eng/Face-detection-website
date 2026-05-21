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


# EXTRACT REAL ARCFACE EMBEDDING
def extract_face_embedding(image_bytes: bytes) -> list:

    try:

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        if img is None:

            raise FaceRecognitionError(
                "No valid face detected"
            )

        # REAL ARCFACE
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )

        if (
            not embedding_objs or
            len(embedding_objs) == 0
        ):

            raise FaceRecognitionError(
                "No valid face detected"
            )

        embedding = np.array(
            embedding_objs[0]["embedding"],
            dtype=np.float32
        )

        # NORMALIZE EMBEDDING
        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        print(
            f"Embedding extracted successfully "
            f"({len(embedding)} dimensions)"
        )

        return embedding.tolist()

    except Exception as e:

        print("Embedding extraction error:", e)

        raise FaceRecognitionError(
            "No valid face detected"
        )


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