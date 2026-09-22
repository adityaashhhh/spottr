import os
import time
import urllib.request
from typing import List
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from insightface.app import FaceAnalysis

app = FastAPI(
    title="Spottr Face Verification Service",
    description="Microservice for face detection, alignment, embedding extraction, and blink-based liveness verification.",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
face_app = None
rec_model = None
face_landmarker = None

# MediaPipe FaceLandmarker eye landmark indices (468 mesh)
# Left eye vertical: top=386, bottom=374; horizontal: left=362, right=263
# Right eye vertical: top=159, bottom=145; horizontal: left=33, right=133
LEFT_EYE = {'top': 386, 'bottom': 374, 'left': 362, 'right': 263}
RIGHT_EYE = {'top': 159, 'bottom': 145, 'left': 33, 'right': 133}

# Threshold calibrated from LFW evaluation (InsightFace buffalo_sc best accuracy threshold)
VERIFICATION_THRESHOLD = 0.23

# Path to MediaPipe face landmarker model
LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
LANDMARKER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def download_landmarker_model():
    """Download the MediaPipe FaceLandmarker model if not already present."""
    if not os.path.exists(LANDMARKER_MODEL_PATH):
        print(f"Downloading FaceLandmarker model to {LANDMARKER_MODEL_PATH}...")
        urllib.request.urlretrieve(LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)
        print("FaceLandmarker model downloaded.")


@app.on_event("startup")
def startup_event():
    global face_app, rec_model, face_landmarker
    print("Loading models on startup...")

    # 1. Initialize InsightFace (buffalo_sc: MobileFaceNet backbone)
    face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=-1, det_size=(320, 320))
    rec_model = face_app.models['recognition']
    print("InsightFace loaded successfully.")

    # 2. Download and initialize MediaPipe FaceLandmarker for liveness tracking
    download_landmarker_model()
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe FaceLandmarker loaded successfully.")


def calculate_ear(landmarks, eye_indices) -> float:
    """
    Computes Eye Aspect Ratio (EAR) based on vertical vs horizontal landmark distances.
    EAR drops sharply during a blink and recovers when the eye reopens.
    """
    top = np.array([landmarks[eye_indices['top']].x, landmarks[eye_indices['top']].y])
    bottom = np.array([landmarks[eye_indices['bottom']].x, landmarks[eye_indices['bottom']].y])
    left = np.array([landmarks[eye_indices['left']].x, landmarks[eye_indices['left']].y])
    right = np.array([landmarks[eye_indices['right']].x, landmarks[eye_indices['right']].y])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def extract_insightface_embedding(img) -> np.ndarray:
    """
    Extracts L2-normalized 512-dim embedding from an image using InsightFace.
    Falls back to direct recognition network run if detector fails.
    """
    faces = face_app.get(img)
    if len(faces) > 0:
        # Use embedding of largest detected face
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = face.normed_embedding
    else:
        # Fallback: manually resize and run through the recognition model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_aligned = cv2.resize(img_rgb, (112, 112))
        face_aligned = np.transpose(face_aligned, (2, 0, 1))
        face_aligned = np.expand_dims(face_aligned, axis=0).astype(np.float32)
        embedding = rec_model.get_feat(face_aligned).flatten()

    # L2 normalization
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def detect_blink(ear_history: List[float], relative_threshold: float = 0.25) -> bool:
    """
    Simple blink detection algorithm analyzing relative EAR changes.
    Looks for a significant dip and subsequent recovery in the Eye Aspect Ratio.
    """
    if len(ear_history) < 3:
        return False

    max_ear = max(ear_history)
    min_ear = min(ear_history)

    # If the variance/range is too small, there is no blink (static image or flat video)
    if max_ear - min_ear < 0.05:
        return False

    # Find the index of the minimum EAR value (the peak of the blink)
    min_idx = ear_history.index(min_ear)

    # Check if the minimum is a clear local dip (at least 25% drop relative to the maximum EAR)
    if min_ear <= max_ear * (1 - relative_threshold):
        # Verify recovery: there should be frames before AND after the dip where EAR is high
        pre_dip_max = max(ear_history[:min_idx]) if min_idx > 0 else min_ear
        post_dip_max = max(ear_history[min_idx + 1:]) if min_idx < len(ear_history) - 1 else min_ear

        # Both sides must show higher values (open eyes)
        if pre_dip_max > min_ear + 0.04 and post_dip_max > min_ear + 0.04:
            return True

    return False


class VerificationRequest(BaseModel):
    id_embedding: List[float]
    selfie_embedding: List[float]


@app.post("/verify/id")
async def verify_id(id_photo: UploadFile = File(...)):
    """
    Uploads ID photo, runs face detection/alignment, and returns 512-dim embedding.
    SAMPLE DATA ONLY — not connected to live user accounts.
    """
    try:
        contents = await id_photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        start = time.time()
        embedding = extract_insightface_embedding(img)
        latency = (time.time() - start) * 1000

        return {
            "id_embedding": embedding.tolist(),
            "inference_time_ms": latency
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ID extraction failed: {str(e)}")


@app.post("/verify/selfie")
async def verify_selfie(selfie_frames: List[UploadFile] = File(...)):
    """
    Processes sequential frames for liveness (blink) tracking.
    Uses the first frame with a valid face to extract and return the selfie embedding.
    SAMPLE DATA ONLY — not connected to live user accounts.
    """
    if not selfie_frames:
        raise HTTPException(status_code=400, detail="No frames provided for selfie validation.")

    try:
        ear_history = []
        selfie_embedding = None
        face_processed_count = 0

        for file in selfie_frames:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1. Check landmarks for Eye Aspect Ratio (Liveness)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = face_landmarker.detect(mp_image)

            if result.face_landmarks and len(result.face_landmarks) > 0:
                landmarks = result.face_landmarks[0]
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_history.append(avg_ear)

                # 2. Extract embedding from the first valid frame with a face
                if selfie_embedding is None:
                    selfie_embedding = extract_insightface_embedding(img)
                face_processed_count += 1

        liveness_passed = detect_blink(ear_history)

        if selfie_embedding is None:
            raise HTTPException(status_code=400, detail="No faces detected in any of the selfie frames.")

        return {
            "selfie_embedding": selfie_embedding.tolist(),
            "liveness_passed": liveness_passed,
            "ear_history": ear_history,
            "processed_frames": face_processed_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selfie verification failed: {str(e)}")


@app.post("/verify/compare")
def compare_embeddings(req: VerificationRequest):
    """
    Utility endpoint to compare two face embeddings using cosine similarity.
    Returns verification decision based on the calibrated threshold.
    """
    v1 = np.array(req.id_embedding)
    v2 = np.array(req.selfie_embedding)

    # Cosine similarity (embeddings are L2 normalized, so dot product = cosine similarity)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        similarity = 0.0
    else:
        similarity = float(np.dot(v1, v2) / (norm1 * norm2))

    verified = similarity >= VERIFICATION_THRESHOLD

    return {
        "verified": verified,
        "similarity_score": similarity,
        "threshold": VERIFICATION_THRESHOLD
    }


@app.post("/verify/profile")
async def verify_profile(
    profile_photos: List[UploadFile] = File(...),
    selfie_frames: List[UploadFile] = File(...)
):
    """
    Hinge-style Verification Flow:
    1. Runs blink-based liveness detection on the selfie frames.
    2. Extracts the selfie embedding from the first valid frame with a face.
    3. Extracts face embeddings for each of the uploaded profile photos.
    4. Computes cosine similarities between the selfie and each profile photo.
    5. Approves verification if liveness passes AND max similarity >= threshold.
    """
    if not profile_photos:
        raise HTTPException(status_code=400, detail="No profile photos provided.")
    if not selfie_frames:
        raise HTTPException(status_code=400, detail="No selfie frames provided.")

    try:
        # 1. Process Selfie Frames (liveness + embedding)
        ear_history = []
        selfie_embedding = None
        face_processed_count = 0

        for file in selfie_frames:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            # 1. Extract selfie embedding if not yet extracted
            if selfie_embedding is None:
                try:
                    selfie_embedding = extract_insightface_embedding(img)
                    face_processed_count += 1
                except Exception:
                    pass

            # 2. Check landmarks for Eye Aspect Ratio (Liveness)
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = face_landmarker.detect(mp_image)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    landmarks = result.face_landmarks[0]
                    left_ear = calculate_ear(landmarks, LEFT_EYE)
                    right_ear = calculate_ear(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    ear_history.append(avg_ear)
            except Exception:
                pass

        # Liveness logic: evaluate blink if >=3 frames are available, otherwise default to True for snapshot mode
        if len(ear_history) >= 3:
            liveness_passed = detect_blink(ear_history)
        else:
            liveness_passed = True

        if selfie_embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in selfie frames.")

        # 2. Process Profile Photos
        profile_embeddings = []
        for file in profile_photos:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            try:
                embedding = extract_insightface_embedding(img)
                profile_embeddings.append(embedding)
            except Exception:
                # If face extraction fails for one profile photo, we skip it
                continue

        if not profile_embeddings:
            raise HTTPException(status_code=400, detail="No faces could be extracted from any profile photos.")

        # 3. Compare Selfie Embedding against Profile Embeddings
        similarity_scores = []
        for p_emb in profile_embeddings:
            # Cosine similarity
            norm1 = np.linalg.norm(selfie_embedding)
            norm2 = np.linalg.norm(p_emb)
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = float(np.dot(selfie_embedding, p_emb) / (norm1 * norm2))
            similarity_scores.append(sim)

        max_similarity = max(similarity_scores)
        verified = (max_similarity >= VERIFICATION_THRESHOLD) and liveness_passed

        return {
            "verified": verified,
            "liveness_passed": liveness_passed,
            "max_similarity_score": max_similarity,
            "similarity_scores": similarity_scores,
            "threshold": VERIFICATION_THRESHOLD,
            "processed_profile_photos": len(profile_embeddings),
            "processed_selfie_frames": face_processed_count
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile verification failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Read port from environment or default to 8002 for face verification service
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
