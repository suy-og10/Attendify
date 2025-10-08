import os
import cv2
import numpy as np
import pickle
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import sys
import platform
try:
    import msvcrt  # For Windows console key handling in headless mode
except ImportError:
    msvcrt = None


def create_output_directory(output_dir='recognized_faces'):
    """Creates an output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def load_known_embeddings():
    """Load pre-generated dataset embeddings from known_embeddings.pkl located next to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_file = os.path.join(script_dir, 'known_embeddings.pkl')

    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(
            "Embeddings file not found. Please run generate_embeddings.py first to create known_embeddings.pkl"
        )

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
        known_embeddings = data['embeddings']
        known_names = data['names']

    if len(known_embeddings) == 0:
        raise ValueError("No embeddings found in known_embeddings.pkl")

    print(f"Loaded {len(known_embeddings)} known faces.")
    return np.array(known_embeddings), known_names


def init_embedder(ctx_id=0):
    """Initialize the InsightFace embedder (buffalo_l)."""
    print("Initializing face analysis model (buffalo_l)...")
    app = insightface.app.FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id)
    print("Model initialized.")
    return app


def match_faces(frame, faces, known_embeddings, known_names,
                similarity_threshold=0.5, confidence_threshold=0.65):
    """Match detected faces to known embeddings and draw annotations on the frame."""
    for face in faces:
        try:
            test_emb = face.embedding.reshape(1, -1)
            sims = cosine_similarity(test_emb, known_embeddings)

            best_match_idx = int(np.argmax(sims))
            best_similarity = float(sims[0][best_match_idx])

            valid_matches = [(i, float(s)) for i, s in enumerate(sims[0]) if s >= similarity_threshold]

            if valid_matches:
                valid_matches.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_score = valid_matches[0]
                name = known_names[best_idx]
                confidence = min(1.0, (best_score - similarity_threshold) / (1 - similarity_threshold))
                if confidence < confidence_threshold:
                    name = f"Possible {name.split('_')[0]}"
            else:
                name = "Unknown"
                confidence = 0.0

            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            # Continue processing other faces even if one fails
            print(f"Error processing a detected face: {e}")

    return frame


def main():
    print("Starting real-time face recognition...")

    # Load known embeddings and names
    known_embeddings, known_names = load_known_embeddings()

    # Initialize embedder (use CPU by default: ctx_id=0 tries GPU 0; if that fails, InsightFace usually falls back)
    # On some systems, to force CPU you can try ctx_id=-1
    try:
        embedder = init_embedder(ctx_id=0)
    except Exception:
        print("Falling back to CPU (ctx_id=-1)")
        embedder = init_embedder(ctx_id=-1)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera (index 0). Ensure a camera is connected and not in use.")

    # Warm-up and create output directory for snapshots
    output_dir = create_output_directory()

    print("Press 'q' to quit, 's' to save an annotated snapshot.")

    # Display settings
    headless = False  # will switch to True if cv2.imshow is unavailable
    do_flip = False   # set True to mirror the webcam view horizontally

    # Try to create the window explicitly (helps on some Windows setups)
    try:
        cv2.namedWindow('Real-time Face Recognition', cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print("Unable to create OpenCV window. Will run in headless mode. Reason:")
        print(str(e))
        headless = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera. Exiting...")
            break

        # Optional: you can resize for performance, e.g., to width 960 while maintaining aspect ratio
        # height, width = frame.shape[:2]
        # target_width = 960
        # if width > target_width:
        #     scale = target_width / width
        #     frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Optional flip for mirror-like webcam view
        if do_flip:
            frame = cv2.flip(frame, 1)

        # Detect faces and get embeddings for this frame
        faces = embedder.get(frame)

        # Draw matches/labels
        annotated = match_faces(frame, faces, known_embeddings, known_names)

        if not headless:
            # Try to show frame; if it fails, switch to headless mode
            try:
                cv2.imshow('Real-time Face Recognition', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
                    cv2.imwrite(output_path, annotated)
                    print(f"Saved snapshot to: {os.path.abspath(output_path)}")
            except cv2.error as e:
                print("OpenCV GUI not available. Switching to headless console mode. Reason:")
                print(str(e))
                headless = True

        if headless:
            # In headless mode, print brief status and use console key handling on Windows
            # Show current recognized labels in console at a low rate
            # Note: Without a GUI, we cannot display video; we only log and allow keypresses.
            # Try to capture keys in Windows console via msvcrt
            if platform.system() == 'Windows' and msvcrt is not None:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == 'q':
                        break
                    elif ch.lower() == 's':
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
                        cv2.imwrite(output_path, annotated)
                        print(f"Saved snapshot to: {os.path.abspath(output_path)}")
            else:
                # Fallback: no key capture available; exit on Ctrl+C
                pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print("Stopped real-time face recognition.")


if __name__ == '__main__':
    main()
