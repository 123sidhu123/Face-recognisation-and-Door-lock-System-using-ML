#!/usr/bin/env python3
# face_unlock_CORRECTED_FINAL.py
# SIMPLIFIED & CORRECTED - EXACT LOGIC FLOW

import os
import time
import cv2
import face_recognition
import serial
import serial.tools.list_ports
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import json
import imaplib
import email as email_module
import dlib
from scipy.spatial import distance as dist

print("""
╔════════════════════════════════════════════════════════════════════════╗
║    FACE RECOGNITION DOOR LOCK - CORRECTED & SIMPLIFIED v7.0           ║
║                                                                        ║
║  LOGIC:                                                                ║
║  1. IF BLINK DETECTED:                                                 ║
║     ├─ Check known_faces → UNLOCK                                      ║
║     ├─ Check blocked_faces → DENY                                      ║
║     └─ Unknown → Email approval                                        ║
║                                                                        ║
║  2. IF NO BLINK:                                                       ║
║     ├─ Face in frame? → Capture 5 frames → BLOCK PERMANENTLY           ║
║     └─ No face? → RESTART                                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# CONFIG
DATASET_DIR = "known_faces"
BLOCKED_FACES_DIR = "blocked_faces"
TEMP_BUFFER_DIR = "temp_buffer"

FRAME_SCALE = 0.25
TOLERANCE = 0.6
PERSON_MATCH_THRESHOLD = 0.5
MIN_MATCHES = 2

# BLINK DETECTION - PROVEN WORKING
BLINK_TIMEOUT = 30
EYE_AR_OPEN = 0.30
EYE_AR_CLOSE = 0.10
CONSECUTIVE_FRAMES = 1

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# EMAIL
EMAIL_USER = "nouse0793@gmail.com"
EMAIL_PASSWORD = "tojr ifym ibww fnnl"
OWNER_EMAIL = "govindusidharthareddy@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
EMAIL_TIMEOUT = 60

BLOCK_FILE = "blocked_faces.json"
SERIAL_PORT = None
BAUDRATE = 115200
UNLOCK_CMD = "UNLOCK\n"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"

# ============================================================
# CORE FUNCTIONS
# ============================================================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_eye_info(detector, predictor, gray):
    """Get eye state"""
    try:
        rects = detector(gray, 0)
        if not rects:
            return None, 0.0

        face = max(rects, key=lambda r: r.area())
        shape = predictor(gray, face)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape_np[LEFT_EYE]
        right_eye = shape_np[RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        eyes_open = ear > EYE_AR_OPEN
        return eyes_open, ear

    except:
        return None, 0.0


def detect_blink(detector, predictor, cap, timeout=30):
    """
    DETECT BLINK - Returns (blink_found, frame)
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: BLINK DETECTION (30 seconds)")
    print(f"Look at camera and blink naturally")
    print(f"{'='*70}\n")

    start_time = time.time()
    prev_eyes_open = None
    closed_counter = 0

    while time.time() - start_time < timeout:
        remaining = timeout - (time.time() - start_time)

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes_open, ear = get_eye_info(detector, predictor, gray)

        # Display
        display = frame.copy()
        cv2.putText(display, f"Time: {remaining:.1f}s", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if eyes_open is not None:
            status = "OPEN" if eyes_open else "CLOSED"
            color = (0, 255, 0) if eyes_open else (0, 0, 255)
            cv2.putText(display, f"Eyes: {status} (EAR: {ear:.3f})", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"Blink: {closed_counter}/{CONSECUTIVE_FRAMES}",
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        cv2.imshow("Blink Detection", display)
        if cv2.waitKey(30) == ord('q'):
            cv2.destroyWindow("Blink Detection")
            return False, None

        # BLINK LOGIC
        if eyes_open is not None:
            if prev_eyes_open is not None:
                if not eyes_open and prev_eyes_open:
                    closed_counter += 1
                    print(f"[BLINK] Eyes closed - {closed_counter}/{CONSECUTIVE_FRAMES}")

                elif eyes_open and closed_counter > 0:
                    if closed_counter >= CONSECUTIVE_FRAMES:
                        print(f"\n✅ BLINK DETECTED!\n")
                        cv2.destroyWindow("Blink Detection")
                        return True, frame
                    closed_counter = 0

            prev_eyes_open = eyes_open

        time.sleep(0.03)

    print(f"\n❌ NO BLINK - TIMEOUT\n")
    cv2.destroyWindow("Blink Detection")
    return False, None


def get_face_from_frame(frame, detector, predictor):
    """Extract face encoding from frame"""
    try:
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        if encodings:
            return True, encodings[0], locations
        else:
            return False, None, locations
    except:
        return False, None, []


def load_known_faces():
    """Load known faces database"""
    known = {}

    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        return known

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        encodings = []
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                img = face_recognition.load_image_file(os.path.join(person_path, img_file))
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.extend(encs)
            except:
                pass

        if encodings:
            known[person] = encodings
            print(f"[LOAD] {len(encodings)} for {person}")

    print()
    return known


def load_blocked_faces():
    """Load blocked encodings"""
    blocked = []

    if os.path.exists(BLOCK_FILE):
        try:
            with open(BLOCK_FILE, 'r') as f:
                blocked = [np.array(b) for b in json.load(f)]
        except:
            pass

    if os.path.exists(BLOCKED_FACES_DIR):
        for folder in os.listdir(BLOCKED_FACES_DIR):
            folder_path = os.path.join(BLOCKED_FACES_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            for img_file in os.listdir(folder_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                try:
                    img = face_recognition.load_image_file(os.path.join(folder_path, img_file))
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        blocked.extend(encs)
                except:
                    pass

    print(f"[BLOCKED] Loaded {len(blocked)} blocked encodings\n")
    return blocked


def is_blocked(face_enc, blocked_list):
    """Check if face is blocked"""
    if not blocked_list:
        return False

    for blocked_enc in blocked_list:
        distance = np.linalg.norm(face_enc - blocked_enc)
        if distance < 0.5:
            return True

    return False


def match_known_face(face_enc, known_faces):
    """Match with known faces - Returns (person_name, score)"""
    if not known_faces:
        return None, 0.0

    best_match = None
    best_score = 0.0

    for person, encs in known_faces.items():
        matches = face_recognition.compare_faces(encs, face_enc, tolerance=TOLERANCE)
        match_count = sum(matches)

        if match_count >= MIN_MATCHES:
            score = match_count / len(encs)
            if score > best_score:
                best_score = score
                best_match = person

    return best_match, best_score


def capture_5_frames_with_gap(cap):
    """Capture 5 frames"""
    print("[CAPTURE] Capturing 5 frames...\n")
    frames = []

    for i in range(5):
        ret, frame = cap.read()
        if ret:
            frames.append(frame.copy())
            print(f"[CAPTURE] Frame {i+1}/5")

            display = frame.copy()
            cv2.putText(display, f"FRAME {i+1}/5", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Capture", display)
            cv2.waitKey(500)

        if i < 4:
            time.sleep(2.0)

    cv2.destroyWindow("Capture")
    print()
    return frames


def verify_same_person(frames):
    """Verify all frames are same person"""
    if len(frames) < 5:
        return False, []

    print("[VERIFY] Checking if all frames are SAME person...\n")

    encodings = []
    for idx, frame in enumerate(frames):
        try:
            face_found, enc, _ = get_face_from_frame(frame, detector, predictor)
            if face_found:
                encodings.append(enc)
                print(f"[VERIFY] Frame {idx+1}: Face found")
            else:
                print(f"[VERIFY] Frame {idx+1}: NO face")
                return False, []
        except:
            return False, []

    if len(encodings) < 5:
        return False, []

    # Compare all to first
    ref = encodings[0]
    all_match = True

    for i in range(1, 5):
        matches = face_recognition.compare_faces([ref], encodings[i], tolerance=0.4)
        if not matches[0]:
            all_match = False
            break

    if all_match:
        print(f"\n✅ All frames SAME person!\n")
        return True, encodings
    else:
        print(f"\n❌ Frames don't match\n")
        return False, []


def save_to_temp_buffer(frames):
    """Save to temp buffer"""
    temp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(TEMP_BUFFER_DIR, temp_id)
    os.makedirs(temp_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(temp_dir, f"frame_{idx+1}.jpg"), frame)

    print(f"[BUFFER] Saved to {temp_dir}\n")
    return temp_id


def save_to_blocked(frames, encodings):
    """Save to blocked faces"""
    blocked_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    blocked_dir = os.path.join(BLOCKED_FACES_DIR, blocked_id)
    os.makedirs(blocked_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(blocked_dir, f"blocked_{idx+1}.jpg"), frame)

    # Save encodings
    blocked_list = []
    if os.path.exists(BLOCK_FILE):
        try:
            with open(BLOCK_FILE, 'r') as f:
                blocked_list = json.load(f)
        except:
            blocked_list = []

    for enc in encodings:
        blocked_list.append(enc.tolist())

    with open(BLOCK_FILE, 'w') as f:
        json.dump(blocked_list, f)

    print(f"[BLOCKED] ✅ Permanently blocked\n")


def send_email_approval(frames, request_id):
    """Send email with 5 frames"""
    try:
        print("[EMAIL] Sending email...\n")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = OWNER_EMAIL
        msg['Subject'] = f"[DOOR LOCK] Unknown - ID: {request_id}"

        body = f"""
UNKNOWN PERSON
==============

REPLY WITH:
- YES           → Allow entry (no save)
- YES:John      → Allow entry + Save as John
- NO            → Block permanently

REQUEST ID: {request_id}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        msg.attach(MIMEText(body, 'plain'))

        # Attach frames
        for idx, frame in enumerate(frames):
            path = f"/tmp/frame_{idx+1}.jpg"
            cv2.imwrite(path, frame)

            with open(path, 'rb') as att:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(att.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename=frame_{idx+1}.jpg')
                msg.attach(part)

        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("[EMAIL] ✅ Sent\n")
        return True

    except Exception as e:
        print(f"[EMAIL] ❌ Error: {e}\n")
        return False


def check_email_reply(request_id):
    """Check email reply"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select('INBOX')

        status, messages = mail.search(None, f'FROM "{OWNER_EMAIL}"')

        if status != 'OK' or not messages[0]:
            return None

        for email_id in reversed(messages[0].split()):
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            if status != 'OK':
                continue

            msg = email_module.message_from_bytes(msg_data[0][1])

            if f"ID: {request_id}" not in msg['Subject']:
                continue

            body = ""
            if msg.is_multipart():
                for part in msg.get_payload():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode()
            else:
                body = msg.get_payload(decode=True).decode()

            reply = body.strip().split('\n')[0].strip().upper()

            mail.close()
            mail.logout()

            if reply.startswith('YES:'):
                name = reply.replace('YES:', '').strip()
                return ('YES_WITH_NAME', name)
            elif reply == 'YES':
                return ('YES_ONLY', None)
            elif reply.startswith('NO'):
                return ('NO', None)

        mail.close()
        mail.logout()
        return None

    except:
        return None


def save_to_known_faces(frames, person_name):
    """Save frames to known_faces"""
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        path = os.path.join(person_dir, f"{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg")
        cv2.imwrite(path, frame)

    print(f"[DATABASE] ✅ Saved {len(frames)} frames for {person_name}\n")


def unlock_door(ser, person_name="Guest"):
    """Unlock door"""
    print(f"\n{'='*70}")
    print(f"🔓 UNLOCKING FOR: {person_name}")
    print(f"{'='*70}\n")

    if ser:
        try:
            ser.write(UNLOCK_CMD.encode())
            print("[SERIAL] ✅ Command sent\n")
        except:
            print("[SERIAL] ⚠️  Error\n")
    else:
        print("[DEMO] Door unlocked\n")


def find_serial_port():
    try:
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if any(x in str(p.description).lower() for x in ["arduino", "usb", "serial"]):
                return p.device
    except:
        pass
    return None


def open_serial_connection(port, baud):
    if not port:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        return ser
    except:
        return None


# MAIN
detector = None
predictor = None

def main():
    global detector, predictor

    # Init
    for d in [DATASET_DIR, BLOCKED_FACES_DIR, TEMP_BUFFER_DIR]:
        os.makedirs(d, exist_ok=True)

    print("[INIT] Loading models...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    print("[INIT] ✅ Models ready\n")

    print("[CAMERA] Opening...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] No camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[CAMERA] ✅ Ready\n")

    ser = open_serial_connection(SERIAL_PORT or find_serial_port(), BAUDRATE)
    if ser:
        print("[SERIAL] ✅ Connected\n")
    else:
        print("[SERIAL] DEMO mode\n")

    known_faces = load_known_faces()
    blocked_faces = load_blocked_faces()

    print("="*70)
    print("✅ SYSTEM READY")
    print("="*70 + "\n")

    try:
        while True:
            # ========== STEP 1: BLINK DETECTION ==========
            blink_found, blink_frame = detect_blink(detector, predictor, cap, BLINK_TIMEOUT)

            if blink_found:
                # ========== BLINK DETECTED - PROCESS 1 ==========
                print(f"\n{'='*70}")
                print(f"BLINK DETECTED - Face Recognition")
                print(f"{'='*70}\n")

                # Extract face
                face_found, face_enc, _ = get_face_from_frame(blink_frame, detector, predictor)

                if not face_found:
                    print("[FACE] ❌ No face\n")
                    continue

                # Check BLOCKED list
                if is_blocked(face_enc, blocked_faces):
                    print(f"\n{'='*70}")
                    print(f"🚫 BLOCKED - ACCESS DENIED")
                    print(f"{'='*70}\n")
                    continue

                # Check KNOWN faces
                person, score = match_known_face(face_enc, known_faces)

                if person and score > PERSON_MATCH_THRESHOLD:
                    print(f"\n{'='*70}")
                    print(f"✅ RECOGNIZED: {person} ({score:.1%})")
                    print(f"{'='*70}")
                    unlock_door(ser, person)
                    continue

                # UNKNOWN - Email approval
                print(f"\n{'='*70}")
                print(f"❓ UNKNOWN PERSON")
                print(f"{'='*70}\n")

                frames = capture_5_frames_with_gap(cap)
                same_person, encodings = verify_same_person(frames)

                if not same_person:
                    print("[ERROR] Frames mismatch\n")
                    continue

                # Send email
                temp_id = save_to_temp_buffer(frames)
                if not send_email_approval(frames, temp_id):
                    continue

                # Wait for reply
                print(f"Waiting {EMAIL_TIMEOUT}s for reply...\n")
                start = time.time()
                reply = None

                while time.time() - start < EMAIL_TIMEOUT:
                    remaining = EMAIL_TIMEOUT - (time.time() - start)
                    print(f"[WAIT] {remaining:.0f}s...", end='\r')

                    reply = check_email_reply(temp_id)
                    if reply:
                        break

                    time.sleep(5)

                print()

                if not reply:
                    print("[TIMEOUT] No reply\n")
                    continue

                reply_type, reply_data = reply

                # Handle reply
                if reply_type == 'YES_ONLY':
                    print(f"\n✅ TEMPORARY ACCESS\n")
                    unlock_door(ser, "Guest")
                    import shutil
                    shutil.rmtree(os.path.join(TEMP_BUFFER_DIR, temp_id), ignore_errors=True)

                elif reply_type == 'YES_WITH_NAME':
                    print(f"\n✅ APPROVED: {reply_data}\n")
                    save_to_known_faces(frames, reply_data)
                    known_faces = load_known_faces()
                    unlock_door(ser, reply_data)

                elif reply_type == 'NO':
                    print(f"\n🚫 BLOCKED\n")
                    save_to_blocked(frames, encodings)
                    blocked_faces = load_blocked_faces()

            else:
                # ========== NO BLINK - PROCESS 2 ==========
                print(f"\n{'='*70}")
                print(f"NO BLINK - Checking for face")
                print(f"{'='*70}\n")

                # Get current frame
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] No frame\n")
                    continue

                # Check if face exists
                face_found, face_enc, _ = get_face_from_frame(frame, detector, predictor)

                if face_found:
                    # Face found -> Capture & Block
                    print("[FACE] Found without blink - Capturing 5 frames\n")

                    frames = capture_5_frames_with_gap(cap)
                    same_person, encodings = verify_same_person(frames)

                    if same_person:
                        print("[ACTION] Blocking permanently\n")
                        save_to_blocked(frames, encodings)
                        blocked_faces = load_blocked_faces()
                    else:
                        print("[ERROR] Frames mismatch\n")

                else:
                    # No face -> Restart
                    print("[FACE] No face detected - Restarting\n")

    except KeyboardInterrupt:
        print("\n[EXIT]")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        print("[SHUTDOWN]")


if __name__ == "__main__":
    main()