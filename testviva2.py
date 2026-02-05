#!/usr/bin/env python3
# face_unlock_FINAL_v10.py
# 
# LIBERAL BLINK DETECTION - Actually detects real blinking


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
║    FACE RECOGNITION DOOR LOCK - FINAL WORKING v10.0                  ║
║                                                                        ║
║  ✅ LIBERAL BLINK DETECTION (actually detects blinking)              ║
║  ✅ IMMEDIATE known_faces CHECK                                       ║
║  ✅ EMAIL APPROVAL for unknown faces                                  ║
║  ✅ PHOTO DETECTION                                                   ║
║  ✅ BLOCKED_FACES CHECK                                               ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")


# CONFIG
DATASET_DIR = "known_faces"
BLOCKED_FACES_DIR = "blocked_faces"
TEMP_BUFFER_DIR = "temp_buffer"

FRAME_SCALE = 0.25
TOLERANCE = 0.6
PERSON_MATCH_THRESHOLD = 0.7
MIN_MATCHES = 2

# BLINK DETECTION - LIBERAL (actually works)
BLINK_TIMEOUT = 30
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

# ✅ ✅ ✅ UPDATED: SERIAL PORT SET CORRECTLY
SERIAL_PORT = "/dev/cu.usbserial-A5069RR4"
BAUDRATE = 115200
UNLOCK_CMD = "UNLOCK\n"

LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# =========================
# 🔥 MODIFIED FUNCTION ONLY
# =========================
def detect_blink(detector, predictor, cap, timeout=30):
    """
    Shows live face boxes and labels (KNOWN/UNKNOWN/BLOCKED) while waiting for blink.
    """
    # Use the same known/blocked maps that main() prepares
    global known_faces, blocked_faces

    print(f"\n{'='*70}")
    print(f"WAITING FOR BLINK ({timeout} seconds)")
    print(f"Blink naturally - any eye closure will be detected")
    print(f"{'='*70}\n")

    start = time.time()
    ear_history = []
    blink_state = "open"
    blink_detected = False

    inv_scale = 1.0 / FRAME_SCALE  # e.g., 4.0 if FRAME_SCALE is 0.25

    while time.time() - start < timeout:
        remaining = timeout - (time.time() - start)

        ret, frame = cap.read()
        if not ret:
            continue

        # --- Face boxes + labels (on full-res frame) ---
        # We use the same scaling as other parts of your code for encodings
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)

        display = frame.copy()
        cv2.putText(display, f"Time: {remaining:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if encs and locs:
            # Show only the largest face box (like before you used max area face for blink)
            # Compute areas on scaled coords
            areas = []
            for (top, right, bottom, left) in locs:
                w = right - left
                h = bottom - top
                areas.append(w * h)
            idx = int(np.argmax(areas))

            face_enc = encs[idx]
            (top, right, bottom, left) = locs[idx]

            # Recognition status
            person, score = match_known(face_enc, known_faces)
            if is_blocked(face_enc, blocked_faces):
                label = "BLOCKED"
                color = (0, 0, 255)   # red
            elif person and score > PERSON_MATCH_THRESHOLD:
                label = person.upper()
                color = (0, 255, 0)   # green
            else:
                label = "UNKNOWN"
                color = (0, 255, 255) # yellow

            # Map scaled coords back to original frame size
            top = int(top * inv_scale)
            right = int(right * inv_scale)
            bottom = int(bottom * inv_scale)
            left = int(left * inv_scale)

            cv2.rectangle(display, (left, top), (right, bottom), color, 2)
            cv2.putText(display, label, (left, max(20, top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(display, "NO FACE", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Blink detection (same logic as your original) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if rects:
            # Use largest face for landmarks (consistent with original code)
            face = max(rects, key=lambda r: r.area())

            try:
                shape = predictor(gray, face)
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])

                left_eye = shape_np[LEFT_EYE]
                right_eye = shape_np[RIGHT_EYE]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                ear_history.append(ear)
                if len(ear_history) > 50:
                    ear_history.pop(0)

                # LIBERAL thresholds: closed <0.18, open >0.22 (with hysteresis)
                if blink_state == "open" and ear < 0.18:
                    blink_state = "closed"
                    blink_detected = True

                elif blink_state == "closed" and ear > 0.22:
                    if blink_detected:
                        print(f"\n✅ BLINK DETECTED! (EAR: {ear:.3f})\n")
                        cv2.destroyWindow("Blink")
                        return True, frame
                    blink_state = "open"

                color_ear = (0, 0, 255) if ear < 0.18 else (0, 255, 0)
                cv2.putText(display, f"EAR: {ear:.3f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color_ear, 2)
                cv2.putText(display, f"State: {blink_state.upper()}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_ear, 2)

            except Exception:
                pass

        # Show
        cv2.imshow("Blink", display)
        cv2.waitKey(1)

    print(f"\n❌ NO BLINK\n")
    cv2.destroyWindow("Blink")
    return False, None
# =========================
# END MODIFICATION
# =========================


def extract_face(frame, detector, predictor):
    try:
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        if encodings:
            return True, encodings[0], locations
        return False, None, locations
    except:
        return False, None, []


def load_known_faces():
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
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = face_recognition.load_image_file(os.path.join(person_path, img_file))
                    enc = face_recognition.face_encodings(img)
                    if enc:
                        encodings.extend(enc)
                except:
                    pass

        if encodings:
            known[person] = encodings
            print(f"[LOAD] {len(encodings)} images for '{person}'")

    print()
    return known


def load_blocked_faces():
    blocked = []

    if os.path.exists(BLOCK_FILE):
        try:
            with open(BLOCK_FILE, 'r') as f:
                blocked = [np.array(b) for b in json.load(f)]
        except:
            pass

    print(f"[BLOCKED] Loaded {len(blocked)} blocked encodings\n")
    return blocked


def is_blocked(face_enc, blocked_list):
    if not blocked_list:
        return False

    for blocked_enc in blocked_list:
        distance = np.linalg.norm(face_enc - blocked_enc)
        if distance < 0.5:
            return True
    return False


def match_known(face_enc, known_faces):
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


def capture_5_frames(cap):
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
    if len(frames) < 5:
        return False, []

    print("[VERIFY] Checking frames...\n")

    encodings = []
    for idx, frame in enumerate(frames):
        try:
            face_found, enc, _ = extract_face(frame, detector, predictor)
            if face_found:
                encodings.append(enc)
                print(f"[VERIFY] Frame {idx+1}: OK")
            else:
                print(f"[VERIFY] Frame {idx+1}: NO face")
                return False, []
        except:
            return False, []

    if len(encodings) < 5:
        return False, []

    ref = encodings[0]
    for i in range(1, 5):
        matches = face_recognition.compare_faces([ref], encodings[i], tolerance=0.4)
        if not matches[0]:
            print(f"\n❌ Frames don't match\n")
            return False, []

    print(f"\n✅ All frames match!\n")
    return True, encodings


def save_temp(frames):
    temp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = os.path.join(TEMP_BUFFER_DIR, temp_id)
    os.makedirs(temp_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(temp_dir, f"frame_{idx+1}.jpg"), frame)

    print(f"[BUFFER] Saved\n")
    return temp_id


def send_email(frames, request_id):
    try:
        print("[EMAIL] Sending...\n")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = OWNER_EMAIL
        msg['Subject'] = f"[DOOR] Unknown face - ID: {request_id}"

        body = f"""
UNKNOWN PERSON DETECTED

REPLY:
- YES → Allow (temp)
- YES:Name → Allow + Save
- NO → Block

ID: {request_id}
"""
        msg.attach(MIMEText(body, 'plain'))

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
    except:
        print("[EMAIL] ❌ Error\n")
        return False


def check_email(request_id):
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


def save_blocked(frames, encodings):
    blocked_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    blocked_dir = os.path.join(BLOCKED_FACES_DIR, blocked_id)
    os.makedirs(blocked_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(os.path.join(blocked_dir, f"blocked_{idx+1}.jpg"), frame)

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

    print(f"[BLOCKED] ✅ Blocked\n")


def save_known(frames, person_name):
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        path = os.path.join(person_dir, f"{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.jpg")
        cv2.imwrite(path, frame)

    print(f"[DATABASE] ✅ Saved {person_name}\n")


def unlock_door(ser, person_name="Guest"):
    print(f"\n{'='*70}")
    print(f"🔓 UNLOCKING FOR: {person_name}")
    print(f"{'='*70}\n")

    if not ser:
        print("[DEMO] Door unlocked (no serial connected)\n")
        return

    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(b"UNLOCK\n")

        # ✅ Wait for DONE
        start = time.time()
        while time.time() - start < 15:  # plenty of time
            if ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                print("[SERIAL]", line)

                if line == "DONE":
                    print("[SERIAL] ✅ Door cycle completed\n")
                    return

            time.sleep(0.05)

        print("[SERIAL] ⚠ TIMED OUT WAITING FOR DONE")

    except Exception as e:
        print("[SERIAL] ERROR:", e)




def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        return ser
    except:
        return None


detector = None
predictor = None

# ✅ Make these globals so detect_blink can read them, and main() updates them
known_faces = {}
blocked_faces = []


def main():
    global detector, predictor, known_faces, blocked_faces

    for d in [DATASET_DIR, BLOCKED_FACES_DIR, TEMP_BUFFER_DIR]:
        os.makedirs(d, exist_ok=True)

    print("[INIT] Loading dlib...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    print("[INIT] ✅ Ready\n")

    print("[CAMERA] Opening...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] No camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("[CAMERA] ✅ Open\n")

    ser = open_serial(SERIAL_PORT, BAUDRATE)
    if ser:
        print(f"[SERIAL] ✅ Connected → {SERIAL_PORT}\n")
    else:
        print("[SERIAL] ❌ FAILED — Check wiring or port")
        print(f"Trying port: {SERIAL_PORT}\n")

    # ✅ Update the module-level globals so detect_blink() sees correct data
    known_faces = load_known_faces()
    blocked_faces = load_blocked_faces()

    print("="*70)
    print("✅ READY")
    print("="*70 + "\n")

    try:
        while True:
            blink_found, blink_frame = detect_blink(detector, predictor, cap, BLINK_TIMEOUT)

            if blink_found:
                print(f"{'='*70}")
                print(f"BLINK - Live face recognized")
                print(f"{'='*70}\n")

                face_found, face_enc, _ = extract_face(blink_frame, detector, predictor)

                if not face_found:
                    continue

                if is_blocked(face_enc, blocked_faces):
                    print(f"🚫 BLOCKED\n")
                    continue

                person, score = match_known(face_enc, known_faces)

                if person and score > PERSON_MATCH_THRESHOLD:
                    print(f"✅ RECOGNIZED: {person} ({score:.1%})\n")
                    unlock_door(ser, person)
                    continue

                print(f"❓ UNKNOWN\n")

                frames = capture_5_frames(cap)
                same, encs = verify_same_person(frames)

                if not same:
                    continue

                temp_id = save_temp(frames)

                if not send_email(frames, temp_id):
                    continue

                print(f"Waiting for reply...\n")
                start = time.time()

                while time.time() - start < EMAIL_TIMEOUT:
                    reply = check_email(temp_id)
                    if reply:
                        break
                    time.sleep(5)

                if not reply:
                    continue

                reply_type, reply_data = reply

                if reply_type == 'YES_ONLY':
                    unlock_door(ser, "Guest")

                elif reply_type == 'YES_WITH_NAME':
                    save_known(frames, reply_data)
                    known_faces = load_known_faces()
                    unlock_door(ser, reply_data)

                elif reply_type == 'NO':
                    save_blocked(frames, encs)
                    blocked_faces = load_blocked_faces()

            else:
                print(f"{'='*70}")
                print(f"NO BLINK - Photo")
                print(f"{'='*70}\n")

                ret, frame = cap.read()
                if not ret:
                    continue

                face_found, face_enc, _ = extract_face(frame, detector, predictor)

                if face_found:
                    frames = capture_5_frames(cap)
                    same, encs = verify_same_person(frames)

                    if same:
                        person, score = match_known(face_enc, known_faces)

                        if person:
                            print(f"Photo of {person} - REJECT\n")
                        else:
                            print(f"Unknown photo - BLOCK\n")
                            save_blocked(frames, encs)
                            blocked_faces = load_blocked_faces()
                else:
                    print("[NO FACE]\n")

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
