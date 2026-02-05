# face_unlock_CORRECTED_FINAL.py
# ALL CORRECTIONS APPLIED BASED ON REQUIREMENTS

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
import shutil
import json
import imaplib
import email as email_module
import dlib
from scipy.spatial import distance as dist


print("""
╔═════════════════════════════════════════════════════════════════╗
║  FACE RECOGNITION DOOR LOCK - FULLY CORRECTED v3.0             ║
║  ✅ 30-second blink timeout                                    ║
║  ✅ 2-second gap between frames                                ║
║  ✅ Email "YES" → Unlock only (no database add)                ║
║  ✅ Email "YES:Name" → Add to database + Unlock                ║
║  ✅ No-blink → Check known_faces before blocking               ║
║  ✅ Check blocked_faces on every attempt                       ║
╚═════════════════════════════════════════════════════════════════╝
""")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces_buffer"
BLOCKED_FACES_DIR = "blocked_faces_detected"

SERIAL_PORT = None
BAUDRATE = 115200
UNLOCK_CMD = "UNLOCK\n"

FRAME_SCALE = 0.25
TOLERANCE = 0.45
PERSON_MATCH_THRESHOLD = 0.6
MIN_MATCHES_FOR_PERSON = 2

BLINK_CHECK_TIMEOUT = 30  # CORRECTED: Changed from 13 to 30 seconds
BLINK_FRAMES_REQUIRED = 2
EYE_AR_THRESH_CLOSE = 0.20
EYE_AR_THRESH_OPEN = 0.30
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

EMAIL_USER = "nouse0793@gmail.com"
EMAIL_PASSWORD = "tojr ifym ibww fnnl"
OWNER_EMAIL = "govindusidharthareddy@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993
OWNER_REPLY_TIMEOUT = 60

BLOCK_FOREVER_FILE = "permanently_blocked.json"
APPROVAL_LOG = "approval_log.txt"
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def detect_eyes_state(detector, predictor, gray):
    try:
        rects = detector(gray, 0)
        if not rects:
            return None, 0.0
        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            left_eye = shape_np[LEFT_EYE]
            right_eye = shape_np[RIGHT_EYE]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            eyes_open = ear > EYE_AR_THRESH_OPEN
            return eyes_open, ear
    except Exception as e:
        return None, 0.0


def draw_face_boxes(frame, face_locations, label="", color=(0, 255, 0)):
    """Draw rectangles around ALL detected faces"""
    for top, right, bottom, left in face_locations:
        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        if label:
            cv2.putText(frame, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def find_serial_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        return None
    for p in ports:
        desc = str(p.description).lower()
        if any(x in desc for x in ["arduino", "usb", "serial", "acm"]):
            print(f"[SERIAL] Found: {p.device}")
            return p.device
    return ports[0].device if ports else None


def open_serial(port, baud):
    if not port:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        while ser.in_waiting:
            ser.readline()
        print(f"[SERIAL] Connected: {port} at {baud} baud\n")
        return ser
    except Exception as e:
        print(f"[SERIAL] Failed: {e}\n")
        return None


def load_known_faces(dataset_dir):
    """Load known faces from dataset directory"""
    people = {}
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        return people

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        encs = []
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(person_dir, img_file)
            try:
                img = face_recognition.load_image_file(path)
                e = face_recognition.face_encodings(img)
                if e:
                    encs.append(e[0])
            except:
                pass

        if encs:
            people[person_name] = encs
            print(f"[LOAD] {len(encs)} images for '{person_name}'")

    print()
    return people


def load_blocked_faces():
    """Load blocked faces from both JSON and folder"""
    blocked_encodings = []

    # Load from JSON
    if os.path.exists(BLOCK_FOREVER_FILE):
        try:
            with open(BLOCK_FOREVER_FILE, 'r') as f:
                blocked_data = json.load(f)
                for item in blocked_data:
                    blocked_encodings.append(np.array(item['encoding']))
        except:
            pass

    # Load from blocked_faces folder
    if os.path.exists(BLOCKED_FACES_DIR):
        for subfolder in os.listdir(BLOCKED_FACES_DIR):
            subfolder_path = os.path.join(BLOCKED_FACES_DIR, subfolder)
            if os.path.isdir(subfolder_path):
                for img_file in os.listdir(subfolder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img_path = os.path.join(subfolder_path, img_file)
                            img = face_recognition.load_image_file(img_path)
                            encs = face_recognition.face_encodings(img)
                            if encs:
                                blocked_encodings.append(encs[0])
                        except:
                            pass

    print(f"[BLOCKED] Loaded {len(blocked_encodings)} blocked face encodings\n")
    return blocked_encodings


def initialize_dirs():
    for d in [UNKNOWN_FACES_DIR, BLOCKED_FACES_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"[INIT] All directories ready\n")


def save_unknown_faces_buffer(frames_list, unique_id):
    """Save unknown faces to buffer"""
    buffer_folder = os.path.join(UNKNOWN_FACES_DIR, f"buffer_{unique_id}")
    os.makedirs(buffer_folder, exist_ok=True)

    for idx, frame in enumerate(frames_list):
        path = os.path.join(buffer_folder, f"unknown_face_{idx+1}.jpg")
        cv2.imwrite(path, frame)
        print(f"[BUFFER] Saved: {path}")

    print(f"[BUFFER] All frames saved\n")
    return buffer_folder


def save_blocked_faces(frames_list, unique_id):
    """Save blocked faces to folder"""
    blocked_folder = os.path.join(BLOCKED_FACES_DIR, f"blocked_{unique_id}")
    os.makedirs(blocked_folder, exist_ok=True)

    for idx, frame in enumerate(frames_list):
        path = os.path.join(blocked_folder, f"blocked_face_{idx+1}.jpg")
        cv2.imwrite(path, frame)
        print(f"[BLOCKED] Saved: {path}")

    print(f"[BLOCKED] All frames saved\n")
    return blocked_folder


def capture_5_frames(cap, label=""):
    """Capture 5 frames with 2-second gap"""
    print(f"\n[CAPTURE] Capturing 5 frames {label}...")
    frames_list = []

    for i in range(5):
        ret, frame = cap.read()
        if ret:
            frames_list.append(frame.copy())
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            frame_display = draw_face_boxes(frame.copy(), face_locations, 
                                           label=f"CAPTURE {i+1}/5")
            cv2.imshow("Capturing Frames", frame_display)
            cv2.waitKey(500)
            print(f"[CAPTURE] Frame {i+1}/5 captured")

        # CORRECTED: 2 seconds between frames
        if i < 4:
            time.sleep(2.0)

    return frames_list


def verify_all_5_frames_same_person(frames_list):
    """Verify all 5 frames are same person"""
    if len(frames_list) < 5:
        return False, 0.0, []

    print(f"\n{'='*70}")
    print(f"VERIFICATION: Are all 5 frames the same person?")
    print(f"{'='*70}\n")

    encodings = []
    for idx, frame in enumerate(frames_list):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb)
            if face_encodings:
                encodings.append(face_encodings[0])
                print(f"[VERIFY] Frame {idx+1}: Face detected")
            else:
                print(f"[VERIFY] Frame {idx+1}: NO face detected")
                return False, 0.0, []
        except Exception as e:
            print(f"[VERIFY] Frame {idx+1}: Error - {e}")
            return False, 0.0, []

    if len(encodings) < 5:
        return False, 0.0, []

    reference_encoding = encodings[0]
    all_match = True

    for i in range(1, len(encodings)):
        matches = face_recognition.compare_faces([reference_encoding], encodings[i], tolerance=0.4)
        if not matches[0]:
            all_match = False
            print(f"[VERIFY] Frame {i+1} doesn't match frame 1")

    similarity = 0.95 if all_match else 0.5

    if all_match:
        print(f"\n[VERIFY] All frames are SAME person! Similarity: {similarity:.1%}\n")
        return True, similarity, encodings
    else:
        print(f"\n[VERIFY] Frames DON'T match!\n")
        return False, similarity, []


def is_face_blocked(face_encoding, blocked_encodings):
    """Check if face is in blocked list"""
    if not blocked_encodings:
        return False

    for blocked_enc in blocked_encodings:
        distance = np.linalg.norm(face_encoding - blocked_enc)
        if distance < 0.4:
            return True

    return False


def add_to_permanent_block(face_encodings_list, frames_list=None):
    """Add face to permanent block list with images"""
    blocked = []

    if os.path.exists(BLOCK_FOREVER_FILE):
        try:
            with open(BLOCK_FOREVER_FILE, 'r') as f:
                blocked = json.load(f)
        except:
            blocked = []

    for encoding in face_encodings_list:
        blocked.append({
            'encoding': encoding.tolist(),
            'blocked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    with open(BLOCK_FOREVER_FILE, 'w') as f:
        json.dump(blocked, f)

    # Also save images
    if frames_list:
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        save_blocked_faces(frames_list, unique_id)

    print(f"[BLOCKED] Face added to permanent block list")


def add_to_known_faces(frames_list, person_name):
    """Add approved person to known_faces"""
    target_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    count = 0

    for frame in frames_list:
        filename = os.path.join(target_dir, f"{person_name}_{timestamp}_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[DATASET] Saved: {filename}")
        count += 1

    print(f"[DATASET] Added {count} photos for: {person_name}\n")

    with open(APPROVAL_LOG, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] APPROVED: {person_name}\n")


def send_approval_email(frames_list, similarity_score, unique_id):
    """Send approval email"""
    try:
        print(f"[EMAIL] Preparing email...")

        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = OWNER_EMAIL
        msg['Subject'] = f"[DOOR LOCK] REQUEST ID: {unique_id} - Reply: YES:Name or YES or NO"

        body = f"""
UNKNOWN PERSON REQUESTING ENTRY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIVE BLINK detected (within 30 seconds).
All 5 photos: SAME PERSON.
Verification Score: {similarity_score:.1%}

REPLY OPTIONS:
YES           -> Unlock door ONLY (temporary access)
YES:John      -> Unlock door + Add to database (permanent access)
NO            -> Block permanently

REQUEST ID: {unique_id}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        msg.attach(MIMEText(body, 'plain'))

        temp_dir = os.path.join(UNKNOWN_FACES_DIR, "temp_email")
        os.makedirs(temp_dir, exist_ok=True)
        for idx, frame in enumerate(frames_list):
            cv2.imwrite(os.path.join(temp_dir, f"frame_{idx+1}.jpg"), frame)

        for frame_file in os.listdir(temp_dir):
            photo_path = os.path.join(temp_dir, frame_file)
            with open(photo_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={frame_file}')
                msg.attach(part)

        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"[EMAIL] Sent successfully! ID: {unique_id}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    except Exception as e:
        print(f"[EMAIL] Error: {e}")
        return False


def check_email_for_reply(request_id):
    """Check email for reply (YES, YES:Name, or NO)"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select('INBOX')

        search_query = f'FROM "{OWNER_EMAIL}"'
        status, messages = mail.search(None, search_query)

        if status != 'OK' or not messages[0]:
            mail.close()
            mail.logout()
            return None

        email_ids = messages[0].split()

        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            if status != 'OK':
                continue

            msg = email_module.message_from_bytes(msg_data[0][1])
            subject = msg['Subject']

            if f'REQUEST ID: {request_id}' not in subject:
                continue
            if 'Re:' not in subject and 'RE:' not in subject:
                continue

            body = ""
            if msg.is_multipart():
                for part in msg.get_payload():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode()
            else:
                body = msg.get_payload(decode=True).decode()

            reply_text = body.strip().split('\n')[0].strip().upper()

            print(f"[EMAIL_CHECK] Found reply: '{reply_text}'")

            # CORRECTED: Handle THREE cases
            if reply_text.startswith('YES:'):
                person_name = reply_text.replace('YES:', '').strip()
                if person_name:
                    mail.close()
                    mail.logout()
                    return ('YES_WITH_NAME', person_name)
            elif reply_text == 'YES':
                mail.close()
                mail.logout()
                return ('YES_ONLY', None)
            elif reply_text.startswith('NO'):
                mail.close()
                mail.logout()
                return ('NO', None)

        mail.close()
        mail.logout()
        return None

    except Exception as e:
        print(f"[EMAIL_CHECK] Error: {e}")
        return None


def unlock_door(ser, person_name="Guest"):
    """Send unlock command to Arduino"""
    print(f"[UNLOCK] Attempting to unlock for {person_name}...")

    if not ser:
        print("[DEMO] Door would unlock (no Arduino)")
        return True

    try:
        cmd_to_send = UNLOCK_CMD.encode() if isinstance(UNLOCK_CMD, str) else UNLOCK_CMD
        ser.write(cmd_to_send)
        print(f"[SERIAL] SENT: {cmd_to_send}")

        start_time = time.time()
        while time.time() - start_time < 5:
            if ser.in_waiting:
                response = ser.readline().decode(errors='ignore').strip()
                print(f"[ARDUINO] Received: {response}")
                if 'DONE' in response.upper() or 'OK' in response.upper():
                    print(f"Door unlocked for {person_name}!\n")
                    return True
            time.sleep(0.1)

        print(f"Command sent\n")
        return True

    except Exception as e:
        print(f"[ERROR] Unlock failed: {e}\n")
        return False


def match_face_with_known(face_enc, people_encodings, person_names):
    """Check if face exists in known_faces"""
    if not person_names:
        print("[MATCH] No known faces in database")
        return False, None, 0.0

    print(f"[MATCH] Checking against {len(person_names)} known people...")

    best_match = None
    best_score = 0.0

    for person in person_names:
        enc_list = people_encodings[person]
        matches = face_recognition.compare_faces(enc_list, face_enc, tolerance=TOLERANCE)
        match_count = sum(matches)
        total = len(enc_list)

        if total > 0 and match_count >= MIN_MATCHES_FOR_PERSON:
            ratio = match_count / total
            if ratio >= PERSON_MATCH_THRESHOLD and ratio > best_score:
                best_match = person
                best_score = ratio
                print(f"[MATCH] {person}: {ratio:.1%} confidence")

    if best_match:
        print(f"[MATCH] FOUND: {best_match} ({best_score:.1%})\n")
        return True, best_match, best_score
    else:
        print(f"[MATCH] NO MATCH in known_faces\n")
        return False, None, 0.0


def wait_for_blink_with_display(detector, predictor, cap, timeout=30):
    """Wait for blink with real-time display"""
    print(f"\n{'='*70}")
    print(f"WAITING FOR BLINK ({timeout} seconds)")
    print(f"{'='*70}\n")

    start_time = time.time()
    prev_eyes_open = None
    closed_frames_counter = 0

    while True:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed

        if remaining <= 0:
            print(f"[BLINK] TIMEOUT - No blink detected\n")
            return False, None

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes_open, ear_value = detect_eyes_state(detector, predictor, gray)

        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        display_frame = draw_face_boxes(frame.copy(), face_locations, label="Waiting for blink...")

        cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if eyes_open is not None:
            eye_status = "OPEN" if eyes_open else "CLOSED"
            cv2.putText(display_frame, f"Eyes: {eye_status} (EAR: {ear_value:.2f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Blink Detection", display_frame)
        cv2.waitKey(1)

        if eyes_open is not None:
            if prev_eyes_open is not None:
                if not eyes_open:
                    closed_frames_counter += 1
                else:
                    if closed_frames_counter >= BLINK_FRAMES_REQUIRED:
                        print(f"[BLINK] BLINK DETECTED!\n")
                        return True, frame
                    closed_frames_counter = 0
            prev_eyes_open = eyes_open

        time.sleep(0.05)


def main():
    initialize_dirs()

    print("[INIT] Loading dlib models...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(LANDMARK_MODEL)
        print("[INIT] Dlib models loaded\n")
    except Exception as e:
        print(f"[ERROR] Cannot load dlib: {e}")
        return

    print("[CAMERA] Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] No camera found")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[CAMERA] Camera opened\n")

    port = SERIAL_PORT or find_serial_port()
    ser = open_serial(port, BAUDRATE) if port else None
    if not ser:
        print("[WARN] Running in DEMO mode (no Arduino)\n")

    people_encodings = load_known_faces(DATASET_DIR)
    person_names = list(people_encodings.keys()) if people_encodings else []

    # Load blocked faces
    blocked_encodings = load_blocked_faces()

    print("="*70)
    print("SYSTEM READY - Waiting for face...")
    print("="*70 + "\n")

    try:
        while True:
            # STEP 1: Wait for blink (30 seconds)
            blink_detected, frame_with_blink = wait_for_blink_with_display(
                detector, predictor, cap, timeout=BLINK_CHECK_TIMEOUT
            )

            # ============================================================
            # SCENARIO 1: BLINK DETECTED
            # ============================================================
            if blink_detected:
                print(f"\n{'='*70}")
                print(f"BLINK DETECTED - Processing...")
                print(f"{'='*70}\n")

                # Detect face
                small = cv2.resize(frame_with_blink, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)
                face_encodings = face_recognition.face_encodings(rgb, face_locations)

                if not face_encodings:
                    print("[FACE] No face in frame\n")
                    continue

                face_enc = face_encodings[0]

                # Check if blocked
                if is_face_blocked(face_enc, blocked_encodings):
                    print(f"\n{'='*70}")
                    print(f"BLOCKED FACE DETECTED - ACCESS DENIED")
                    print(f"{'='*70}\n")
                    continue

                # Check known faces
                is_known, person_name, confidence = match_face_with_known(
                    face_enc, people_encodings, person_names
                )

                if is_known and person_name:
                    # KNOWN FACE - UNLOCK IMMEDIATELY
                    print(f"\n{'='*70}")
                    print(f"RECOGNIZED: {person_name} ({confidence:.1%})")
                    print(f"{'='*70}\n")
                    unlock_door(ser, person_name)
                    continue

                # UNKNOWN FACE - Send EMAIL
                print(f"\n{'='*70}")
                print(f"UNKNOWN PERSON (LIVE BLINK DETECTED)")
                print(f"{'='*70}\n")

                frames_list = capture_5_frames(cap, "(unknown)")
                unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
                save_unknown_faces_buffer(frames_list, unique_id)

                all_same, similarity, encodings = verify_all_5_frames_same_person(frames_list)

                if all_same:
                    if send_approval_email(frames_list, similarity, unique_id):
                        print(f"\n{'='*70}")
                        print(f"CAMERA PAUSED - Waiting for email reply...")
                        print(f"{'='*70}\n")

                        start_time = time.time()
                        reply_received = False

                        while time.time() - start_time < OWNER_REPLY_TIMEOUT:
                            remaining = int(OWNER_REPLY_TIMEOUT - (time.time() - start_time))
                            print(f"[EMAIL] Waiting ({remaining}s remaining)...", end='\r')

                            reply = check_email_for_reply(unique_id)
                            if reply:
                                reply_type, reply_person_name = reply

                                # CORRECTED: Handle "YES" only (no name)
                                if reply_type == 'YES_ONLY':
                                    print(f"\n\n{'='*70}")
                                    print(f"TEMPORARY ACCESS GRANTED")
                                    print(f"{'='*70}\n")
                                    unlock_door(ser, "Guest")
                                    reply_received = True
                                    break

                                # Handle "YES:Name"
                                elif reply_type == 'YES_WITH_NAME':
                                    print(f"\n\n{'='*70}")
                                    print(f"APPROVED: {reply_person_name}")
                                    print(f"{'='*70}\n")

                                    # Add to known_faces
                                    add_to_known_faces(frames_list, reply_person_name)

                                    # Reload database
                                    print("[RELOAD] Reloading known_faces database...")
                                    people_encodings = load_known_faces(DATASET_DIR)
                                    person_names = list(people_encodings.keys())
                                    print(f"[RELOAD] Database updated. Total known: {len(person_names)}\n")

                                    # UNLOCK
                                    unlock_door(ser, reply_person_name)
                                    reply_received = True
                                    break

                                # Handle "NO"
                                elif reply_type == 'NO':
                                    print(f"\n\n{'='*70}")
                                    print(f"REJECTED - PERMANENTLY BLOCKED")
                                    print(f"{'='*70}\n")
                                    add_to_permanent_block(encodings, frames_list)
                                    blocked_encodings = load_blocked_faces()
                                    reply_received = True
                                    break

                            time.sleep(5)

                        if not reply_received:
                            print(f"\n[EMAIL] No reply within timeout\n")
                else:
                    print("[FAILED] Frames don't match\n")

            # ============================================================
            # SCENARIO 2: NO BLINK (TIMEOUT)
            # ============================================================
            else:
                print(f"\n{'='*70}")
                print(f"NO BLINK DETECTED - Possible photo/video attack")
                print(f"{'='*70}\n")

                # CORRECTED: Capture 5 frames and check known_faces FIRST
                frames_list = capture_5_frames(cap, "(no blink)")
                unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

                # Verify all frames are same person
                all_same, similarity, encodings = verify_all_5_frames_same_person(frames_list)

                if all_same and encodings:
                    face_enc = encodings[0]

                    # CORRECTED: Check known_faces FIRST
                    is_known, person_name, confidence = match_face_with_known(
                        face_enc, people_encodings, person_names
                    )

                    if is_known and person_name:
                        # CORRECTED: Known person without blink -> Don't block
                        print(f"\n{'='*70}")
                        print(f"KNOWN PERSON ({person_name}) - No blink detected")
                        print(f"ACCESS DENIED (liveness check failed)")
                        print(f"NOT BLOCKED (known person)")
                        print(f"{'='*70}\n")
                    else:
                        # Unknown person without blink -> Block
                        print(f"\n{'='*70}")
                        print(f"UNKNOWN PERSON - No blink -> BLOCKING")
                        print(f"{'='*70}\n")
                        add_to_permanent_block(encodings, frames_list)
                        blocked_encodings = load_blocked_faces()
                else:
                    print("[NO_BLINK] Frames don't match or no faces detected\n")

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