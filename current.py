# face_unlock_pause_resume.py
# PAUSE CAMERA FOR 1 MINUTE WHILE CHECKING EMAIL
# 
# After email sent:
# - Camera PAUSES (frozen)
# - Only check email for 1 minute
# - If reply found → Process it
# - If no reply → Resume camera normally

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
from datetime import datetime, timedelta
import shutil
import json
import imaplib
import email as email_module

print("""
╔════════════════════════════════════════════════════════════════════╗
║   FACE RECOGNITION DOOR LOCK - PAUSE/RESUME VERSION               ║
║   ✅ Camera pauses while checking email (1 minute)                ║
║   ✅ Resume after timeout or on reply                             ║
║   ✅ Only check email, ignore camera during wait                  ║
╚════════════════════════════════════════════════════════════════════╝
""")

# ======================== CONFIGURATION ========================

DATASET_DIR = "known_faces"
SERIAL_PORT = None
BAUDRATE = 115200
FRAME_SCALE = 0.25
TOLERANCE = 0.45
PERSON_MATCH_THRESHOLD = 0.6
MIN_MATCHES_FOR_PERSON = 2
CONSECUTIVE_CONFIRMATION = 3
UNLOCK_CMD = b"UNLOCK\n"
ARDUINO_READ_TIMEOUT = 10

# === BUFFER CONFIG ===
UNKNOWN_FACES_DIR = "unknown_faces_buffer"
UNKNOWN_FACES_THRESHOLD = 5
APPROVAL_PENDING_DIR = "approval_pending"

# === FRAME TIMING ===
FRAME_CAPTURE_GAP = 2

# === EMAIL CONFIG - CHANGE THESE 3 ===
EMAIL_USER = "nouse0793@gmail.com"           # Your Gmail (system email)
EMAIL_PASSWORD = "tojr ifym ibww fnnl"         # App password or regular
OWNER_EMAIL = "govindusidharthareddy@gmail.com"               # Owner email (where replies come from)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# === TIMING CONFIG ===
OWNER_REPLY_TIMEOUT = 1 * 60  # 1 minute to check for reply
BLOCK_FOREVER_FILE = "permanently_blocked.json"

# === LOGGING ===
LOG_FILE = "access_log.txt"
BUFFER_LOG = "buffer_log.txt"
APPROVAL_LOG = "approval_log.txt"

# ==============================================================


def find_serial_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        return None
    priority = []
    for p in ports:
        d = f"{p.device} {p.description}".lower()
        score = 0
        for term in ("usb", "modem", "acm", "arduino", "serial"):
            if term in d:
                score += 1
        priority.append((score, p.device))
    priority.sort(reverse=True)
    return priority[0][1] if priority else None


def open_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        while ser.in_waiting:
            _ = ser.readline()
        print(f"[SERIAL] ✓ Connected: {port}\n")
        return ser
    except Exception as e:
        print(f"[SERIAL] ✗ Error: {e}\n")
        return None


def load_known_faces(dataset_dir):
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


def initialize_dirs():
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(APPROVAL_PENDING_DIR, exist_ok=True)
    print(f"[INIT] Directories ready\n")


def get_next_buffer_id():
    existing = os.listdir(UNKNOWN_FACES_DIR)
    if not existing:
        return 1
    ids = []
    for f in existing:
        if f.startswith("frame_") and f.endswith(".jpg"):
            try:
                id_num = int(f.replace("frame_", "").replace(".jpg", ""))
                ids.append(id_num)
            except:
                pass
    return max(ids) + 1 if ids else 1


def save_full_frame(frame):
    """Save full frame to buffer"""
    buffer_id = get_next_buffer_id()
    filename = f"{UNKNOWN_FACES_DIR}/frame_{buffer_id}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def count_buffer_frames():
    if not os.path.exists(UNKNOWN_FACES_DIR):
        return 0
    files = [f for f in os.listdir(UNKNOWN_FACES_DIR) 
            if f.startswith("frame_") and f.endswith(".jpg")]
    return len(files)


def get_face_encoding_from_image(image_path):
    """Extract face encoding from image"""
    try:
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            return encodings[0]
    except:
        pass
    return None


def verify_all_5_frames_same_person():
    """Verify all 5 frames are the SAME person"""
    
    print(f"\n{'='*70}")
    print(f"🔍 VERIFICATION: Are all 5 frames the same person?")
    print(f"{'='*70}\n")
    
    buffer_frames = sorted([f for f in os.listdir(UNKNOWN_FACES_DIR) 
                           if f.startswith("frame_") and f.endswith(".jpg")])
    
    if len(buffer_frames) != UNKNOWN_FACES_THRESHOLD:
        return False, 0.0, []
    
    encodings = []
    
    for frame_file in buffer_frames:
        frame_path = os.path.join(UNKNOWN_FACES_DIR, frame_file)
        encoding = get_face_encoding_from_image(frame_path)
        
        if encoding is None:
            return False, 0.0, []
        
        encodings.append(encoding)
        print(f"[VERIFY] ✓ {frame_file}")
    
    reference_encoding = encodings[0]
    distances = []
    all_match = True
    
    for i in range(1, len(encodings)):
        distance = np.linalg.norm(reference_encoding - encodings[i])
        distances.append(distance)
        
        matches = face_recognition.compare_faces([reference_encoding], encodings[i], 
                                                 tolerance=0.4)
        
        if not matches[0]:
            all_match = False
    
    avg_distance = np.mean(distances)
    similarity = 1.0 - avg_distance
    
    if all_match:
        print(f"\n[VERIFY] ✅ All frames are SAME person! Similarity: {similarity:.1%}\n")
        return True, similarity, encodings
    else:
        print(f"\n[VERIFY] ❌ Frames DON'T match!\n")
        return False, similarity, []


def is_permanently_blocked(face_encoding):
    """Check if face is permanently blocked"""
    if not os.path.exists(BLOCK_FOREVER_FILE):
        return False
    
    try:
        with open(BLOCK_FOREVER_FILE, 'r') as f:
            blocked = json.load(f)
    except:
        return False
    
    for blocked_face_data in blocked:
        blocked_encoding = np.array(blocked_face_data['encoding'])
        distance = np.linalg.norm(face_encoding - blocked_encoding)
        
        if distance < 0.4:
            return True
    
    return False


def add_to_permanent_block(face_encodings_list):
    """Add face to permanent block list"""
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


def send_approval_email(buffer_photos, similarity_score, unique_id):
    """Send email with unique request ID"""
    try:
        print(f"\n[EMAIL] Preparing to send...\n")
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = OWNER_EMAIL
        msg['Subject'] = f"[DOOR LOCK] REQUEST ID: {unique_id} - Reply: YES:Name or NO"
        
        body = f"""
UNKNOWN PERSON REQUESTING ENTRY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Same unknown person appeared 5 times.
All 5 photos VERIFIED as SAME PERSON.

Verification Score: {similarity_score:.1%}

WHAT TO REPLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ YES:John Delivery
   → Person allowed in
   
❌ NO
   → Person BLOCKED FOREVER

IMPORTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Reply within 1 MINUTE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUEST ID: {unique_id}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        for photo_path in buffer_photos:
            if os.path.exists(photo_path):
                try:
                    with open(photo_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', 
                                      f'attachment; filename={os.path.basename(photo_path)}')
                        msg.attach(part)
                except:
                    pass
        
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"[EMAIL] ✓ Sent! REQUEST ID: {unique_id}")
        return True
    
    except Exception as e:
        print(f"[EMAIL] ✗ Error: {e}\n")
        return False


def check_email_for_reply_to_current(request_id):
    """Check for reply to current request"""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        mail.select('INBOX')
        
        search_query = f'FROM "{OWNER_EMAIL}"'
        status, messages = mail.search(None, search_query)
        
        if status != 'OK':
            mail.close()
            mail.logout()
            return None
        
        email_ids = messages[0].split()
        
        if not email_ids:
            mail.close()
            mail.logout()
            return None
        
        # Search through emails for one with our REQUEST ID
        for email_id in reversed(email_ids):
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            
            if status != 'OK':
                continue
            
            msg = email_module.message_from_bytes(msg_data[0][1])
            subject = msg['Subject']
            
            # Check if this email is a REPLY to our request
            if f'REQUEST ID: {request_id}' not in subject:
                continue
            
            # Check if it's a reply (Re: in subject)
            if 'Re:' not in subject and 'RE:' not in subject:
                continue
            
            # Get email body
            body = ""
            if msg.is_multipart():
                for part in msg.get_payload():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode()
            else:
                body = msg.get_payload(decode=True).decode()
            
            # Parse reply
            body_lines = body.strip().split('\n')
            reply_text = body_lines[0].strip().upper()
            
            print(f"[EMAIL_CHECK] ✓ Found reply: '{reply_text}'")
            
            # Check if YES or NO
            if reply_text.startswith('YES:'):
                person_name = reply_text.replace('YES:', '').strip()
                
                if not person_name or person_name == 'YES':
                    mail.close()
                    mail.logout()
                    return None
                
                mail.close()
                mail.logout()
                return ('YES', person_name)
            
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


def move_buffer_to_pending():
    """Move buffer to pending"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pending_folder = os.path.join(APPROVAL_PENDING_DIR, f"pending_{timestamp}")
    
    os.makedirs(pending_folder, exist_ok=True)
    
    for photo in os.listdir(UNKNOWN_FACES_DIR):
        src = os.path.join(UNKNOWN_FACES_DIR, photo)
        dst = os.path.join(pending_folder, photo)
        shutil.copy(src, dst)
    
    return pending_folder


def add_to_dataset(pending_folder, person_name):
    """Add approved photos to dataset"""
    target_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    for photo in sorted(os.listdir(pending_folder)):
        if photo.endswith('.jpg'):
            src = os.path.join(pending_folder, photo)
            dst = os.path.join(target_dir, f"{person_name}_{datetime.now().strftime('%H%M%S')}_{count}.jpg")
            shutil.copy(src, dst)
            count += 1
    
    print(f"[DATASET] ✓ Added {count} photos for: {person_name}")
    
    for photo in os.listdir(UNKNOWN_FACES_DIR):
        os.remove(os.path.join(UNKNOWN_FACES_DIR, photo))
    
    with open(APPROVAL_LOG, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] APPROVED: {person_name}\n")


def clear_buffer():
    """Clear buffer"""
    for photo in os.listdir(UNKNOWN_FACES_DIR):
        os.remove(os.path.join(UNKNOWN_FACES_DIR, photo))


def unlock_door(ser, person_name):
    """Unlock door"""
    try:
        print(f"\n[UNLOCK] Unlocking for {person_name}...")
        
        if ser:
            ser.write(UNLOCK_CMD)
            
            t0 = time.time()
            done = False
            
            while time.time() - t0 < ARDUINO_READ_TIMEOUT:
                if ser.in_waiting:
                    line = ser.readline().decode(errors='ignore').strip()
                    if line:
                        print(f"[ARDUINO] {line}")
                    if 'DONE' in line.upper():
                        done = True
                        break
                else:
                    time.sleep(0.05)
            
            if done:
                print(f"✅ Door unlocked!\n")
                return True
        else:
            print("[DEMO] Door would unlock\n")
            return True
    
    except Exception as e:
        print(f"[ERROR] {e}\n")
        return False


def wait_for_email_reply(request_id, current_request_id_holder):
    """
    PAUSE CAMERA FOR 1 MINUTE
    Only check email, don't process camera input
    """
    print(f"\n{'='*70}")
    print(f"⏸️  CAMERA PAUSED - CHECKING EMAIL FOR 1 MINUTE")
    print(f"{'='*70}")
    print(f"[PAUSE] Waiting for reply to REQUEST ID: {request_id}")
    print(f"[PAUSE] Camera is FROZEN - no face processing\n")
    
    email_send_time = datetime.now()
    
    while True:
        elapsed = (datetime.now() - email_send_time).total_seconds()
        remaining = int(OWNER_REPLY_TIMEOUT - elapsed)
        
        print(f"[PAUSE] Checking email... ({remaining}s remaining)", end='\r')
        
        # ONLY CHECK EMAIL - DON'T PROCESS CAMERA
        reply = check_email_for_reply_to_current(request_id)
        
        if reply is not None:
            reply_type, person_name = reply
            print(f"\n[PAUSE] ✓ Email reply received!\n")
            return (reply_type, person_name)
        
        # Check timeout
        if elapsed >= OWNER_REPLY_TIMEOUT:
            print(f"\n[PAUSE] ✗ 1 minute timeout - no reply received")
            print(f"[PAUSE] Resuming normal operation...\n")
            return None
        
        time.sleep(5)  # Check every 5 seconds


def main():
    initialize_dirs()
    
    port = SERIAL_PORT or find_serial_port()
    ser = None
    if port:
        ser = open_serial(port, BAUDRATE)
    else:
        print("[WARN] No Arduino - Demo mode\n")
    
    people_encodings = load_known_faces(DATASET_DIR)
    if not people_encodings:
        print("[ERROR] No faces!")
        return
    
    person_names = list(people_encodings.keys())
    print(f"[AUTHORIZED] {', '.join(person_names)}\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("[CAMERA] ✓ Ready\n")
    
    # State
    consecutive_counter = 0
    last_matched_person = None
    consecutive_unknown_counter = 0
    last_unknown_face_encoding = None
    verification_done = False
    all_5_encodings = []
    pending_folder = None
    current_request_id = None
    last_frame_time = 0
    
    print("="*70)
    print("SYSTEM RUNNING - Press 'q' to quit")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Process frame
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            
            matched_this_frame = None
            
            # === PROCESS EACH FACE ===
            for idx, face_enc in enumerate(face_encodings):
                detected = False
                best_match = None
                best_score = 0
                
                if is_permanently_blocked(face_enc):
                    top, right, bottom, left = face_locations[idx]
                    top = int(top / FRAME_SCALE)
                    right = int(right / FRAME_SCALE)
                    bottom = int(bottom / FRAME_SCALE)
                    left = int(left / FRAME_SCALE)
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.putText(frame, "🚫 BLOCKED", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    consecutive_unknown_counter = 0
                    continue
                
                for person in person_names:
                    enc_list = people_encodings[person]
                    matches = face_recognition.compare_faces(enc_list, face_enc, tolerance=TOLERANCE)
                    match_count = sum(matches)
                    total = len(enc_list)
                    
                    if total == 0:
                        continue
                    
                    ratio = match_count / total
                    
                    if match_count >= MIN_MATCHES_FOR_PERSON and ratio >= PERSON_MATCH_THRESHOLD:
                        if ratio > best_score:
                            best_match = person
                            best_score = ratio
                            detected = True
                
                top, right, bottom, left = face_locations[idx]
                top = int(top / FRAME_SCALE)
                right = int(right / FRAME_SCALE)
                bottom = int(bottom / FRAME_SCALE)
                left = int(left / FRAME_SCALE)
                
                # === RECOGNIZED FACE ===
                if detected and best_match:
                    matched_this_frame = best_match
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                    cv2.putText(frame, f"✓ {best_match}", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    consecutive_unknown_counter = 0
                    verification_done = False
                
                # === UNKNOWN FACE ===
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.putText(frame, "❌ Unknown", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    encoding_distance = np.linalg.norm(face_enc - last_unknown_face_encoding) if last_unknown_face_encoding is not None else float('inf')
                    
                    if encoding_distance < 0.4:
                        consecutive_unknown_counter += 1
                    else:
                        consecutive_unknown_counter = 1
                        last_unknown_face_encoding = face_enc
                        verification_done = False
                    
                    cv2.putText(frame, f"Count: {consecutive_unknown_counter}/{UNKNOWN_FACES_THRESHOLD}", 
                               (left, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                # === SAVE FRAME WITH 2-SECOND GAP ===
                if consecutive_unknown_counter > 0 and consecutive_unknown_counter <= UNKNOWN_FACES_THRESHOLD:
                    if (current_time - last_frame_time) >= FRAME_CAPTURE_GAP:
                        save_full_frame(frame)
                        last_frame_time = current_time
            
            # === CHECK BUFFER ===
            buffer_count = count_buffer_frames()
            
            if buffer_count >= UNKNOWN_FACES_THRESHOLD and not verification_done:
                all_same, similarity, encodings = verify_all_5_frames_same_person()
                
                if all_same:
                    print(f"\n[SUCCESS] Verified! Sending email...\n")
                    
                    all_5_encodings = encodings
                    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
                    current_request_id = unique_id
                    
                    buffer_photos = [os.path.join(UNKNOWN_FACES_DIR, f) 
                                   for f in sorted(os.listdir(UNKNOWN_FACES_DIR)) 
                                   if f.endswith('.jpg')]
                    
                    pending_folder = move_buffer_to_pending()
                    
                    if send_approval_email(buffer_photos, similarity, unique_id):
                        # === PAUSE CAMERA AND WAIT FOR EMAIL ===
                        reply = wait_for_email_reply(unique_id, current_request_id)
                        
                        if reply is not None:
                            reply_type, person_name = reply
                            
                            if reply_type == 'YES':
                                print(f"\n{'='*70}")
                                print(f"✅ APPROVED! Person: {person_name}")
                                print(f"{'='*70}\n")
                                
                                add_to_dataset(pending_folder, person_name)
                                unlock_door(ser, person_name)
                                
                                people_encodings = load_known_faces(DATASET_DIR)
                                person_names = list(people_encodings.keys())
                            
                            elif reply_type == 'NO':
                                print(f"\n{'='*70}")
                                print(f"❌ REJECTED")
                                print(f"{'='*70}\n")
                                
                                add_to_permanent_block(all_5_encodings)
                        
                        # === RESUME NORMAL OPERATION ===
                        print(f"[RESUME] Camera resumed\n")
                        clear_buffer()
                        verification_done = True
                        consecutive_unknown_counter = 0
                        all_5_encodings = []
                        pending_folder = None
                        current_request_id = None
                else:
                    print(f"[FAILED] Clearing...")
                    clear_buffer()
                    consecutive_unknown_counter = 0
                    verification_done = False
            
            # === CONSECUTIVE FRAME MATCHING ===
            if matched_this_frame and matched_this_frame == last_matched_person:
                consecutive_counter += 1
            elif matched_this_frame and matched_this_frame != last_matched_person:
                last_matched_person = matched_this_frame
                consecutive_counter = 1
            else:
                last_matched_person = None
                consecutive_counter = 0
            
            # === UNLOCK ===
            if consecutive_counter >= CONSECUTIVE_CONFIRMATION:
                print(f"\n✅ {last_matched_person} matched!")
                unlock_door(ser, last_matched_person)
                
                consecutive_counter = 0
                last_matched_person = None
                time.sleep(1.0)
            
            # Display
            cv2.putText(frame, f"Frames: {consecutive_counter}/{CONSECUTIVE_CONFIRMATION}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            buffer_count = count_buffer_frames()
            if buffer_count > 0:
                cv2.putText(frame, f"Buffer: {buffer_count}/5", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 200), 1)
            
            cv2.imshow("🔓 Door Lock", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
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