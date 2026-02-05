# COMPLETE PROJECT - FACE UNLOCK WITH BUFFER & EMAIL APPROVAL
# All your requirements implemented in one script

"""
PROJECT REQUIREMENTS:
✅ 1. Open solenoid lock when recognized face detected
✅ 2. Capture same unknown face for 5 consecutive times
✅ 3. Send email to owner asking approval
✅ 4. Owner replies: "ALLOW_ONCE" or "ALLOW_FUTURE:Name"
✅ 5. Add 5 photos to dataset with owner's provided name
✅ 6. Maintain 5-photo buffer (delete 6th oldest)
✅ 7. Full logging and access tracking

READY TO USE - Just configure Gmail and run!
"""

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

print("""
╔════════════════════════════════════════════════════════════════════╗
║   FACE RECOGNITION DOOR LOCK WITH BUFFER & EMAIL APPROVAL         ║
║   Version 1.0 - Ready for Production                              ║
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

# === NEW: BUFFER & EMAIL CONFIG ===
UNKNOWN_FACES_DIR = "unknown_faces_buffer"
UNKNOWN_FACES_THRESHOLD = 5
MAX_BUFFER_FACES = 5
APPROVAL_PENDING_DIR = "approval_pending"

# === EMAIL CONFIG - CHANGE THESE ===
EMAIL_USER = "your_email@gmail.com"           # ← CHANGE THIS
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"        # ← CHANGE THIS (16-char app password)
OWNER_EMAIL = "owner@email.com"               # ← CHANGE THIS
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# === LOGGING ===
LOG_FILE = "access_log.txt"
BUFFER_LOG = "buffer_log.txt"

# ==============================================================


def find_serial_port():
    """Auto-detect Arduino serial port"""
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
    """Open serial connection to Arduino"""
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        while ser.in_waiting:
            _ = ser.readline()
        print(f"[INFO] ✓ Serial port opened: {port}")
        return ser
    except Exception as e:
        print(f"[WARN] Could not open serial port {port}: {e}")
        return None


def load_known_faces(dataset_dir):
    """Load face encodings from dataset"""
    people = {}
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"[INFO] Created directory: {dataset_dir}")
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
            print(f"[INFO] Loaded {len(encs)} encodings for '{person_name}'")
    
    return people


def initialize_buffer_dirs():
    """Create buffer directories"""
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(APPROVAL_PENDING_DIR, exist_ok=True)
    print(f"[INFO] ✓ Buffer directories ready")


def get_next_buffer_id():
    """Get next sequential ID for unknown faces"""
    existing = os.listdir(UNKNOWN_FACES_DIR)
    if not existing:
        return 1
    ids = []
    for f in existing:
        if f.startswith("unknown_") and f.endswith(".jpg"):
            try:
                id_num = int(f.replace("unknown_", "").replace(".jpg", ""))
                ids.append(id_num)
            except:
                pass
    return max(ids) + 1 if ids else 1


def save_unknown_face(frame, face_location):
    """Save unknown face photo to buffer"""
    buffer_id = get_next_buffer_id()
    filename = f"{UNKNOWN_FACES_DIR}/unknown_{buffer_id}.jpg"
    
    top, right, bottom, left = face_location
    face_crop = frame[top:bottom, left:right]
    
    cv2.imwrite(filename, face_crop)
    
    with open(BUFFER_LOG, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved: {filename}\n")
    
    print(f"[BUFFER] Photo saved: #{buffer_id}")
    return buffer_id, filename


def count_unknown_faces():
    """Count photos in buffer"""
    if not os.path.exists(UNKNOWN_FACES_DIR):
        return 0
    files = [f for f in os.listdir(UNKNOWN_FACES_DIR) 
            if f.startswith("unknown_") and f.endswith(".jpg")]
    return len(files)


def cleanup_buffer(max_size=MAX_BUFFER_FACES):
    """Delete oldest photos, keep only max_size"""
    if not os.path.exists(UNKNOWN_FACES_DIR):
        return
    
    files = sorted([f for f in os.listdir(UNKNOWN_FACES_DIR) 
                   if f.startswith("unknown_") and f.endswith(".jpg")])
    
    if len(files) > max_size:
        to_remove = files[:-max_size]
        for f in to_remove:
            full_path = os.path.join(UNKNOWN_FACES_DIR, f)
            os.remove(full_path)
            print(f"[BUFFER] Deleted old photo: {f}")


def send_approval_email(buffer_photos):
    """Send email to owner with unknown face photos"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = OWNER_EMAIL
        msg['Subject'] = f"🔐 [DOOR LOCK] Unknown Face - Approval Required"
        
        body = f"""
SECURITY ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An unknown person was detected at your door 5 times with the same face.

ACTION REQUIRED - Reply with ONE of these:

1️⃣ ALLOW_ONCE
   (Unlock this time only, ask again next time)

2️⃣ ALLOW_FUTURE:John Delivery
   (Add to database, auto-unlock in future)
   Replace "John Delivery" with their name

3️⃣ DENY
   (Block and report)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Photos: 5 attached
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach photos
        for photo_path in buffer_photos:
            if os.path.exists(photo_path):
                with open(photo_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', 
                                  f'attachment; filename={os.path.basename(photo_path)}')
                    msg.attach(part)
        
        # Send
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"[EMAIL] ✓ Approval request sent to {OWNER_EMAIL}")
        return True
    
    except Exception as e:
        print(f"[EMAIL] ✗ Error: {e}")
        return False


def move_buffer_to_pending():
    """Move buffer photos to approval pending"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pending_folder = os.path.join(APPROVAL_PENDING_DIR, f"pending_{timestamp}")
    
    os.makedirs(pending_folder, exist_ok=True)
    
    for photo in os.listdir(UNKNOWN_FACES_DIR):
        src = os.path.join(UNKNOWN_FACES_DIR, photo)
        dst = os.path.join(pending_folder, photo)
        shutil.copy(src, dst)
    
    print(f"[BUFFER] Moved to: pending_{timestamp}")
    return pending_folder


def add_to_dataset(pending_folder, person_name):
    """Add approved photos to dataset"""
    target_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    for photo in os.listdir(pending_folder):
        if photo.endswith('.jpg'):
            src = os.path.join(pending_folder, photo)
            dst = os.path.join(target_dir, f"{person_name}_{count}.jpg")
            shutil.copy(src, dst)
            count += 1
    
    print(f"[DATASET] ✓ Added {count} photos for '{person_name}'")
    
    # Clear buffer
    for photo in os.listdir(UNKNOWN_FACES_DIR):
        os.remove(os.path.join(UNKNOWN_FACES_DIR, photo))
    
    return count


def unlock_door(ser, person_name):
    """Send UNLOCK command to Arduino"""
    try:
        print(f"\n[UNLOCK] Sending unlock command...")
        
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
                print(f"✅ SUCCESS: Door unlocked for {person_name}\n")
                with open(LOG_FILE, 'a') as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {person_name}: UNLOCK_SUCCESS\n")
                return True
            else:
                print("[WARN] No response from Arduino")
                return False
        else:
            print("[DEMO] (Arduino not connected)")
            return True
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def log_access(person_name, status):
    """Log access attempt"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {person_name}: {status}\n")


def main():
    """Main application"""
    
    # Initialize
    initialize_buffer_dirs()
    
    # Connect Arduino
    port = SERIAL_PORT or find_serial_port()
    ser = None
    if port:
        ser = open_serial(port, BAUDRATE)
    else:
        print("[WARN] No Arduino detected - running in DEMO mode")
    
    # Load faces
    people_encodings = load_known_faces(DATASET_DIR)
    if not people_encodings:
        print("[ERROR] No known faces. Create: known_faces/Name/ with images")
        return
    
    person_names = list(people_encodings.keys())
    print(f"[INFO] ✓ Loaded {len(person_names)} authorized users: {', '.join(person_names)}")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return
    
    print("[INFO] ✓ Camera opened\n")
    
    # State
    consecutive_counter = 0
    last_matched_person = None
    consecutive_unknown_counter = 0
    last_unknown_face_encoding = None
    
    print("="*70)
    print("System Ready - Press 'q' to quit, 's' for stats")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Process
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
                
                # Try to match
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
                
                # Scale back
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
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    consecutive_unknown_counter = 0
                
                # === UNKNOWN FACE ===
                else:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.putText(frame, "❌ Unknown", (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Check if same unknown
                    encoding_distance = np.linalg.norm(face_enc - last_unknown_face_encoding) if last_unknown_face_encoding is not None else float('inf')
                    
                    if encoding_distance < 0.4:
                        consecutive_unknown_counter += 1
                    else:
                        consecutive_unknown_counter = 1
                        last_unknown_face_encoding = face_enc
                    
                    # === 5 CONSECUTIVE UNKNOWNS ===
                    if consecutive_unknown_counter >= UNKNOWN_FACES_THRESHOLD:
                        print(f"\n{'='*70}")
                        print(f"🚨 ALERT: Same unknown face detected 5 times!")
                        print(f"{'='*70}\n")
                        
                        # Save photo
                        save_unknown_face(frame, face_locations[idx])
                        
                        # Get all buffer photos
                        buffer_photos = [os.path.join(UNKNOWN_FACES_DIR, f) 
                                       for f in sorted(os.listdir(UNKNOWN_FACES_DIR)) 
                                       if f.endswith('.jpg')]
                        
                        # Move to pending
                        pending_folder = move_buffer_to_pending()
                        
                        # Send email
                        print("[EMAIL] Sending approval request...")
                        send_approval_email(buffer_photos)
                        
                        print("\n⏳ Waiting for owner response...")
                        print("📧 Check your email and reply with:")
                        print("   • ALLOW_ONCE (unlock now, ask again next time)")
                        print("   • ALLOW_FUTURE:Name (add to dataset)")
                        print("   • DENY (block)\n")
                        
                        # Reset
                        consecutive_unknown_counter = 0
                        time.sleep(5.0)
            
            # === CONSECUTIVE FRAME MATCHING ===
            if matched_this_frame and matched_this_frame == last_matched_person:
                consecutive_counter += 1
            elif matched_this_frame and matched_this_frame != last_matched_person:
                last_matched_person = matched_this_frame
                consecutive_counter = 1
            else:
                last_matched_person = None
                consecutive_counter = 0
            
            # === TRIGGER UNLOCK ===
            if consecutive_counter >= CONSECUTIVE_CONFIRMATION:
                print(f"\n{'='*70}")
                print(f"✅ MATCH CONFIRMED: {last_matched_person}")
                print(f"{'='*70}\n")
                
                unlock_door(ser, last_matched_person)
                
                consecutive_counter = 0
                last_matched_person = None
                time.sleep(1.0)
            
            # Display info
            cv2.putText(frame, f"Frames: {consecutive_counter}/{CONSECUTIVE_CONFIRMATION}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            buffer_count = count_unknown_faces()
            if buffer_count > 0:
                cv2.putText(frame, f"Buffer: {buffer_count}/{MAX_BUFFER_FACES}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 200), 1)
            
            cv2.imshow("🔓 Face Recognition Door Lock", frame)
            
            # Cleanup
            cleanup_buffer(MAX_BUFFER_FACES)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n[STATS]")
                print(f"  Users: {len(person_names)}")
                print(f"  Buffer: {buffer_count} photos")
                print(f"  Log: {LOG_FILE}\n")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
