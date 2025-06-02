import os
import time
import numpy as np
import cv2
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
import smtplib
from email.message import EmailMessage
from twilio.rest import Client
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)

# Environment variables
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL")
FROM_PASSWORD = os.getenv("FROM_PASSWORD")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
ADMIN_PHONE_NUMBER = os.getenv("ADMIN_PHONE_NUMBER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI setup
openai.api_key = OPENAI_API_KEY

# Constants
EMAIL_COOLDOWN = 300  # seconds
CALL_COOLDOWN = 300   # seconds

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load face database
face_db_path = 'face_db.npy'
face_db = np.load(face_db_path, allow_pickle=True).item() if os.path.exists(face_db_path) else {}

# Alert tracking
last_email_time = 0
last_call_time = 0
unknown_faces_times = []

# --- Utility Functions ---

def generate_smart_message(face_count, time_str):
    prompt = (
        f"A security system detected an unknown person {face_count} times. "
        f"Event captured at {time_str}. Generate a short, professional alert message."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return "Unknown face detected multiple times."

def send_email_alert(photo_path, face_count):
    global last_email_time
    if time.time() - last_email_time < EMAIL_COOLDOWN:
        print("Email cooldown active.")
        return

    try:
        msg = EmailMessage()
        msg['Subject'] = 'Smart Alert: Unknown Face Detected!'
        msg['From'] = FROM_EMAIL
        msg['To'] = ADMIN_EMAIL

        current_time = time.strftime("%I:%M %p", time.localtime())
        smart_message = generate_smart_message(face_count, current_time)
        msg.set_content(smart_message)

        with open(photo_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='unknown_face.jpg')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(FROM_EMAIL, FROM_PASSWORD)
            smtp.send_message(msg)

        print("Email alert sent.")
        last_email_time = time.time()
    except Exception as e:
        print("Email error:", e)

def send_call_alert():
    global last_call_time
    if time.time() - last_call_time < CALL_COOLDOWN:
        print("Call cooldown active.")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.calls.create(
            twiml='<Response><Say voice="alice">Alert! Unknown person is trying to access your system repeatedly. Please check immediately.</Say></Response>',
            to=ADMIN_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        print("Call alert placed.")
        last_call_time = time.time()
    except Exception as e:
        print("Call error:", e)

# --- Routes ---

@app.route('/recognize', methods=['POST'])
def recognize_face():
    global unknown_faces_times

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files['image'].stream).convert('RGB')
    face_tensor = mtcnn(image)

    if face_tensor is None:
        return jsonify({"result": "No face detected"}), 200

    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]

    min_dist = float('inf')
    identity = None
    for person, db_emb in face_db.items():
        dist = np.linalg.norm(db_emb - embedding)
        if dist < min_dist:
            min_dist = dist
            identity = person

    if min_dist < 0.9:
        return jsonify({"result": f"Hello, {identity}", "distance": float(min_dist)})

    # Unknown face handling
    now = time.time()
    unknown_faces_times = [t for t in unknown_faces_times if now - t < 60]
    unknown_faces_times.append(now)

    # Save unknown face
    img_array = face_tensor.permute(1, 2, 0).byte().numpy()
    face_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    photo_path = 'unknown_face.jpg'
    cv2.imwrite(photo_path, face_img)

    # Triggers
    if len(unknown_faces_times) >= 3:
        send_email_alert(photo_path, len(unknown_faces_times))

    if len(unknown_faces_times) >= 5:
        send_call_alert()

    return jsonify({"result": "Unknown face", "distance": float(min_dist)})

# --- Main ---

if __name__ == '__main__':
    app.run(debug=True)
