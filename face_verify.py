import os
import torch
import numpy as np


import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import smtplib
from email.message import EmailMessage

import openai
openai.api_key = "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"

from twilio.rest import Client

app = Flask(__name__)
CORS(app)

ADMIN_EMAIL = "abhikukreti222@gmail.com"
FROM_EMAIL = "abhikukreti222@gmail.com"
FROM_PASSWORD = "onfstwaqytnpeysw"
EMAIL_COOLDOWN = 300
CALL_COOLDOWN = 300

last_email_time = 0
last_call_time = 0

TWILIO_ACCOUNT_SID = 'AC8a2c0d26ab2c022e09a177a2e1594930'
TWILIO_AUTH_TOKEN = '7c1bad10e10b7bc5a4b7101cbfad193f'
TWILIO_PHONE_NUMBER = '+19787218679'
ADMIN_PHONE_NUMBER = '+917983198013'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

face_db_path = 'face_db.npy'
if os.path.exists(face_db_path):
    face_db = np.load(face_db_path, allow_pickle=True).item()
else:
    face_db = {}

unknown_faces_times = []

def generate_smart_message(face_count, time_str):
    try:
        prompt = f"""
        A security system has detected an unknown person {face_count} times at the door camera. 
        The system captured the event at {time_str}. 
        Generate a short, professional, alert-style message for an admin.
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT error:", e)
        return "Unknown face detected multiple times."


def send_email_alert(photo_path, face_count):
    global last_email_time
    if time.time() - last_email_time < EMAIL_COOLDOWN:
        print("Cooldown active. Email not sent.")
        return

    try:
        msg = EmailMessage()
        msg['Subject'] = 'Smart Alert: Unknown Face Detected!'
        msg['From'] = FROM_EMAIL
        msg['To'] = ADMIN_EMAIL

        current_time = time.strftime("%I:%M %p", time.localtime())
        smart_message = generate_smart_message(face_count, current_time)
        print(f"Generated email message: {smart_message}")  # Debug print

        msg.set_content(smart_message)

        with open(photo_path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename='unknown_face.jpg')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(FROM_EMAIL, FROM_PASSWORD)
            smtp.send_message(msg)

        print("Smart alert email sent.")
        last_email_time = time.time()
    except Exception as e:
        print("Error sending email:", e)

def send_call_alert():
    global last_call_time
    if time.time() - last_call_time < CALL_COOLDOWN:
        print("Cooldown active. Call not placed.")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        call = client.calls.create(
            twiml='<Response><Say voice="alice">Alert! Unknown person is trying to access your system repeatedly. Please check immediately.</Say></Response>',
            to=ADMIN_PHONE_NUMBER,
            from_=TWILIO_PHONE_NUMBER
        )
        print("Alert call placed.")
        last_call_time = time.time()
    except Exception as e:
        print("Error placing call:", e)

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

    now = time.time()
    unknown_faces_times.append(now)
    unknown_faces_times = [t for t in unknown_faces_times if now - t < 60]

    img_array = face_tensor.permute(1, 2, 0).byte().numpy()
    face_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    photo_path = 'unknown_face.jpg'
    cv2.imwrite(photo_path, face_img)

    if len(unknown_faces_times) >= 3:
        send_email_alert(photo_path, len(unknown_faces_times))

    if len(unknown_faces_times) >= 5:
        send_call_alert()

    return jsonify({"result": "Unknown face", "distance": float(min_dist)})

if __name__ == '__main__':
    app.run(debug=True)
