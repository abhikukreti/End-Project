import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1


device = 'cuda' if torch.cuda.is_available() else 'cpu'


mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

face_db = {}  #

root_folder = 'known_faces'  


for person in os.listdir(root_folder):
    person_folder = os.path.join(root_folder, person)
    embeddings = []

    for subfolder in ['original', 'augmented']:
        folder_path = os.path.join(person_folder, subfolder)
        if not os.path.exists(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Image not loaded: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = mtcnn(img_rgb)

            if face is not None:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
                    embeddings.append(embedding[0])
            else:
                print(f"No face found in {img_path}")

    if embeddings:
        
        face_db[person] = np.mean(embeddings, axis=0)
        print(f"Added {person} to database.")


np.save('face_db.npy', face_db)
print("Database save ho gaya: face_db.npy")
