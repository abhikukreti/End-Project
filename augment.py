import cv2
import numpy as np
import os

 
base_path = r"C:\Users\Dell\OneDrive\Desktop\end project\known_Faces\Akansha"
input_dir = os.path.join(base_path, "original")  

output_dir = os.path.join(base_path, "augmented")  

base_path = r"C:\Users\Dell\OneDrive\Desktop\end project\known_Faces\Nupur"
input_dir = os.path.join(base_path, "original")  

output_dir = os.path.join(base_path, "augmented")  


os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def augment_image(img_path, output_dir):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    
   
    flipped = cv2.flip(img, 1)
    cv2.imwrite(f"{output_dir}/{name}_flip{ext}", flipped)
    
    
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(f"{output_dir}/{name}_rot15{ext}", rotated)
    
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(f"{output_dir}/{name}_rot-15{ext}", rotated)
    
    
    bright = cv2.convertScaleAbs(img, beta=40)
    cv2.imwrite(f"{output_dir}/{name}_bright{ext}", bright)
    dark = cv2.convertScaleAbs(img, beta=-40)
    cv2.imwrite(f"{output_dir}/{name}_dark{ext}", dark)
    
    
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    cv2.imwrite(f"{output_dir}/{name}_blur{ext}", blurred)


for img_file in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_file)
    augment_image(img_path, output_dir)
    print(f"Augmented: {img_file}")

print("✅ All augmentations done! Check the 'augmented' folder.")
