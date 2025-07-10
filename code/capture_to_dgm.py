import cv2
import os

# === Config ===
folder_name = "dgm"
face_dir = os.path.abspath(os.path.join("..", "faces", folder_name))
img_prefix = f"{folder_name}_straight_happy_open"
num_images = 10

# === Prepare Directory ===
os.makedirs(face_dir, exist_ok=True)
print(f"[INFO] Saving images to: {face_dir}")

# === Open Camera ===
cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture Faces")

count = 1
while count <= num_images:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture Faces", gray)

    k = cv2.waitKey(1)
    if k % 256 == 32:  # SPACE pressed
        pgm_name = f"{img_prefix}_{count}.pgm"
        pgm_path = os.path.join(face_dir, pgm_name)
        cv2.imwrite(pgm_path, gray)
        print(f"[SAVED] {pgm_path}")
        count += 1

cam.release()
cv2.destroyAllWindows()

print("[DONE] All images saved as PGM format.")
