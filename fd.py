import cv2
import os
import numpy as np

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images from a folder and return a list of images and labels
def load_images_from_folder(folder, label):
    if not os.path.exists(folder):
        print(f"Error: The folder '{folder}' does not exist.")
        return [], []
    
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                print(f"No faces found in image {img_path}")
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                images.append(face)
                labels.append(label)
        else:
            print(f"Could not read image {img_path}")
    return images, labels

# Paths to the dataset folders (Update these paths if necessary)
dataset_folders = {
    "hero": "D:/face_detect/dataset/hero",
    "heroine": "D:/face_detect/dataset/heroine",
    "heroine_mother": "D:/face_detect/dataset/heroine_mother",
    "heroine_father": "D:/face_detect/dataset/heroine_father",
    "heroine_sister": "D:/face_detect/dataset/heroine_sister",
    "hero_sister": "D:/face_detect/dataset/hero_sister",
    "villain": "D:/face_detect/dataset/villain",
    "villain_2": "D:/face_detect/dataset/villain_2",
}

# Prepare dataset
actors_images = {}
for i, (actor, folder) in enumerate(dataset_folders.items()):
    actors_images[actor] = load_images_from_folder(folder, i)

# Combine all images and labels
images = []
labels = []
for actor, (imgs, lbls) in actors_images.items():
    images.extend(imgs)
    labels.extend(lbls)

# Convert lists to numpy arrays
if len(images) == 0 or len(labels) == 0:
    print("Error: No images or labels were loaded. Please check the dataset paths and images.")
else:
    images = [cv2.resize(img, (100, 100)) for img in images]  # Resize for uniformity
    images = np.array(images)
    labels = np.array(labels)

    # Debugging: Check if the data is loaded correctly
    print(f"Total images: {len(images)}, Total labels: {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Labels distribution: {dict(zip(unique, counts))}")

    # Train a face recognizer (LBPHFaceRecognizer)
    global recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

# Function to recognize faces
def recognize_faces(image):
    global recognizer
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray_image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        label, confidence = recognizer.predict(face_resized)
        
        # Debugging: Print the label and confidence
        print(f"Detected label: {label}, Confidence: {confidence}")
        
        # Map label to actor name
        actor_names = ["hero", "heroine", "heroine mother", "heroine father", "heroine sister", "hero sister", "villain", "villain2"]
        name = "Unknown"
        
        # If confidence is below a threshold, use the label
        threshold = 50  # You can adjust this threshold
        if confidence < threshold and label < len(actor_names):
            name = actor_names[label]

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (25, 255, 25), 2)
        
        # Draw the label with the name below the face
        cv2.rectangle(image, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

    return image

def main():
    global recognizer

    choice = input("Enter 'c' for camera input or 'f' for file input: ").strip().lower()

    if choice == 'c':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            result_image = recognize_faces(frame)
            cv2.imshow('Face Recognition', result_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == 'f':
        file_path = input("Enter the path to the image file: ").strip()
        if not os.path.isfile(file_path):
            print("Error: File not found.")
            return
        
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Could not read image.")
            return
        
        result_image = recognize_faces(image)
        cv2.imshow('Face Recognition', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid choice. Please enter 'c' or 'f'.")

if __name__ == "__main__":
    main()
