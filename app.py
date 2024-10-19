import os
import cv2
import mediapipe as mp

# Initialize the MediaPipe Face Detection and Drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Function to detect faces and save each detected face as an image
def detect_and_save_faces(input_image_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return

    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        # Draw face detections and save each face
        if results.detections:
            for idx, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape

                # Get the bounding box coordinates
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)

                # Extract the face from the image
                face_image = image[y_min:y_min + box_height, x_min:x_min + box_width]

                # Save the detected face
                face_output_path = os.path.join(output_folder, f"face_{idx+1}.jpg")
                cv2.imwrite(face_output_path, face_image)
                print(f"Saved face {idx+1} to {face_output_path}")

    # Display the original image with bounding boxes drawn on detected faces
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

    # Save the image with the detected faces
    output_image_path = os.path.join(output_folder, "detected_faces_image.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Saved image with detected faces to {output_image_path}")

# Paths
input_image_path = "WhatsApp Image 2024-10-08 at 10.09.07_c47dc282.jpg"  # Provide your input image path here
output_folder = "output"        # Folder to save detected faces

# Run the face detection and saving process
detect_and_save_faces(input_image_path, output_folder)
