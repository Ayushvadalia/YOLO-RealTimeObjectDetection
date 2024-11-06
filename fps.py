from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8x.pt")

# Path to the input video
input_video_path = "C:/Users/Ayush Tushar Vadalia/Documents/YOLO IPCVproject/2932301-uhd_4096_2160_24fps.mp4"

# Set up the VideoCapture and VideoWriter
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second for video (might not reflect actual processing speed)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output
output_video_path = "C:/Users/Ayush Tushar Vadalia/Documents/YOLO IPCVproject/tracked_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Variables for FPS calculation
start_time = None
processed_frames = 0

# Process each frame and save it with detections
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    # Start time for FPS calculation (before prediction)
    if start_time is None:
        start_time = cv2.getTickCount()

    # Run predictions on the frame
    results = model.predict(frame, stream=True)

    # Overlay results on the frame
    for result in results:
        frame_with_detections = result.plot()  # Draw detections on the frame

    # End time for FPS calculation (after processing)
    end_time = cv2.getTickCount()
    processed_frames += 1

    # Calculate and print FPS every few frames (adjust interval as needed)
    if processed_frames % 10 == 0:
        process_time = (end_time - start_time) / cv2.getTickFrequency()
        fps_processing = 1 / process_time
        print(f"Processed {processed_frames} frames. FPS (Processing): {fps_processing:.2f}")
        start_time = None  # Reset for next FPS calculation

    # Write the frame with detections to the output video
    out.write(frame_with_detections)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as: {output_video_path}")