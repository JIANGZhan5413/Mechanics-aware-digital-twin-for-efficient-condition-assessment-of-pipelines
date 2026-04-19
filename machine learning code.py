import cv2
from ultralytics import YOLO
import time
import os

def process_video(model_path, video_path, output_path=None, conf_threshold=0.25):
    """
    Process video with YOLOv8 model and save the output
    """
    try:
        # Load the YOLOv8 model
        print("Loading model...")
        model = YOLO(model_path)
        
        # Load the video
        print("Opening video file...")
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise Exception(f"Error: Could not open video {video_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Generate output path if not provided
        if output_path is None:
            base_path = os.path.splitext(video_path)[0]
            output_path = f"{base_path}_processed.mp4"
            
        # Initialize video writer
        print("Initializing video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Error: Could not create output video file")
            
        # Initialize progress tracking
        start_time = time.time()
        frame_count = 0
        
        print("Starting video processing...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform detection
            results = model(frame, conf=conf_threshold)[0]
            
            # Draw the results on the frame
            annotated_frame = results.plot()
            
            # Write the frame
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 30 == 0:  # Update progress every 30 frames
                elapsed_time = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_processing = frame_count / elapsed_time
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) "
                      f"Processing speed: {fps_processing:.1f} FPS")
        
        # Clean up
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        average_fps = frame_count / total_time
        print("\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average processing speed: {average_fps:.1f} FPS")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up in case of error
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        raise

if __name__ == "__main__":
    # Configuration
    model_path = r"C:\Users\smart\Desktop\ensemble model\100% of data\Cracks single\runs\detect\train\weights\best.pt"
    video_path = r"C:\Users\smart\Desktop\123.webm"
    
    # Process the video
    try:
        process_video(
            model_path=model_path,
            video_path=video_path,
            conf_threshold=0.25  # Adjust confidence threshold as needed
        )
    except Exception as e:
        print(f"Failed to process video: {str(e)}")