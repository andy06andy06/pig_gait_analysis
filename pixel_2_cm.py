import cv2
import os
import random
import math

def get_random_video_frame(video_dir):
    # Get list of all mp4 files
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not videos:
        print(f"No mp4 videos found in {video_dir}")
        return None, None
    
    # Select random video
    video_name = random.choice(videos)
    video_path = os.path.join(video_dir, video_name)
    print(f"Selected video: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None, None
        
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select random frame
    random_frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error reading frame")
        return None, None
        
    return frame, video_name

def get_coordinates(prompt):
    while True:
        try:
            coords_str = input(prompt)
            # Allow input as "x,y" or "x y"
            coords = coords_str.replace(',', ' ').split()
            if len(coords) != 2:
                print("Please enter exactly two numbers (x and y).")
                continue
            x = float(coords[0])
            y = float(coords[1])
            return (x, y)
        except ValueError:
            print("Invalid input. Please enter numbers.")

def main():
    video_dir = "../videos"  # Assuming script is in 'code' folder and videos in 'videos' folder
    
    # Adjust path if running from a different directory
    if not os.path.exists(video_dir):
        # Try absolute path or relative from workspace root
        video_dir = "pig_gait_v1-Andy-2025-02-26/videos"
        if not os.path.exists(video_dir):
             print(f"Could not find video directory. Current dir: {os.getcwd()}")
             return

    frame, video_name = get_random_video_frame(video_dir)
    if frame is None:
        return

    # Save the frame to a file
    output_filename = "frame_to_measure.jpg"
    cv2.imwrite(output_filename, frame)
    print(f"\nSaved random frame to '{output_filename}'.")
    print("Please open this image in VSCode or an image viewer to find the coordinates of your two points.")
    
    # Get coordinates from user
    print("\n--- Point Selection ---")
    p1 = get_coordinates("Enter coordinates for Point 1 (x y): ")
    p2 = get_coordinates("Enter coordinates for Point 2 (x y): ")
    
    # Calculate pixel distance
    pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    print(f"Pixel Distance: {pixel_dist:.2f} pixels")
    
    # Get real world distance
    while True:
        try:
            real_dist_str = input("Enter real world distance between the two points (cm): ")
            real_dist = float(real_dist_str)
            if real_dist <= 0:
                print("Distance must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # Calculate ratio
    ratio = real_dist / pixel_dist
    
    print(f"\n--- Results for {video_name} ---")
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")
    print(f"Pixel Distance: {pixel_dist:.2f} pixels")
    print(f"Real Distance: {real_dist:.2f} cm")
    print(f"Conversion Factor: {ratio:.6f} cm/pixel")
    print(f"Inverse Factor: {1/ratio:.6f} pixels/cm")
    print("------------------------------\n")

    # Clean up (optional - keep the file for reference)
    # os.remove(output_filename)

if __name__ == "__main__":
    main()
