import sys
import os
import cv2

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_video_frames.py <video_id_or_path>")
        print("Example: python view_video_frames.py 1118023")
        return

    arg = sys.argv[1]
    if os.path.exists(arg):
        vpath = arg
    else:
        vpath = f"../videos/{arg}.mp4"
        if not os.path.exists(vpath):
            vpath = f"../videos/{arg}"
    
    if not os.path.exists(vpath):
        print(f"Error: Video file '{vpath}' not found!")
        return

    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    current_frame = 0
    paused = True

    cv2.namedWindow("Video Frame Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Frame Viewer", 1280, 720)

    print("\nControls:")
    print("  [D] or [→] : Next frame (+1)")
    print("  [A] or [←] : Prev frame (-1)")
    print("  [W]        : Jump forward (+10)")
    print("  [S]        : Jump backward (-10)")
    print("  [Space]    : Play / Pause")
    print("  [Q] or [Esc]: Quit\n")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Draw frame number overlay
        overlay = frame.copy()
        text = f"Frame: {current_frame} / {total_frames - 1} | Time: {current_frame/fps:.2f}s ({fps:.1f} FPS)"
        cv2.rectangle(overlay, (10, 10), (550, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Video Frame Viewer", frame)

        wait_time = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(wait_time) & 0xFF

        if key in [ord('q'), 27]: # Esc or q
            break
        elif key == ord(' '): # Space
            paused = not paused
        elif key in [ord('d'), 83]: # d or right arrow
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key in [ord('a'), 81]: # a or left arrow
            current_frame = max(0, current_frame - 1)
        elif key == ord('w'):
            current_frame = min(total_frames - 1, current_frame + 10)
        elif key == ord('s'):
            current_frame = max(0, current_frame - 10)
        else:
            if not paused:
                current_frame = min(total_frames - 1, current_frame + 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
