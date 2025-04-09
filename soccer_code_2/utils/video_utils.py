import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

def save_video_chunks(video_path, output_dir, chunks):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exist
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of original video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)


    chunk_count = 0
    for chunk in chunks:
        player_id = chunk['player_id']
        start_frame = chunk['start_frame']
        end_frame = chunk['end_frame']
        chunk_count += 1

        # Set video capture to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Define output video writer
        output_path = os.path.join(output_dir, f'chunk_{chunk_count}_player_{player_id}_frames_{start_frame}_{end_frame}.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

        print(f"Saving chunk {chunk_count}: Player {player_id} | Frames {start_frame}-{end_frame}")

        # Read and write frames
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_num}")
                break
            out.write(frame)

        out.release()  # Save the current video clip

    cap.release()  # Release video file
    print("All possession chunks have been saved.")