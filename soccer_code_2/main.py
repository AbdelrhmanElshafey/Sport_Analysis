from utils import read_video, save_video, save_video_chunks
from Tracker import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Read Video
    video_frames = read_video('input_videos/match_video.mp4')

    # Intialize Tracker
    tracker = Tracker('Fine-Tuned Models/Yolov5/fine_tuned_yolov8.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs_1.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub_1.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    player_with_ball_sequence = []

    # Step 1: Assign ball possession frame by frame and track player IDs
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            player_with_ball_sequence.append(assigned_player)
        else:
            # If no player has ball, repeat last assigned player (or -1 if first frame)
            last_player = player_with_ball_sequence[-1] if player_with_ball_sequence else -1
            player_with_ball_sequence.append(last_player)
            last_team = team_ball_control[-1] if team_ball_control else -1
            team_ball_control.append(last_team)


    team_ball_control= np.array(team_ball_control)
    # Convert to numpy array for convenience
    player_with_ball_sequence = np.array(player_with_ball_sequence)

    # Step 2: Detect change points where the assigned player changes
    change_points = [0]  # First frame is start of first chunk
    for i in range(1, len(player_with_ball_sequence)):
        if player_with_ball_sequence[i] != player_with_ball_sequence[i - 1]:
            change_points.append(i)

    # Add last frame as end of final chunk
    change_points.append(len(player_with_ball_sequence))

    # Step 3: Divide video into chunks based on change points and make sure each chunk is at least 50 frames
    chunks = []
    min_frames = 50  # Minimum frames per chunk

    i = 0  # Index to iterate through change points

    while i < len(change_points) - 1:
        start_frame = change_points[i]
        end_frame = change_points[i + 1] - 1  # Tentative end frame
        player_id = player_with_ball_sequence[start_frame]

        # Check if current chunk is less than min_frames
        num_frames = end_frame - start_frame + 1

        # If chunk is too short, try to merge with next chunks until it reaches at least 50 frames
        while num_frames < min_frames and (i + 1) < len(change_points) - 1:
            # Move to next change point to extend the chunk
            i += 1
            end_frame = change_points[i + 1] - 1  # Extend end frame
            num_frames = end_frame - start_frame + 1  # Recalculate number of frames

        # Append valid (or merged) chunk
        chunks.append({
            'player_id': player_id,
            'start_frame': start_frame,
            'end_frame': end_frame
        })

        # Move to next chunk starting point
        i += 1

    # Generate Convex Hull Video
    convex_hull_video_frames = tracker.draw_convex_hull(video_frames, tracks)

    # Save video chunks
    save_video_chunks('input_videos/match_video.mp4', 'video_chunks', chunks)

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    save_video(output_video_frames, 'output_videos/test_fine_tuned_yolov8.avi')
    # Generate heatmaps for each team
    
    tracker.generate_dynamic_team_heatmap(video_frames, tracks, output_path= 'output_videos/team_heatmaps.avi')

if __name__ == '__main__':
    main()
