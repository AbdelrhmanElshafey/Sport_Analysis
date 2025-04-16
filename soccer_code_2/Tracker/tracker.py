from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
from scipy.spatial import ConvexHull, distance
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_convex_hull(self, frame, players, color):
        """
        Draws and fills convex hull for a team with high opacity.

        :param frame: The video frame
        :param players: List of player positions (x, y)
        :param color: Color for drawing the convex hull
        """
        if len(players) < 3:
            return frame  # Convex Hull needs at least 3 points

        points = np.array(players)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Create overlay for transparency
        overlay = frame.copy()
        polygon_pts = np.array([hull_points], dtype=np.int32)

        # Convert color to semi-transparent format
        fill_color = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))  # 50% opacity
        cv2.fillPoly(overlay, polygon_pts, fill_color)

        # Blend with frame
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Draw formation lines
        for i in range(len(hull_points)):
            pt1 = tuple(hull_points[i])
            pt2 = tuple(hull_points[(i + 1) % len(hull_points)])
            cv2.line(frame, pt1, pt2, color, thickness=3)

        return frame

    def draw_passing_lanes(self, frame, players, color):
        """
        Draws dashed lines between close players to visualize passing lanes.

        :param frame: The video frame
        :param players: List of player positions (x, y)
        :param color: Line color
        """
        if len(players) < 2:
            return frame

        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players):
                if i != j and distance.euclidean(p1, p2) < 150:
                    for k in range(0, 10, 2):
                        pt1 = (int(p1[0] + (p2[0] - p1[0]) * k / 10), int(p1[1] + (p2[1] - p1[1]) * k / 10))
                        pt2 = (int(p1[0] + (p2[0] - p1[0]) * (k + 1) / 10), int(p1[1] + (p2[1] - p1[1]) * (k + 1) / 10))
                        cv2.line(frame, pt1, pt2, color, thickness=1)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            team_positions = {}
            team_heatmaps = {}
            player_heatmaps = {}

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                color_tuple = tuple(color)

                pos = get_center_of_bbox(player["bbox"])

                if color_tuple not in team_positions:
                    team_positions[color_tuple] = []
                    team_heatmaps[color_tuple] = []

                team_positions[color_tuple].append(pos)
                team_heatmaps[color_tuple].append(pos)

                if track_id not in player_heatmaps:
                    player_heatmaps[track_id] = []
                player_heatmaps[track_id].append(pos)

                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            for color_tuple, positions in team_positions.items():
               frame = self.draw_convex_hull(frame, positions, color_tuple)
               frame = self.draw_passing_lanes(frame, positions, color_tuple)


            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames

    
    def generate_dynamic_team_heatmap(self, video_frames, tracks, output_path):
            """
            Generates a heatmap that updates dynamically in each frame to analyze team behavior.

            :param video_frames: List of video frames
            :param tracks: Player tracking data
            :param output_path: Path to save the output heatmap video
            """
            height, width, _ = video_frames[0].shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            heatmap_accumulator = np.zeros((height, width), dtype=np.float32)

            for frame_num, frame in enumerate(video_frames):
                frame_copy = frame.copy()
                player_dict = tracks["players"][frame_num]

                for _, player in player_dict.items():
                    pos = get_center_of_bbox(player["bbox"])
                    x, y = int(pos[0]), int(pos[1])
                    cv2.circle(heatmap_accumulator, (x, y), 40, 1, -1)  # Adding heat spots

                # Apply Gaussian Blur for smooth heatmap effect
                heatmap = cv2.GaussianBlur(heatmap_accumulator, (75, 75), 0)
                heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                heatmap = np.uint8(heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Overlay heatmap on frame
                blended = cv2.addWeighted(frame_copy, 0.6, heatmap, 0.4, 0)

                # Write to video
                out.write(blended)

            out.release()
            print(f"Dynamic team heatmap saved at {output_path}")