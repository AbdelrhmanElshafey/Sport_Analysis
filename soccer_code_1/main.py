import argparse
from enum import Enum
from typing import Iterator, List, Tuple

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_percentage_based_heatmap
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    VORONOI = 'VORONOI'
    HEATMAP_TEAM0 = 'HEATMAP_TEAM0'
    HEATMAP_TEAM1 = 'HEATMAP_TEAM1'
    ALL_DETECTORS = "ALL_DETECTORS"

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    ball_detections: sv.Detections
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    
    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,edge_color=sv.Color.BLACK, radius=15, pitch=radar)
    return radar

def render_voronoi_diagram(
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
        color_lookup: np.ndarray
) -> np.ndarray:
    # Ensure keypoints are valid and within a reasonable boundary
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # Prepare the radar visualization by drawing the pitch
    voronoi = draw_pitch(config=CONFIG)

    # Separate the players by team using color_lookup and draw them
    for team_id in np.unique(color_lookup):
        team_xy = transformed_xy[color_lookup == team_id]
        if team_xy.size > 0:
            voronoi = draw_points_on_pitch(
                config=CONFIG,
                xy=team_xy,
                face_color=sv.Color.from_hex(COLORS[team_id]),
                radius=20,
                pitch=voronoi)

    # Draw Voronoi diagram if there are enough players in at least two teams
    team_indices = [color_lookup == i for i in range(len(COLORS))]
    if any(team_xy.size > 0 for team_xy in team_indices):
        team_1_xy = transformed_xy[team_indices[0]]
        team_2_xy = transformed_xy[team_indices[1]]
        voronoi = draw_pitch_voronoi_diagram(
            config=CONFIG,
            team_1_xy=team_1_xy,
            team_2_xy=team_2_xy,
            team_1_color=sv.Color.from_hex(COLORS[0]),
            team_2_color=sv.Color.from_hex(COLORS[1]),
            pitch=voronoi)

    return voronoi

def render_heatmap(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    team_heatmap_buffer: List[np.ndarray],
    team: int
) -> np.ndarray:
    # Transform detections to pitch space
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )

    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    # Prepare the radar visualization by drawing the pitch
    heatmap = draw_pitch(config=CONFIG)

    
    # Split players by team
    if team == 0:
        team_0_xy = transformed_xy[color_lookup == 0]
        # Accumulate positions into global buffer
        team_heatmap_buffer.append(np.array(team_0_xy))
        # Draw player positions
        heatmap = draw_points_on_pitch(
            config=CONFIG,
            xy=team_0_xy,
            face_color=sv.Color.from_hex(COLORS[0]),
            edge_color=sv.Color.WHITE,
            radius=16,
            pitch=heatmap)
        team_color = sv.Color.from_hex(COLORS[0])
        # Draw radar with cumulative heatmap
        heatmap = draw_percentage_based_heatmap(
            config=CONFIG,
            team_frames_xy=np.array(team_heatmap_buffer),
            #team_color=team_color,
            opacity=0.6,
            pitch=heatmap
        )
    else:
        team_1_xy = transformed_xy[color_lookup == 1]
        # Accumulate positions into global buffer
        team_heatmap_buffer.append(np.array(team_1_xy))
        # Draw player positions
        heatmap = draw_points_on_pitch(
            config=CONFIG,
            xy=team_1_xy,
            face_color=sv.Color.from_hex(COLORS[1]),
            edge_color=sv.Color.WHITE,
            radius=16,
            pitch=heatmap)
        team_color = sv.Color.from_hex(COLORS[1])
        # Draw radar with cumulative heatmap
        heatmap = draw_percentage_based_heatmap(
            config=CONFIG,
            team_frames_xy=np.array(team_heatmap_buffer),
            #team_color=team_color,
            opacity=0.6,
            pitch=heatmap
        )

    return heatmap

def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame

def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame

def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback,overlap_filter=sv.OverlapFilter.NONE,slice_wh=(640, 640),)

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        # Ball detection
        detections_ball = slicer(frame).with_nms(threshold=0.1)
        detections_ball = ball_tracker.update(detections_ball)
        # Get all ball detections
        ball_detections = detections_ball[detections_ball.class_id == BALL_CLASS_ID]
  
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
        yield annotated_frame

def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    ball_tracker = BallTracker(buffer_size=20)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback,overlap_filter=sv.OverlapFilter.NONE,slice_wh=(640, 640),)
    
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in frame_generator:
        # Ball detection
        detections_ball = slicer(frame).with_nms(threshold=0.1)
        detections_ball = ball_tracker.update(detections_ball)
        # Get all ball detections
        ball_detections = detections_ball[detections_ball.class_id == BALL_CLASS_ID]
        
        # Keep only the ball with the highest confidence (if any)
        if len(ball_detections) > 0:
            top_conf_index = ball_detections.confidence.argmax()
            ball_detections = ball_detections[top_conf_index : top_conf_index + 1]  # Keep it as a Detections object
        
            # Pad the selected detection
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        else:
            ball_detections = sv.Detections.empty()
        
        # Pitch Detection
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Player Detection
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees) 
        )

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup, ball_detections)
        
        # Check the radar type and shape
        radar = sv.resize_image(radar, (w, h))
        
        yield radar

def run_voronoi(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    
    for frame in frame_generator:
        
        # Pitch Detection
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Player Detection
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)


        detections = sv.Detections.merge([players, goalkeepers])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist()
        )

        h, w, _ = frame.shape
        voronoi = render_voronoi_diagram(detections, keypoints, color_lookup)
        
        voronoi = sv.resize_image(voronoi, (w, h))
        
        yield voronoi

def run_heatmap_team0(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    team_0_heatmap_buffer = []

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        detections = sv.Detections.merge([players, goalkeepers])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist()
        )

        h, w, _ = frame.shape
        heatmap = render_heatmap(detections, keypoints, color_lookup, team_0_heatmap_buffer, team = 0)
        heatmap = sv.resize_image(heatmap, (w, h))
        
        yield heatmap

def run_heatmap_team1(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    team_1_heatmap_buffer = []

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        detections = sv.Detections.merge([players, goalkeepers])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist()
        )

        h, w, _ = frame.shape
        heatmap = render_heatmap(detections, keypoints, color_lookup, team_1_heatmap_buffer, team = 1)
        heatmap = sv.resize_image(heatmap, (w, h))
        
        yield heatmap

def run_all_detectors(source_video_path: str, device: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(callback=callback,overlap_filter=sv.OverlapFilter.NONE,slice_wh=(640, 640),)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    team_0_heatmap_buffer = []
    team_1_heatmap_buffer = []

    for frame in frame_generator:

        h, w, _ = frame.shape

        # Ball detection
        ball_detections = slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)
        
        # Pitch Detection
        keypoints_detections = pitch_detection_model(frame, verbose=False)[0]
        keypoints_detections = sv.KeyPoints.from_ultralytics(keypoints_detections)
        
        # Players detections
        players_detections = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        players_detections = sv.Detections.from_ultralytics(players_detections)
        players_detections = tracker.update_with_detections(players_detections)

        players = players_detections[players_detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = players_detections[players_detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        players_detections = sv.Detections.merge([players, goalkeepers])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist()
        )
        labels = [str(tracker_id) for tracker_id in players_detections.tracker_id]

        
        # --- players & ball detections & tracking ---
        players_ball_annotated_frame = frame.copy()
        players_ball_annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            players_ball_annotated_frame, players_detections, custom_color_lookup=color_lookup)
        players_ball_annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            players_ball_annotated_frame, players_detections, labels, custom_color_lookup=color_lookup)
        players_ball_annotated_frame = ball_annotator.annotate(players_ball_annotated_frame, ball_detections)



        # --- 2D Radar representation of players & ball & pitch ---
        radar = render_radar(players_detections, keypoints_detections, color_lookup, ball_detections)
        
        radar = sv.resize_image(radar, (w, h))

        # --- Space controlled by each team ---
        voronoi = render_voronoi_diagram(players_detections, keypoints_detections, color_lookup)
        
        voronoi = sv.resize_image(voronoi, (w, h))

        # --- Heatmap of first team ---
        heatmap_1 = render_heatmap(players_detections, keypoints_detections, color_lookup, team_0_heatmap_buffer, team = 0)
        heatmap_1 = sv.resize_image(heatmap_1, (w, h))
        
        # --- Heatmap of second team ---
        heatmap_2 = render_heatmap(players_detections, keypoints_detections, color_lookup, team_1_heatmap_buffer, team = 1)
        heatmap_2 = sv.resize_image(heatmap_2, (w, h))
    
        yield players_ball_annotated_frame, radar, voronoi, heatmap_1, heatmap_2

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.ALL_DETECTORS:
        video_info = sv.VideoInfo.from_video_path(source_video_path)

        players_ball_tracking_sink = sv.VideoSink(target_video_path.replace('.mp4', '_players_ball_tracking.mp4'), video_info)
        radar_sink = sv.VideoSink(target_video_path.replace('.mp4', '_radar.mp4'), video_info)
        voronoi_sink = sv.VideoSink(target_video_path.replace('.mp4', '_voronoi.mp4'), video_info)
        heatmap_1_sink = sv.VideoSink(target_video_path.replace('.mp4', '_heatmap_team0.mp4'), video_info)
        heatmap_2_sink = sv.VideoSink(target_video_path.replace('.mp4', '_heatmap_team1.mp4'), video_info)

        with players_ball_tracking_sink, radar_sink, voronoi_sink, heatmap_1_sink, heatmap_2_sink:
            
            for players_ball_frame, radar_frame, voronoi_frame, heatmap_1_frame, heatmap_2_frame in run_all_detectors(source_video_path, device=device):
                players_ball_tracking_sink.write_frame(players_ball_frame)
                radar_sink.write_frame(radar_frame)
                voronoi_sink.write_frame(voronoi_frame)
                heatmap_1_sink.write_frame(heatmap_1_frame)
                heatmap_2_sink.write_frame(heatmap_2_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    else:
        if mode == Mode.PITCH_DETECTION:
            frame_generator = run_pitch_detection(source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_DETECTION:
            frame_generator = run_player_detection(source_video_path=source_video_path, device=device)
        elif mode == Mode.BALL_DETECTION:
            frame_generator = run_ball_detection(source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_TRACKING:
            frame_generator = run_player_tracking(source_video_path=source_video_path, device=device)
        elif mode == Mode.TEAM_CLASSIFICATION:
            frame_generator = run_team_classification(source_video_path=source_video_path, device=device)
        elif mode == Mode.RADAR:
            frame_generator = run_radar(source_video_path=source_video_path, device=device)
        elif mode == Mode.VORONOI:
            frame_generator = run_voronoi(source_video_path=source_video_path, device=device)
        elif mode == Mode.HEATMAP_TEAM0:
            frame_generator = run_heatmap_team0(source_video_path=source_video_path, device=device)
        elif mode == Mode.HEATMAP_TEAM1:
            frame_generator = run_heatmap_team1(source_video_path=source_video_path, device=device)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")

        video_info = sv.VideoInfo.from_video_path(source_video_path)
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in frame_generator:
                sink.write_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
