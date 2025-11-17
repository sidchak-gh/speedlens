import os
import sys
import numpy as np
from typing import List

# Add ByteTrack to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# ViewTransformer class for speed calculation
import cv2

class ViewTransformer:
    """
    Transforms image coordinates to real-world coordinates using perspective transformation.
    Used for calculating vehicle speeds in real-world units (meters, km/h).
    """
    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Initialize the perspective transformer.
        
        Args:
            source: 4 points in image coordinates forming a quadrilateral
            target: 4 points in real-world coordinates (meters)
        """
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from image coordinates to real-world coordinates.
        
        Args:
            points: Array of shape (N, 2) with [x, y] coordinates
            
        Returns:
            Transformed points in real-world coordinates
        """
        if len(points) == 0:
            return np.array([])
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)