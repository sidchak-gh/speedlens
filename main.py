# Import Libraries
import os
import cv2
import sys
import time
import numpy as np
from collections import defaultdict, deque

# Add ByteTrack to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ByteTrack'))

from utils import *
import speed_config
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from yolox.tracker.byte_tracker import BYTETracker, STrack


class vehicle_tracker_and_counter:

    def __init__(self,
                source_video_path="assets/vehicle-counting.mp4",
                target_video_path="assets/vehicle-counting-result.mp4",
                use_tensorrt=False):
        
        # YOLOv8 Object Detector
        self.model_name = "yolov8x.pt"
        self.yolo = YOLO(self.model_name)

        if use_tensorrt:
            try: 
                # Try to load model if it is already exported
                self.model = YOLO('yolov8x.engine')
            except:
                # Export model
                self.yolo.export(format='engine')  # creates 'yolov8x.engine'
                # Load the exported TensorRT model
                self.model = YOLO('yolov8x.engine')
        else:
            self.model = self.yolo
            self.model.fuse()

        self.CLASS_NAMES_DICT = self.yolo.model.names
        self.CLASS_ID = [2, 3, 5, 7]
  
        # Line for counter
        self.line_start = sv.Point(50, 1500)
        self.line_end = sv.Point(3840-50, 1500)

        # BYTETracke Object Tracker
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # Video input and output path
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        # Create VideoInfo instance
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        
        # Speed calculation setup
        self.view_transformer = ViewTransformer(speed_config.SOURCE, speed_config.TARGET)
        self.coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))
        # Create frame generator
        self.generator = sv.get_video_frames_generator(self.source_video_path)
        # Create LineCounter instance
        self.line_counter = sv.LineZone(start=self.line_start, end=self.line_end)
        # Create instance of BoxAnnotator and LineCounterAnnotator
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=self.video_info.resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=self.video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
        self.line_annotator = sv.LineZoneAnnotator(thickness=thickness)
            

    def run(self):
        # Open target video file
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            # loop over video frames
            for frame in tqdm(self.generator, total=self.video_info.total_frames):
                # model prediction on single frame and conversion to supervision Detections
                start_time = time.time()
                results = self.model(frame)
                end_time = time.time()
                fps = np.round(1/(end_time - start_time), 2)
                cv2.putText(frame, f'FPS: {fps}s', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)

                detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # filtering out detections with unwanted classes
                mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections = detections[mask]
                # tracking detections
                tracks = self.byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections = detections[mask]
                
                # Calculate speeds using perspective transformation
                points = []
                for detection in detections:
                    x1, y1, x2, y2 = detection[0]  # xyxy coordinates
                    bottom_center_x = (x1 + x2) / 2
                    bottom_center_y = y2
                    points.append([bottom_center_x, bottom_center_y])
                points = np.array(points)
                
                # Transform to real-world coordinates
                if len(points) > 0:
                    transformed_points = self.view_transformer.transform_points(points)
                else:
                    transformed_points = np.array([])
                
                # format custom labels with speed
                labels = []
                for idx in range(len(detections)):
                    confidence = detections.confidence[idx]
                    class_id = detections.class_id[idx]
                    tracker_id = detections.tracker_id[idx] if detections.tracker_id is not None else None
                    
                    if tracker_id is not None and idx < len(transformed_points):
                        _, y = transformed_points[idx]
                        self.coordinates[tracker_id].append(y)
                        
                        # Need at least 0.5 seconds of data
                        if len(self.coordinates[tracker_id]) < self.video_info.fps / 2:
                            speed_text = ""
                        else:
                            # Calculate speed from first to last position
                            coordinate_start = self.coordinates[tracker_id][-1]
                            coordinate_end = self.coordinates[tracker_id][0]
                            distance = abs(coordinate_end - coordinate_start)
                            time_elapsed = len(self.coordinates[tracker_id]) / self.video_info.fps
                            speed = distance / time_elapsed * 3.6  # Convert m/s to km/h
                            speed_text = f" {int(speed)} km/h"
                        
                        label = f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}{speed_text}"
                    else:
                        label = f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    
                    labels.append(label)
                # updating line counter
                self.line_counter.trigger(detections=detections)
                # annotate and display frame
                frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
                frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                frame = self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)
                sink.write_frame(frame)

if __name__ == '__main__':

    input_video="assets/vehicle-counting.mp4"
    output_video="assets/vehicle-counting-result.mp4"
    pipeline = vehicle_tracker_and_counter(source_video_path=input_video, target_video_path=output_video, use_tensorrt=False)
    pipeline.run()
