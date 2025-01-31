
from deepface import DeepFace
import cv2 as cv
import numpy as np
from src.logger.metadata_writer import MetaDataWriter
from src.logger.dataclass import FaceMetaData
import pandas as pd
import json
import pdb
class DeepFacePipe():
    def __init__(self,
                 write_directory:str = 'outputs',
                 detection_backbone:str = 'fastmtcnn',
                 recognition_backbone:str = 'VGG-Face') -> None:
        self.write_directory = write_directory
        self.detection_backbone = detection_backbone
        self.recognition_backbone = recognition_backbone
    
    def _search_frame(self, target_image: np.ndarray,
                  ref_path: str,
                  time_stamp: float,
                  filename: str):
        """
        Takes a target image (which can contain multiple faces) and does two things:
        1. Detects all faces.
        2. Finds the distance scores to all detected faces in the image.
        """
        dfs = DeepFace.find(
            img_path=target_image,
            db_path=ref_path,
            model_name=self.recognition_backbone,
            detector_backend=self.detection_backbone
        )

        # Get the minimum distance if faces are detected
        dfs = pd.concat(dfs,ignore_index=True)
        if not dfs.empty:
            min_row = dfs.loc[dfs['distance'].idxmin()]
            
            # Metadata dict
            metadata = FaceMetaData(
                timestamp=time_stamp,
                x=int(min_row['source_x']),
                y=int(min_row['source_y']),
                h=int(min_row['source_h']),
                w=int(min_row['source_w'])
            )
            return metadata, min_row['source_x'], min_row['source_y'], min_row['source_w'], min_row['source_h']
        else:
            return None, None, None, None, None


    def search_video(self,
                    vid_path: str,
                    ref_path: str,
                    frame_tol: int = 15,
                    max_frames: int = None):  # New parameter to limit the number of frames
        cap = cv.VideoCapture(vid_path)
        frame_count = 0
        frame_no_detection_count = 0  # Track consecutive frames without detection
        sequence_number = 0  # Track the sequence of video splits

        frames = []
        all_metadata = []
        video_writer = None  # We'll use this to save new video sequences

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Stop processing if max_frames is reached
            if max_frames and frame_count >= max_frames:
                break
            
            time_stamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
            try:
                metadata, x, y, w, h = self._search_frame(frame, ref_path, time_stamp, f'frame_{frame_count}.jpg')
            except Exception as e:
                print(e)
                metadata=None
            if metadata is None:
                frame_no_detection_count += 1
            else:
                frame_no_detection_count = 0  # Reset if detection is found
                # Draw a rectangle around the detected face
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, 2px thickness
            
            frames.append(frame)
            if metadata:
                all_metadata.append(metadata)
            
            # If no detections for `frame_tol` consecutive frames, split the video
            if frame_no_detection_count >= frame_tol:
                # Save the current frames and metadata to a new video sequence and JSON
                if frames:
                    output_video_path = f"{self.write_directory}/sequence_{sequence_number}.mp4"
                    output_json_path = f"{self.write_directory}/sequence_{sequence_number}.json"
                    
                    # Create new video writer if not already done
                    if video_writer is None:
                        video_writer = cv.VideoWriter(
                            output_video_path, 
                            cv.VideoWriter_fourcc(*'mp4v'), 
                            30,  # Assuming 30 FPS, adjust if necessary
                            (frames[0].shape[1], frames[0].shape[0])  # Frame size
                        )
                    
                    for frame in frames:
                        video_writer.write(frame)  # Write frames to the new video sequence
                    
                    # Save metadata to JSON
                    with open(output_json_path, 'w') as json_file:
                        json.dump([metadata.__dict__ for metadata in all_metadata], json_file, indent=4)
                
                # Reset for the next sequence
                frames = []
                all_metadata = []
                frame_no_detection_count = 0
                sequence_number += 1

            frame_count += 1
        
        # Final sequence if any frames are left
        if frames:
            output_video_path = f"{self.write_directory}/sequence_{sequence_number}.mp4"
            output_json_path = f"{self.write_directory}/sequence_{sequence_number}.json"
            
            if video_writer is None:
                video_writer = cv.VideoWriter(
                    output_video_path, 
                    cv.VideoWriter_fourcc(*'mp4v'), 
                    30,  # Assuming 30 FPS, adjust if necessary
                    (frames[0].shape[1], frames[0].shape[0])  # Frame size
                )
            
            for frame in frames:
                video_writer.write(frame)
            
            # Save metadata to JSON
            with open(output_json_path, 'w') as json_file:
                json.dump([metadata.__dict__ for metadata in all_metadata], json_file, indent=4)
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv.destroyAllWindows()
        