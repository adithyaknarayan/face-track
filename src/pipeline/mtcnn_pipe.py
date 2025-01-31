import torch
from torch.nn import CosineSimilarity
import cv2 as cv
import os
import json
from tqdm import tqdm
from src.extract.face_extractor import FaceExtractor
from facenet_pytorch import InceptionResnetV1
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.logger.dataclass import FaceMetaData
import numpy as np

class MtcnnPipe():
    def __init__(self, write_directory: str = 'outputs', device: str = 'cuda:0', score_thresh: float = 0.5):
        self.write_directory = write_directory
        os.makedirs(write_directory, exist_ok=True)  # Ensure output directory exists
        self.extractor = FaceExtractor(device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        self.score_fn = CosineSimilarity()
        self.score_thresh = score_thresh
        self.device = device
        self.last_bbox = None
        self.tracker = DeepSort(max_age=30, n_init=2, nn_budget=None)
        self.last_bbox_used_frames = 0  # Track how many frames the last bbox is used

    def _get_max_score_idx(self, aligned_faces, ref_img):
        """Finds the most similar face in the frame to the reference image."""
        if aligned_faces is None or ref_img is None:
            return None
        embed_a = self.resnet(aligned_faces)
        embed_ref = self.resnet(ref_img)
        scores = self.score_fn(embed_a, embed_ref)  # Similarity scores
        max_idx = scores.argmax()
        return max_idx if scores[max_idx] > self.score_thresh else None
    
    def _smoother(self, bboxes, probs, frame, max_idx):
        detections = []
        for i in range(bboxes.shape[0]):
            x1,y1,x2,y2 = bboxes[i]
            prob = probs[i]
            detections.append(([x1, y1, x2 - x1, y2 - y1], prob, None))
        tracks = self.tracker.update_tracks(detections, frame=frame.cpu().numpy())

        corrected = []
        for i,track in enumerate(tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, w, h = track.to_tlwh()
            x2, y2 = int(x1 + w), int(y1 + h)
            corrected.append([x1,y1,x2,y2])
        if corrected != []:
            corrected = corrected
        return corrected

    def _search_frame(self, frame: torch.tensor, ref_img: torch.tensor, timestamp: str):
        """Detects and aligns the most relevant face in the frame."""
        aligned_faces, bboxes, ref_img, probs = self.extractor.extract_and_align(frame, ref_img)
        max_idx = self._get_max_score_idx(aligned_faces, ref_img)
        if max_idx is not None:
            self.last_bbox = bboxes[max_idx]  # Update last detected bbox
            self.last_bbox_used_frames = 0  # Reset counter for the last bbox
            bbox = bboxes[max_idx]
            return FaceMetaData(timestamp=timestamp, x=int(bbox[0]), y=int(bbox[1]), 
                                w=int(bbox[2] - bbox[0]), h=int(bbox[3] - bbox[1]))
        elif self.last_bbox is not None:
            # If face disappears, use the last known bbox and increment the used frames counter
            self.last_bbox_used_frames += 1
            return FaceMetaData(timestamp=timestamp, x=int(self.last_bbox[0]), y=int(self.last_bbox[1]), 
                                w=int(self.last_bbox[2] - self.last_bbox[0]), h=int(self.last_bbox[3] - self.last_bbox[1]))
        return None

    def search_video(self, vid_path: str, ref_path: str, frame_tol: int = 15, max_frames: int = None):
        """Processes the video to detect faces, save cropped faces, and create videos."""
        cap = cv.VideoCapture(vid_path)
        ref_img = torch.tensor(cv.imread(ref_path)).cuda()
        
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get total frame count
        frame_count = 0
        missing_face_count = 0
        segment_index = 0
        recording = True

        frames, metadata, face_clips = [], [], []
        first_face_size = None  # Store the first face dimensions
        video_writer = None
        face_writer = None

        # Add progress bar
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                pbar.update(1)

                timestamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0

                if max_frames and frame_count >= max_frames:
                    break

                detection = self._search_frame(torch.tensor(frame).cuda(), ref_img, timestamp)

                if detection and self.last_bbox_used_frames<frame_tol:
                    missing_face_count = 0
                    x, y, w, h = detection.x, detection.y, detection.w, detection.h
                    metadata.append(detection)

                    #get face
                    face_crop = frame[y:y+h, x:x+w].copy()

                    # Draw bbox on full video
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Store the first detected face size in this segment
                    if first_face_size is None:
                        first_face_size = (face_crop.shape[1], face_crop.shape[0])  # (width, height)

                    # Resize subsequent faces to match the first detected face size
                    if first_face_size:
                        face_crop = cv.resize(face_crop, first_face_size)

                    face_clips.append(face_crop)

                    if not recording:
                        segment_index += 1
                        recording = True
                else:
                    missing_face_count += 1

                    # If no face is detected for more than frame_tol frames OR last bbox used for more than frame_tol frames, save segment
                    if missing_face_count > frame_tol or self.last_bbox_used_frames > frame_tol:
                        # this specifically modifies it so that it cleans out the last few frame of no detection
                        self._save_segment(frames[:-frame_tol], face_clips[:-frame_tol], metadata[:-frame_tol], segment_index, first_face_size, vid_path, ref_path)
                        frames, metadata, face_clips = [], [], []
                        first_face_size = None  # Reset for next segment
                        recording = False

                if recording:
                    frames.append(frame)

            # Save any remaining frames
            if frames:
                self._save_segment(frames, face_clips, metadata, segment_index, first_face_size, vid_path, ref_path)

        cap.release()
        if video_writer:
            video_writer.release()
        if face_writer:
            face_writer.release()
        cv.destroyAllWindows()

    def _save_segment(self, frames, face_clips, metadata, segment_index, first_face_size, vid_path, ref_path):
        """Saves the current video segment with full frame and extracted face."""
        if not frames:
            return
        
        # Save full video with bounding box
        vid_name = vid_path.split('/')[-1].split('.')[0]
        ref_face_name = ref_path.split('/')[-1].split('.')[0]

        # create folder if not found
        base_folder_path = os.path.join(self.write_directory, vid_name)
        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)

        full_video_path = os.path.join(self.write_directory, f"{vid_name}/full_segment_{ref_face_name}_{segment_index}.mp4")
        json_path = os.path.join(self.write_directory, f"{vid_name}/segment_vid_{ref_face_name}_{segment_index}.json")

        height, width, _ = frames[0].shape
        full_video_writer = cv.VideoWriter(full_video_path, cv.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        for frame in frames:
            full_video_writer.write(frame)
        full_video_writer.release()

        # Save extracted face video
        if face_clips and first_face_size:
            face_video_path = os.path.join(self.write_directory, f"{vid_name}/face_segment_{ref_face_name}_{segment_index}.mp4")
            face_video_writer = cv.VideoWriter(face_video_path, cv.VideoWriter_fourcc(*'mp4v'), 30, first_face_size)
            
            for face in face_clips:
                face_video_writer.write(face)
            face_video_writer.release()

        # Save metadata
        with open(json_path, 'w') as json_file:
            metadata_json = {
            "file_name": face_video_path if face_clips else full_video_path,
            "start_timestamp": metadata[0].timestamp if metadata else 0,
            "end_timestamp": metadata[-1].timestamp if metadata else 0,
            "face_coordinates": [md.__dict__ for md in metadata]
            }
            json.dump(metadata_json, json_file, indent=4)

        print(f"Saved {len(frames)} frames to {full_video_path}")