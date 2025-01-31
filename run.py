import argparse
from src.pipeline.deepface_pipe import DeepFacePipe
from src.pipeline.mtcnn_pipe import MtcnnPipe

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run face detection and recognition on a video.")
    parser.add_argument('--vid_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--ref_path', type=str, required=True, help="Path to the reference image database.")
    parser.add_argument('--frame_tol', type=int, default=15, help="Number of consecutive frames without detection before splitting.")
    parser.add_argument('--max_frames', type=int, default=None, help="Maximum number of frames to process (optional).")

    args = parser.parse_args()

    # Initialize DeepFacePipe and run face detection on video
    # deepface_pipe = DeepFacePipe()
    # deepface_pipe.search_video(
    #     vid_path=args.vid_path,
    #     ref_path=args.ref_path,
    #     frame_tol=args.frame_tol,
    #     max_frames=args.max_frames
    # )
    pipe = MtcnnPipe()
    pipe.search_video(
        vid_path=args.vid_path,
        ref_path=args.ref_path,
        frame_tol=args.frame_tol,
        max_frames=args.max_frames
    )


    print(f"Processing completed for video: {args.vid_path}")