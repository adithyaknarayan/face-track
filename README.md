# **Face-Track**
A simple face tracker using **MTCNN** with trajectory smoothing.

## **Setup**
Getting started with this repository is straightforward.

### **Running on a Custom Video**
To run the face tracking pipeline, follow these steps:

1. **Prepare the Data:**  
   - Download any video and place it in the `data/` folder.  
   - Add the reference face image to `data/reference_face/`.

2. **Run the Inference Pipeline:**  
   Execute the following command:  
   ```bash
   python run.py --vid_path /home/adithya/HSL/test/face_detect/data/vid.mp4 \
                 --ref_path /home/adithya/HSL/test/face_detect/data/reference_face/ref2.png \
                 --max_frames 50 \
                 --frame_tol 5

Here's a list of CLI arguments `run.py` can accept.
## **Command-Line Arguments**

| Argument        | Type    | Default              | Description                                                                |
|-----------------|---------|----------------------|----------------------------------------------------------------------------|
| `--vid_path`    | `str`   | **Required**         | Path to the input video file.                                              |
| `--ref_path`    | `str`   | **Required**         | Path to the reference image database.                                      |
| `--frame_tol`   | `int`   | `15`                 | Number of consecutive frames without detection before splitting.           |
| `--max_frames`  | `int`   | `None` (process all) | Maximum number of frames to process.                                       |


# **Results**


# **Adding Smoothing**
I also added some very basic evaluations on the effect of smoothing on the capture of the face. It stabilizes it so some extent, however, more stabilization through some keypoint based approach might have provided better results.
## Full Segment Comparison  
### Smoothed vs. Unsmoothed  
**Smoothed:**  
![<video controls src="demo_src/full_segment_smoothed.mp4" title="Full Segment Smoothed"></video>  ](demo_src/full_smoothed.gif)

**Unsmoothed:**  
![<video controls src="demo_src/full_segment_unsmoothed.mp4" title="Full Segment Unsmoothed"></video>  ](demo_src/full_unsmoothed.gif)

---

## Face Segment Comparison  
### Smoothed vs. Unsmoothed  

**Smoothed:**  
<img src="demo_src/face_smoothed.gif" title="Face Segment Smoothed" width="200" />

**Unsmoothed:**  
<img src="demo_src/face_unsmoothed.gif" title="Face Segment Unsmoothed" width="200" />

# Profiling
To do quick profiling of the code on your system, run the following command.
```
python -m cProfile -o profile.prof run.py --vid_path /home/adithya/HSL/test/face_detect/data/vid.mp4 --ref_path=/home/adithya/HSL/test/face_detect/data/reference_face/ref2.png --max_frames=50 --frame_tol=5
```
Then to visualize, run,

```snakeviz profile.prof ```

