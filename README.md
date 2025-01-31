# face-track
A simple face tracker with MTCNN and trajectory smoothing


# Setup

# Components

# Adding Smoothing
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
![<video controls src="demo_src/face_segment_unsmoothed.mp4" title="Face Segment Unsmoothed"></video>  ](demo_src/face_smoothed.gif)

**Unsmoothed:**  
![<video controls src="demo_src/face_segment_smoothed.mp4" title="Face Segment Smoothed"></video> ](demo_src/face_unsmoothed.gif)

# Profiling
To do quick profiling of the code on your system, run the following command.
```
python -m cProfile -o profile.prof run.py --vid_path /home/adithya/HSL/test/face_detect/data/vid.mp4 --ref_path=/home/adithya/HSL/test/face_detect/data/reference_face/ref2.png --max_frames=50 --frame_tol=5
```
Then to visualize, run,

```snakeviz profile.prof ```

