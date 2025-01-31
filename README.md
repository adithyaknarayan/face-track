# **Face-Track**
A simple face tracker using **MTCNN** with trajectory smoothing.

## **Setup**
Getting started with this repository is straightforward.

### **1. Create a Conda Environment**

First, create a new conda environment called **face** by running the following command:

```bash
conda create -n face python=3.10.16
conda activate face
```
### **2. Install Dependencies**
```
pip install -r requirements.txt
```

### **3. Verify GPU**
Note that most evaluations were done with a GPU. To ensure that the pipeline works as intended, please ensure that cuda is installed.

```
nvidia-smi
```

## **Running on a Custom Video**
To run the face tracking pipeline, follow these steps:

1. **Prepare the Data:**  
   - Download any video and place it in the `data/` folder.  
   - Add the reference face image to `data/reference_face/`.

2. **Run the Inference Pipeline:**  
   Execute the following command:  
   ```bash
   python run.py --vid_path data/vid.mp4 \
                 --ref_path data/reference_face/ref2.png \
                 --max_frames 50 \
                 --frame_tol 5
   ```
3. **(Optional) Run the DeepFace Pipeline:**  
   Execute the following command:  
   ```bash
   python run.py --vid_path data/vid.mp4 \
                 --ref_path data/reference_face/\
                 --max_frames 50 \
                 --frame_tol 5
   ```
   Noe that for this to work, you need to go to `run.py` and uncomment the relevant chunk of code. It is also relatively untested.

Here's a list of CLI arguments `run.py` can accept.
## **Command-Line Arguments**

| Argument        | Type    | Default              | Description                                                                |
|-----------------|---------|----------------------|----------------------------------------------------------------------------|
| `--vid_path`    | `str`   | **Required**         | Path to the input video file.                                              |
| `--ref_path`    | `str`   | **Required**         | Path to the reference image database.                                      |
| `--frame_tol`   | `int`   | `15`                 | Number of consecutive frames without detection before splitting.           |
| `--max_frames`  | `int`   | `None` (process all) | Maximum number of frames to process.                                       |
## **Output Formats**
All outputs should be found in the `outputs` folder. The filename 
## **Face Bounding Box Dataclass Format**
I also use a dataclass to structure the output format for the metadata. This is not necessarily REQUIRED, but for potential future API support, dataclasses make downstream development more consistent.

```python
from dataclasses import dataclass

@dataclass
class FaceMetaData:
    timestamp:float = None
    x:int = None
    y:int = None
    h:int = None
    w:int = None
```


You can see a sample output json for this dataclass below as well.
```json
{
    "file_name": "outputs/ronaldo/face_segment_ronaldo_0.mp4",
    "start_timestamp": 0.0,
    "end_timestamp": 39.92,
    "face_coordinates": [
        {
            "timestamp": 0.0,
            "x": 517,
            "y": 83,
            "h": 69,
            "w": 56
        },
        {
            "timestamp": 0.04,
            "x": 518,
            "y": 80,
            "h": 73,
            "w": 59
        },
        {
            "timestamp": 0.08,
            "x": 516,
            "y": 81,
            "h": 72,
            "w": 59
        }
  ]
}

```


# **Results**
The results shown below include smoothing. For some extra comparisons you can look at the `Adding Smoothing` section below.

For a more concrete look at the results shown below, please refer to the `outputs/` folder. In there, I've included 2 sample videos

## Simple Video with No Clipping

<table>
  <tr>
    <td><img src="outputs/ronaldo/full_segment.gif" alt="alt text"></td>
    <td><img src="outputs/ronaldo/face.gif" alt="alt text" width="200"></td>
  </tr>
</table>

## Video with Cuts
In this example, the two cuts were generated when the camera cuts to someone else's face.
  
| Original Video                     | Split Video Segments                   | Split face crops                             |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| ![alt text](demo_src/full_video.gif) | ![Tracked Face Video 2 Seg 0](outputs/graham/full_seg_0.gif) | ![Full Video Seg 0](outputs/graham/face_seg_0.gif) |
| ![alt text](demo_src/full_video.gif) | ![Tracked Face Video 2 Seg 1](outputs/graham/full_seg_1.gif) | ![Full Video Seg 1](outputs/graham/face_seg_1.gif) |
# **Adding Smoothing**
I also added some very basic evaluations on the effect of smoothing on the capture of the face. TO do this, I used the Kalman Filter in the DeepSORT algorithm.

The Kalman filter estimates the state of an object (position, velocity, etc.) at any given time based on noisy observations. It uses two main steps:
	
   1.	Prediction: Using the previous state and the object’s velocity, it predicts the object’s state (where it should be in the current frame).
	
   2.	Update: When a new detection is made, the filter corrects the predicted state based on the actual measurement (the bounding box coordinates from the detection).

It stabilizes it so some extent. However, more stabilization through some keypoint based approach might have provided better results.



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
python -m cProfile -o profile.prof run.py --vid_path=data/vid.mp4 --ref_path=data/reference_face/ref2.png --max_frames=50 --frame_tol=5
```
Then to visualize, run,

```snakeviz profile.prof ```

## Comparison of Speed between DeepFace and facenet-pytorch on CPU and GPU

| **Library**          | **FPS (CPU)** | **FPS (GPU)** |
|----------------------|---------------|---------------|
| **DeepFace**         | 0.2-0.5       | cudnn-issues  |
| **facenet-pytorch**  | 0.8-1         | 6-8           |
| **facenet-pytorch + Kalman** | 0.2   | 3-4           |

Since I couldn't get the tensorflow-cudnn compatibility to work out, I ended up going with implementing a pipeline using **facenet-pytorch** (which gives a framerate of around 6-8 FPS on my 1650 RTX Laptop GPU).