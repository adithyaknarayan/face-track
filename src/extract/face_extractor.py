import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, fixed_image_standardization
from typing import Tuple
from src.extract.align import FaceAlignment

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2 as cv
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from typing import Tuple
from src.extract.align import FaceAlignment
from typing import List
import numpy as np

class FaceExtractor:
    def __init__(self, target_size:Tuple[int, int] = (160,160), device:str = 'cuda:0') -> None:
        self.target_size = target_size
        self.model = MTCNN(margin=40,
                    factor=0.6,
                    keep_all=True,
                    device=device).eval()
        self.device = device
        self.trans = transforms.Compose([
                        transforms.Resize(target_size),
                        fixed_image_standardization
                    ])
        self.face_aligner = FaceAlignment(self.target_size, device=self.device)


    def extract_and_align(self, img:np.ndarray, ref_img:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        bboxes, probs, landmarks = self.model.detect(img, landmarks=True) # this function annoyingly only return np arrays
        # breakpoint()
        if isinstance(bboxes, np.ndarray):
            bboxes_tensor = torch.tensor(bboxes.astype('float')).to(self.device)
            landmarks_tensor = torch.tensor(landmarks.astype('float')).to(self.device)
            aligned_faces = self.face_aligner.align_face_pytorch(
                img=img,
                bboxes=bboxes_tensor,
                landmarks = landmarks_tensor
            ) # output shape: torch.Size([5, 3, 112, 112])

            #TODO: try to project facial area from original image
            # postprocess
            aligned_faces = self.trans(aligned_faces.permute(0,3,1,2))
            ref_img = self.trans(ref_img.permute(2,0,1)).unsqueeze(0)
            # breakpoint()
            return aligned_faces, bboxes, ref_img, probs
        else:
            return None, None, None, None
        