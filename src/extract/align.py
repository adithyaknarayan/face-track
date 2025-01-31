import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
import cv2
from typing import Tuple, List
import numpy as np


class FaceAlignment():
    def __init__(self, target_size:Tuple[int,int]=(160,160), device:str='cuda:0'):
        self.target_size = target_size
        self.device = device


    def align_face_pytorch(self, img:torch.tensor, bboxes:torch.tensor, landmarks: torch.tensor, device:str = 'cuda:0'):
        """
        Code inspired from:
        https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/

        Each face should be of this shape: torch.Size([112, 112, 3])
        """
        aligned_faces = []
        for i in range(bboxes.shape[0]):
            landmark = landmarks[i]-bboxes[i][:2]
            left_eye = landmark[0]
            right_eye = landmark[1]

            # get a line between the eye and rotate
            dx, dy = right_eye - left_eye
            dx = torch.tensor(dx)
            dy = torch.tensor(dy)
            angle = float(torch.atan2(dy, dx) * 180 / torch.pi)

            # rotate face
            face = img[int(bboxes[i][1]):int(bboxes[i][3]),int(bboxes[i][0]):int(bboxes[i][2])]
            if 0 not in face.shape: # sometimes the detector gives invalid bboxes
                rotated_face = rotate(face.permute(2,0,1),angle = angle)
                rotated_face = transforms.Resize(self.target_size)(rotated_face)
                aligned_faces.append(rotated_face.permute(1,2,0))
        return torch.stack(aligned_faces)