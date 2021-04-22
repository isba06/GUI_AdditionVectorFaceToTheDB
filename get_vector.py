import math
import os
import pickle
import tarfile
import time

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes, get_all_face_attributes, 

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()
    img0 = cv.imread("test/2723565.jpg")
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    imgs[0] = img0
    output = model(imgs)

    feature0 = output[0].cpu().numpy()
    x0 = feature0 / np.linalg.norm(feature0)
