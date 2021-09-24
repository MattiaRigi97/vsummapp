
# python inference.py anchor-based --model-dir ../models/ab_mobilenet/ --splits ../splits/tvsum.yml ../splits/summe.yml --cnn mobilenet --segment_algo kts --num-feature 1280 --num-head 10 --num-hidden 128 --nms-thresh 0.4 --video_name Fire_Domino.mp4

## PACKAGE
import logging
import numpy as np
from PIL import Image as pilImage

# Segment Detection based on KTS
from segmentation.kts.cpd_auto import cpd_auto

# Helpers functions
from helpers import vsumm_helper, bbox_helper
from helpers.data_helper import scale

from tialib import *

# Features Extraction functions
from sklearn import preprocessing
# Torch Modules
import torch
from torchvision import transforms


logger = logging.getLogger()

# Define the preprocessing function for images
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(model, frames, shape):
    with torch.no_grad():
        seq = []
        for frame in frames:
            frame = frame[:,:,::-1]
            im = pilImage.fromarray(frame)
            input_tensor = preprocess(im)
            input_batch = input_tensor.unsqueeze(0)
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                output = model(input_batch)
            seq = np.append(seq, np.array(output[0].cpu()))

        seq = np.reshape(seq, (-1, shape))
        seq = np.asarray(seq, np.float32)
        seq = preprocessing.normalize(seq)

        return seq


def inference(model, feat_extr, shape, filename, frames, n_frame_video, seg_algo, nms_thresh, device, proportion):
    
    model.eval()
    feat_extr.eval()

    with torch.no_grad():

        seq = extract_features(feat_extr, frames, shape)

        seq_len = len(seq)

        # Model prediction
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)
        pred_cls, pred_bboxes = model.predict(seq_torch)
        # Compress and round all value between 0 and seq_len
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)
        # Apply NMS to pred_cls (condifence scores of segments) and to LR bboxes
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
        picks = np.arange(0, seq_len) * 15 # array([    0,    15,    30,    45,    60, ...])

        # VIDEO SEGMENTATION
        kernel = np.matmul(seq, seq.T) # Matrix product of two arrays
        kernel = scale(kernel, 0, 1)
        change_points, _ = cpd_auto(K = kernel, ncp = seq_len - 1, vmax = 1 ) # Call of the KTS Function
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frame_video)) # add 0 and the last frame
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames)).T
        # print("cps shape: " + str(change_points.shape) + "\n")
        n_frame_per_seg = end_frames - begin_frames  # For each segment, calculate the number of frames
        # print("nfps: " + str(n_frame_per_seg))   
        # print("nfps shape: " + str(n_frame_per_seg.shape) + "\n")   

        # Convert predicted bounding boxes to summary
        pred_summ = vsumm_helper.bbox2summary(seq_len, pred_cls, pred_bboxes, change_points, n_frame_video, n_frame_per_seg, picks, proportion)

    return pred_summ

