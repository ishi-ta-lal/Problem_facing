import argparse
import copy
import json
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import onnxruntime
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, increment_path


def load_model(onnx_model_path, device):
    logger.info(f"Loading ONNX model from: {onnx_model_path}")
    session = onnxruntime.InferenceSession(onnx_model_path)
    logger.info("Model loaded successfully.")
    return session


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    logger.debug("Scaling coordinates and landmarks...")
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    logger.debug(f"Original coordinates: {coords}")
    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    logger.debug(f"Scaled coordinates: {coords}")
    return coords


def show_results(img, xyxy, conf, landmarks, class_num):
    logger.debug("Drawing results...")
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    logger.debug(f"Original image shape: {img.shape}")
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    logger.debug(f"New shape: {new_shape}")

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    logger.debug(f"Scale ratio: {r}")

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    logger.debug(f"Padding: dw={dw}, dh={dh}, new_unpad={new_unpad}")

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        logger.debug(f"Resized image shape: {img.shape}")

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    logger.debug(f"Final image shape with border: {img.shape}")

    return img, ratio, (dw, dh)

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    logger.debug(f"Initial prediction shape: {prediction.shape}")
    nc = prediction.shape[2] - 15
    xc = prediction[..., 4] > conf_thres
    logger.debug(f"Number of classes: {nc}, confidence threshold: {conf_thres}, IoU threshold: {iou_thres}")

    min_wh, max_wh = 2, 4096
    time_limit = 10.0
    redundant = True
    multi_label = nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        logger.debug(f"Filtered prediction for image {xi}: {x.shape}")

        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 15), device=x.device)
            v[:, :4] = l[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(l)), l[:, 0].long() + 15] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 15:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 15:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 15, None], x[i, 5:15], j[:, None].float()), 1)
        else:
            conf, j = x[:, 15:].max(1, keepdim=True)
            x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 15:16] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue

        c = x[:, 15:16] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        logger.debug(f"Output for image {xi}: {output[xi].shape}")

        if (time.time() - t) > time_limit:
            logger.warning(f'NMS time limit {time_limit}s exceeded')
            break

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    logger.debug(f"Original coordinates: {coords}")
    logger.debug(f"Scaling coordinates from shape {img1_shape} to {img0_shape}")
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    logger.debug(f"Gain: {gain}, pad: {pad}")

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    logger.debug(f"Scaled coordinates: {coords}")

    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

    logger.debug(f"Clipped coordinates: {boxes}")


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)




def detect(
    session,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img
):
    logger.info("Starting detection process...")
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz = (640, 640)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Determine appropriate img_size for resizing
    imgsz = check_img_size(img_size, s=32)  # s should not be zero, choose an appropriate value

    # Dataloader
    if webcam:
        logger.info(f"Loading streams: {source}")
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        logger.info(f"Loading images: {source}")
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    bbox_coordinates = []

    for path, im, im0s, vid_cap in dataset:
        logger.debug(f"Processing {path}...")

        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)
        
        logger.debug(f"Original image shape: {orgimg.shape}")
        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        logger.debug(f"Resized image shape: {img0.shape}")
        imgsz = check_img_size(img_size, s=32)  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        logger.debug(f"Letterboxed image shape: {img.shape}")

        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()
        logger.debug(f"Transposed image shape: {img.shape}")

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        logger.debug(f"Final image tensor shape: {img.shape}")

        # Inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.debug("Running inference...")
        pred = session.run([output_name], {input_name: img.numpy()})[0]
        logger.debug(f"Inference output shape: {pred.shape}")

        # Convert prediction to PyTorch tensor
        pred = torch.from_numpy(pred)
        logger.debug(f"Prediction tensor shape: {pred.shape}")
        logger.debug(f"Prediction : {pred}")
        

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        logger.debug(f"{len(pred[0])} face{'s' if len(pred[0]) != 1 else ''} detected")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            logger.debug(f"Processing detection {i}...")

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                logger.debug(f"Rescaled boxes: {det[:, :4]}")

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], im0.shape).round()
                logger.debug(f"Rescaled landmarks: {det[:, 5:15]}")

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()

                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)
                    logger.debug(f"Detection result: {xyxy}, {conf}, {landmarks}, {class_num}")

                    bbox_coordinates.append({
                        'path': str(p),
                        'coordinates': xyxy,
                        'confidence': conf,
                        'landmarks': landmarks,
                        'class_num': class_num
                    })

            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        logger.error(f"Failed to write video: {e}")

    logger.info("Detection process completed.")
    return bbox_coordinates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/smallface.onnx', help='ONNX model path')
    parser.add_argument('--source', type=str, default='data/images/modi.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--json-output', type=str, default='output2.json', help='path to save JSON output')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session = load_model(opt.weights, device)
    bbox_coords = detect(session, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)

    if opt.json_output:
        with open(opt.json_output, 'w') as f:
            json.dump(bbox_coords, f, default=lambda x: x.tolist())

    for i, bbox in enumerate(bbox_coords):
        logger.info(f"Raw Output from Python ({i}):")
        logger.info(f"Type of 'path': {type(bbox['path'])}")
        logger.info(f"Type of 'coordinates': {type(bbox['coordinates'])}")
        logger.info(f"Type of 'confidence': {type(bbox['confidence'])}")
        logger.info(f"Type of 'landmarks': {type(bbox['landmarks'])}")
        logger.info(f"Type of 'class_num': {type(bbox['class_num'])}")
