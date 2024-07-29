import os
import cv2
import numpy as np
import onnxruntime
import torch
from skimage import transform as trans
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

onnxmodel = os.path.join(os.path.dirname(__file__), "weights/recogmodel.onnx")
session = onnxruntime.InferenceSession(onnxmodel, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def save_array_to_file(tag, array):
    filename = f"{tag}.txt"
    np.savetxt(filename, array.reshape(-1, array.shape[-1]) if len(array.shape) > 1 else array, fmt='%f')
    logging.debug(f"{tag} saved to {filename}")

def preprocess(img, bbox=None, landmark=None, **kwargs):
    logging.debug("Starting preprocessing")

    if isinstance(img, str):
        img = cv2.imread(img)
    logging.debug(f"Initial image shape: {img.shape}")
    save_array_to_file("initial_image", img)

    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
    logging.debug(f"Image size: {image_size}")

    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = np.array(landmark, dtype=np.float32).reshape(5, 2)
        logging.debug(f"src: {src}")
        logging.debug(f"dst: {dst}")

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        logging.debug(f"Transformation matrix M: {M}")

    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        logging.debug(f"Cropped image shape: {ret.shape}")
        save_array_to_file("cropped_image", ret)
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
            logging.debug(f"Resized image shape: {ret.shape}")
            save_array_to_file("resized_image", ret)
        return ret
    else:
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        logging.debug(f"Warped image shape: {warped.shape}")
        save_array_to_file("warped_image", warped)
        return warped

def getEmbSinglefromMxnet(image, bbox, points):
    logging.debug("Starting getEmbSinglefromMxnet")

    nimg11 = preprocess(image, np.array(bbox), np.array(points), image_size='112,112')
    logging.debug(f"Preprocessed image shape: {nimg11.shape}, format: {nimg11.dtype}")
    save_array_to_file("preprocessed_image", nimg11)

    nimg11 = cv2.cvtColor(nimg11, cv2.COLOR_BGR2RGB)
    logging.debug(f"Converted to RGB image shape: {nimg11.shape}, format: {nimg11.dtype}")
    save_array_to_file("rgb_image", nimg11)

    aligned = np.transpose(nimg11, (2, 0, 1))
    input_blob = np.expand_dims(aligned, axis=0)
    logging.debug(f"Created input blob shape: {input_blob.shape}")
    save_array_to_file("input_blob", input_blob)

    onnx_frameEmbedding = torch.FloatTensor(session.run([output_name], {input_name: input_blob.astype(np.float32)})[0])
    logging.debug(f"ONNX model inference result shape: {onnx_frameEmbedding.shape}")
    save_array_to_file("onnx_model_inference_result", onnx_frameEmbedding.numpy())

    onnx_frameEmbedding /= torch.linalg.norm(onnx_frameEmbedding)
    mx_frameEmbedding = onnx_frameEmbedding.view([1, 512])
    logging.debug(f"Normalized embedding shape: {mx_frameEmbedding.shape}")
    save_array_to_file("normalized_embedding", mx_frameEmbedding.numpy())

    return mx_frameEmbedding


def main():
    # Example face detection output
    face_detection_output = [{
        "coordinates": [73.0, 42.0, 144.0, 133.0],
        "confidence": 0.8560894131660461,
        "landmarks": [92.0, 73.0, 122.0, 73.0, 108.0, 88.0, 95.0, 105.0, 123.0, 104.0],
        "class_num": 0.0
    }]

    # Load the original image
    image = cv2.imread('data/images/modi.jpg')

    # Process each detected face
    for face in face_detection_output:
        bbox = face['coordinates']
        landmarks = face['landmarks']

        # Logging the bounding box and landmarks
        logging.debug(f"Processing face with bbox: {bbox}, landmarks: {landmarks}")

        embedding = getEmbSinglefromMxnet(image, bbox, landmarks)
        logging.debug(f'Face embedding: {embedding}')

if __name__ == '__main__':
    main()
