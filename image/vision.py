"""
This file contains the primary functions that handle the entire detection process.
"""

from onnxruntime import InferenceSession, OrtValue
from helpers import (
    convert_binary_str_to_inference_data,
    json_detection_response,
    check_model,
)


def process_model(model_name: str = "mil-yolov26n.onnx"):
    """
    Validates the model and returns inference session with image size and baked in model labels.

    Return: InferenceSession, image size, labels
    """
    img_size, label_names = check_model(model_name)
    model_session = InferenceSession(model_name, providers=["CPUExecutionProvider"])
    return model_session, img_size, label_names


def process_image(binary_data) -> list:
    """
    Validates and converts the binary image string to a OrtValue for inference.

    Return: OrtValue of binary string, original image size
    """
    processed_image, original_size = convert_binary_str_to_inference_data(binary_data)
    ort_value = OrtValue.ortvalue_from_numpy(processed_image)
    return ort_value, original_size


def detect_objects(model_session, ort_value, original_size, label, image_size):
    """
    Runs inference model.

    Returns: List - Empty list if none found, maximum 10 if some objects found.
    """
    session_name = model_session.get_inputs()[0].name
    results = model_session.run(None, {session_name: ort_value})

    detections = results[0][0]
    return json_detection_response(original_size, detections, label, image_size)
