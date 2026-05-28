"""
This is a helper file that contains varying helper functions after refactoring.
"""

from ast import literal_eval
from base64 import b64decode, b64encode
from binascii import Error as BinasciError
from cv2 import (
    imdecode,
    resize,
    cvtColor,
    IMREAD_COLOR,
    COLOR_BGR2RGB,
    INTER_AREA,
    error as CVError,
)
from numpy import frombuffer, uint8, ndarray, expand_dims, float32
from onnx import load
from google.protobuf.message import DecodeError


def check_model(model_file):
    """
    Checks the model.

    Return: Image size and available labels within the model
    """
    metadata, image_size, label = None, None, None

    if not isinstance(model_file, str):
        raise TypeError("Incorrect model format. REQUIRED: a string as the name.")
    if not model_file.endswith(".onnx"):
        raise TypeError("Incorrect model format. REQUIRED: .onnx.")

    try:
        model = load(model_file)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
    except FileNotFoundError as exception:
        raise FileNotFoundError(f"Model file not found: {model_file}") from exception
    except DecodeError as exception:
        raise ValueError(f"File is not a valid ONNX model: {model_file}") from exception

    if "names" not in metadata:
        raise KeyError("Model doesn't contain object names.")
    if "imgsz" not in metadata:
        raise KeyError("Model doesn't contain image size.")

    image_size: list = metadata["imgsz"][1:-1].split(",")
    label = literal_eval(metadata["names"])
    return int(image_size[0]), label


def convert_binary_str_to_inference_data(binary_as_string: str) -> tuple[ndarray, list]:
    """
    This function takes in a binary string of an image.

    Return: Numpy.ndarray as the converted image.
    """
    np_array = binary_str_to_numpy_array(binary_as_string=binary_as_string)
    return retrive_numpy_array_image_for_model_inference(np_array)


def scale_coords(coords: list, original_shape: list[int, int], size: int = 480) -> list:
    """
    Takes a list of coords [x,y,width,height], original image [width, height]
    and the trained models image size.

    Return: List of coords that have been adjusted to fit on the original image size.
    """
    x_scale, y_scale = original_shape[0] / size, original_shape[1] / size

    coords[0] *= x_scale
    coords[1] *= y_scale
    coords[2] *= x_scale
    coords[3] *= y_scale
    coords[3] = min(coords[3], original_shape[0])

    return [int(max(i, 0)) for i in coords]


def json_detection_response(original_size: list, detections, label, image_size):
    """
    Filters through the detected data. Creates a list of found elements.

    Return: List. Returns an empty list or a maximum of 10 objects in the list.
    """
    percentage_threshold: float = 25
    response_list = []
    for _, detection in enumerate(detections):
        x1, y1, x2, y2, score, class_id = detection
        score = int(score * 100)
        if score > percentage_threshold:
            box = [x1, y1, x2, y2]
            box = scale_coords(box, original_size, size=image_size)
            label_name = label[int(class_id)]
            if len(response_list) > 10:
                break
            response_list.append(
                prediction_to_json(score=score, label=label_name, coords=box)
            )
    return response_list


def prediction_to_json(score, label, coords):
    """
    A singular JSON dictionary that gets filled with score, label and coords input parameters.

    Return: JSON dictionary of a found object.
    """
    return {
        "coords": coords,
        "label": label,
        "score": score,
    }


def binary_str_to_numpy_array(binary_as_string: str) -> ndarray:
    """
    Converts a binary string of an image to a numpy array.

    Return: numpy.ndarray
    """
    if not isinstance(binary_as_string, str):
        raise TypeError("Incorrect input format. REQUIRED: a str.")
    if not binary_as_string.startswith("iVBORw") and not binary_as_string.startswith(
        "/9j/"
    ):
        raise TypeError("Incorrect file extension. REQUIRED: .JPG / .PNG.")

    try:
        decode_binary_string = b64decode(binary_as_string, validate=True)
        return frombuffer(decode_binary_string, uint8)
    except BinasciError as exception:
        raise BinasciError("Base64 contains non valid characters") from exception
    except ValueError as exception:
        raise ValueError("Base64 string is less than 2 characters") from exception


def retrive_numpy_array_image_for_model_inference(
    numpy_ndarray: ndarray,
) -> tuple[ndarray, list]:
    """
    Prepares numpy.ndarray data for onnx runtime inference.
    Takes image as MatLike object. Resizes, converts colour channels from BGR to RGB.
    Transpose data dimension order to pyTorch requirement,
    and adds an extra dimensions to the start.

    [For example (batch_size, channels, height, width) -> (1, 3, 500, 920)]

    Return: Reshaped numpy.ndarray, [original_image_width, original_image_height]
    """
    try:
        image = imdecode(numpy_ndarray, IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image. Potentially invalid format.")

        image_resize = resize(image, (480, 480), interpolation=INTER_AREA)
        image_cvt = cvtColor(image_resize, COLOR_BGR2RGB)
        color_converted_image = image_cvt.transpose((2, 0, 1)).astype(float32)
        color_converted_image /= 255.0
        return expand_dims(color_converted_image, axis=0), [
            image.shape[1],
            image.shape[0],
        ]

    except CVError as exception:
        raise ValueError("OpenCV failed to process image") from exception
