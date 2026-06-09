"""
Experimental file.
"""

import os
from cv2 import imread, imwrite, samples, resize, INTER_AREA


def convert_images(input_path, output_path, image_prefix):
    processed_count: int = 0
    if not os.path.exists(input_path):
        raise NotADirectoryError("Input path doesn't exist")

    if os.path.exists(output_path):
        raise IsADirectoryError(
            "Output directory already exist. Make new output directory."
        )

    os.mkdir(output_path)

    image_list = os.listdir(input_path)
    for image_index, image_name in enumerate(image_list):
        try:
            path = str(input_path + "/" + image_name)
            scaler = scaleImage(path)
            scaled_image = resize_image(path, scaler)
            imwrite(f"{output_path}/{image_prefix}_{image_index}.jpeg", scaled_image)
            processed_count += 1
        except (Exception, AttributeError, ValueError) as exception:
            print(f"Failed to process '{image_name}' ")

    print(f"Scaled {processed_count} images")


def scaleImage(image_path_with_file: str) -> float:
    targetW, targetH = 1024, 576
    img = imread(samples.findFile(image_path_with_file))

    if img is None:
        raise ValueError(f"Failed to read image: {image_path_with_file}")

    w, h = img.shape[1::-1]
    wW, hH = targetW / w, targetH / h
    scale = (wW + hH) / 2
    return scale


def resize_image(input_image, scale: float):
    img = imread(input_image)
    return resize(img, None, fx=scale, fy=scale, interpolation=INTER_AREA)


if __name__ == "__main__":
    args_input_path = (
        "E:/Backup/Coding backup/tank detection backup/tankbackup/rus_mstas"
    )
    args_image_prefix = "rus_mstas"
    args_output_path = "processed_images"

    convert_images(args_input_path, args_output_path, args_image_prefix)
