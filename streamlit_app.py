import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

chunk_size = 1024*1024*32
title = "Dental charting automation from panoramic x-rays"
shift = 1

def merge_chunks(file_path):
    folder_path = file_path + "_chunks"
    file_merged = open(file_path, "wb")
    for file_name in os.listdir(folder_path):
        file_chunk = open(os.path.join(folder_path, file_name), "rb")
        content = file_chunk.read(chunk_size)
        file_merged.write(content)
    file_merged.close()

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

def predict_yolo(model_name, image, threshold=0.5):
    file_path = os.path.join("models", model_name)
    if not os.path.isfile(path):
        merge_chunks(file_path)
    model = YOLO(file_path, task="segment")
    outputs = model.predict(image)
    classes = outputs[0].boxes.cls.tolist()
    print(classes)
    scores = outputs[0].boxes.conf.tolist()
    print(scores)
    boxes = outputs[0].boxes.xyxy.tolist()
    print(boxes)
    masks = (
        [mask.tolist() for mask in outputs[0].masks.xy]
        if outputs[0].masks is not None
        else []
    )
    labels = []
    polygons = []
    for score, label, box, mask in zip(scores, classes, boxes, masks):
        if threshold < score:
            if mask is None:
                polygons.append(
                    [
                        (box[0], box[1]),
                        (box[2], box[1]),
                        (box[2], box[3]),
                        (box[0], box[3]),
                    ]
                )
            else:
                polygons.append(mask)
            labels.append(int(label + shift))
    return polygons, labels

def draw_instances(
    img,
    polygons,
    labels,
    color_polygon,
    color_font,
    thickness=3,
    font_size=1.0,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    alpha=0.15,
):
    overlay = img.copy()
    for polygon, label in zip(polygons, labels):
        coords = np.int32([polygon])
        cv2.polylines(img, np.int32([coords]), True, color_polygon, thickness=thickness)
        cv2.fillPoly(
            overlay, np.int32([coords]), color=color_polygon, lineType=cv2.LINE_AA
        )
        x_list = [coord[0] for coord in polygon]
        y_list = [coord[1] for coord in polygon]
        x_min = min(x_list)
        y_min = min(y_list)
        x_max = max(x_list)
        y_max = max(y_list)
        width = x_max - x_min
        height = y_max - y_min
        text = str(label)
        textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
        cv2.putText(
            img,
            text,
            (
                int(x_min + width / 2 - textsize[0] / 2),
                int(y_min + height / 2 - textsize[1] / 2),
            ),
            font,
            font_size,
            color_font,
            thickness,
        )
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img


st.set_page_config(page_title=title, page_icon="🦷")
st.title(title)

img_file_buffer = st.file_uploader(
    "Choose image file to detect",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)
if img_file_buffer is not None:
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)

    polygons, labels = predict_yolo("yolo_segmentation_tufts_diseases.pt", open_cv_image)
    img = draw_instances(
        open_cv_image.copy(), polygons, labels, color_polygon=(0, 220, 0), color_font=(0, 0, 0)
    )
    st.image(
        img,
        caption=[
            "Tufts-diseases",
        ],
        channels="RGB",
    )
