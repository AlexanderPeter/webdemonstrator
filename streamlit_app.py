import os
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

chunk_size = 1024*1024*32
title = "Dental charting automation from panoramic x-rays"
shift = 1
st.set_page_config(page_title=title, page_icon="🦷")

def merge_chunks(model_name):
    st.toast(f"Preparing {model_name}")
    file_path = os.path.join("models", model_name)
    file_merged = open(file_path, "wb")
    counter = 0
    chunk_path = os.path.join("models", "chunks", model_name, f"chunk{counter}.txt")
    while os.path.isfile(chunk_path):
        file_chunk = open(chunk_path, "rb")
        content = file_chunk.read(chunk_size)
        file_merged.write(content)
        counter += 1
        chunk_path = os.path.join("models", "chunks", model_name, f"chunk{counter}.txt")
    file_merged.close()

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

def predict_yolo(model_name, image, threshold=0.5):
    file_path = os.path.join("models", model_name)
    if not os.path.isfile(file_path):
        merge_chunks(model_name)
    model = YOLO(file_path, task="segment")
    outputs = model.predict(image)
    classes = outputs[0].boxes.cls.tolist()
    scores = outputs[0].boxes.conf.tolist()
    boxes = outputs[0].boxes.xyxy.tolist()
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
    color_polygon=(255, 255, 0),
    color_font=(0, 0, 0),
    thickness=2,
    font_size=1.0,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    alpha=0.25,
    font_thickness=2,
):
    overlay = img.copy()
    for polygon, label in zip(polygons, labels):
        if len(polygon) < 3:
            continue
        coords = np.int32([polygon])
        cv2.polylines(img, np.int32([coords]), True, color_polygon, thickness=thickness)
        cv2.fillPoly(
            overlay, np.int32([coords]), color=color_polygon, lineType=cv2.LINE_AA
        )
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for polygon, label in zip(polygons, labels):
        if len(polygon) < 3:
            continue
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
            font_thickness,
        )
    return img


st.title(title)

available_models = os.listdir("models/chunks/")
selected_model = st.selectbox("What model should be used?", available_models)

hex_color_polygon = st.color_picker('Label color', '#ffff00')
hex_color_font = st.color_picker('Font color', '#000000')


img_file_buffer = st.file_uploader(
    "Choose image file to detect",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if img_file_buffer is not None:
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)

    polygons, labels = predict_yolo(selected_model, open_cv_image)
    color_polygon = tuple(int(hex_color_polygon[1:][i:i+2], 16) for i in (0, 2, 4))
    color_font = tuple(int(hex_color_font[1:][i:i+2], 16) for i in (0, 2, 4))
    img = draw_instances(
        open_cv_image.copy(), polygons, labels, color_polygon=color_polygon, color_font=color_font
    )
    st.image(
        img,
        caption=[f"Prediction with {selected_model}"],
        channels="RGB",
    )
