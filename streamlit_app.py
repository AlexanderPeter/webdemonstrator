# imports
import os
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# constants
chunk_size = 1024*1024*32
title = "Dental charting automation from panoramic x-rays"
shift = 1
available_models = os.listdir("models/chunks/")
color_palette = [
    "#000000", 
    "#ffff00", "#ff8800", 
    "#ff0000", # "#ff0088", 
    "#ff00ff", # "#8800ff", 
    "#0000ff", # "#0088ff", 
    "#00ffff",  "#00ff88", 
    "#00ff00", "#88ff00"
]
tufts_classes = ["Crown", "Decay", "Apical", "RCT", "Filling", "Abutment", "Pontic", "Implant"]
dentex_classes = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]


# functions
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
    hex_colors,
    thickness=2,
    font_size=1.0,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    alpha=0.25,
    font_thickness=2,
):
    overlay = img.copy()
    for polygon, label in zip(polygons, labels):
        if len(hex_colors) <= label:
            hex_color_polygon = hex_colors[len(hex_colors)-1]
        else:
            hex_color_polygon = hex_colors[label % len(hex_colors)]
        color_polygon = tuple(int(hex_color_polygon[1:][i:i+2], 16) for i in (0, 2, 4))
        if len(polygon) < 3:
            continue
        coords = np.int32([polygon])
        cv2.polylines(img, np.int32([coords]), True, color_polygon, thickness=thickness)
        cv2.fillPoly(
            overlay, np.int32([coords]), color=color_polygon, lineType=cv2.LINE_AA
        )
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    color_font = tuple(int(hex_colors[0][1:][i:i+2], 16) for i in (0, 2, 4))
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


# content
st.set_page_config(page_title=title, page_icon="🦷")
st.title(title)
selected_model = st.sidebar.selectbox("What model should be used?", available_models)
hex_colors = []
hex_colors.append(st.sidebar.color_picker("Color font ", color_palette[0]))
if "numbering" in selected_model:
    hex_colors.append(st.sidebar.color_picker("Color label ", color_palette[1]))
elif "tufts_diseases" in selected_model:
    for i in range(len(tufts_classes)):
        hex_colors.append(st.sidebar.color_picker(f"Color class {i}: {tufts_classes[i]}", color_palette[i+1]))
elif "dentex_diseases" in selected_model:
    for i in range(len(dentex_classes)):
        hex_colors.append(st.sidebar.color_picker(f"Color class {i}: {dentex_classes[i]}", color_palette[2*i+1]))
else:
    for i in range(len(color_palette) -1 ):
        hex_colors.append(st.sidebar.color_picker(f"Color class {i}", color_palette[i+1]))

img_file_buffer = st.file_uploader(
    "Choose image file to detect",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if img_file_buffer is not None:
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)
    polygons, labels = predict_yolo(selected_model, open_cv_image)
    img = draw_instances(
        open_cv_image.copy(), polygons, labels, hex_colors
    )
    st.image(
        img,
        caption=[f"Prediction with {selected_model}"],
        channels="RGB",
    )
