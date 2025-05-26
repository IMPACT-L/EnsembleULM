
'''
    This script reads, normalizes, and combines the boxes from
    json files to a proper format for the ensembles script.

'''
import json
import math


def is_valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value) and value is not None

def normalize_box(x, y, w, h, img_width, img_height):
    if not all(is_valid_number(val) for val in [x, y, w, h]):
        return None
    xmin = x / img_width
    ymin = y / img_height
    xmax = (x + w) / img_width
    ymax = (y + h) / img_height

    if xmax > 1:
        xmax = 1
    if ymax > 1:
        ymax = 1
    if xmin<0:
        xmin = 0
    if ymin<0:
        ymin = 0

    return [xmin, ymin, xmax, ymax]

def process_json_file(file_path, img_width, img_height):
    with open(file_path, 'r') as f:
        data = json.load(f)

    boxes = {}
    scores = {}
    labels = {}

    for frame, detections in data.items():
        frame_boxes = []
        frame_scores = []
        frame_labels = []

        for detection in detections:
            if len(detection) != 5 or not all(is_valid_number(val) for val in detection):
                print('might be odd')
                continue

            x, y, w, h, score = detection
            normalized_box = normalize_box(x, y, w, h, img_width, img_height)

            if normalized_box is not None and is_valid_number(score):
                frame_boxes.append(normalized_box)
                frame_scores.append(score)
                frame_labels.append(0)  # All labels are 0

        if frame_boxes:  # Only add the frame if there are valid boxes
            boxes[frame] = frame_boxes
            scores[frame] = frame_scores
            labels[frame] = frame_labels

    return boxes, scores, labels
def find_max_in_boxes_list(boxes_list):
    max_value = float('-inf')
    for frame in boxes_list:
        for model_boxes in frame:
            for box in model_boxes:
                max_value = max(max_value, max(box))
    return max_value

def find_min_in_boxes_list(boxes_list):
    min_value = float('inf')
    for frame in boxes_list:
        for model_boxes in frame:
            for box in model_boxes:
                min_value = min(min_value, min(box))
    return min_value

def combine_json_results(*files, img_width, img_height):
    # Initialize dictionaries to store results
    all_boxes = {}
    all_scores = {}
    all_labels = {}

    # Process each file
    for file in files[0]:
        boxes, scores, labels = process_json_file(file, img_width, img_height)

        # Merge the results into the dictionaries
        for frame in boxes:
            if frame not in all_boxes:
                all_boxes[frame] = []
                all_scores[frame] = []
                all_labels[frame] = []

            all_boxes[frame].append(boxes[frame])
            all_scores[frame].append(scores[frame])
            all_labels[frame].append(labels[frame])

    # Convert the dictionaries to lists, ensuring sorted order of frames
    boxes_list = []
    scores_list = []
    labels_list = []

    for frame in sorted(all_boxes.keys()):
        boxes_list.append(all_boxes[frame])
        scores_list.append(all_scores[frame])
        labels_list.append(all_labels[frame])

    return boxes_list, scores_list, labels_list

