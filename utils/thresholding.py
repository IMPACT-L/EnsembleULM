
'''
    This script thresholds the detection both for one method (model)
    and for all of them.

'''
from ensemble_boxes import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from utils.json2boxes import combine_json_results


def threshold_one_model(boxes_list, scores_list, labels_list,threshold):


        thresholded_frame_boxes = []
        thresholded_frame_scores = []
        thresholded_frame_labels = []
        for frame_boxes, frame_scores, frame_labels in zip(boxes_list, scores_list, labels_list):
            model_thresholded_boxes = []
            model_thresholded_scores = []
            model_thresholded_labels = []
            for box, score, label in zip(frame_boxes, frame_scores, frame_labels):
                if score >= threshold:
                    model_thresholded_boxes.append(box)
                    model_thresholded_scores.append(score)
                    model_thresholded_labels.append(label)

            thresholded_frame_boxes.append(model_thresholded_boxes)
            thresholded_frame_scores.append(model_thresholded_scores)
            thresholded_frame_labels.append(model_thresholded_labels)

        return thresholded_frame_boxes, thresholded_frame_scores, thresholded_frame_labels


def threshold_detections(boxes_list, scores_list, labels_list, threshold):
    thresholded_boxes = []
    thresholded_scores = []
    thresholded_labels = []

    for frame_boxes, frame_scores, frame_labels in zip(boxes_list, scores_list, labels_list):
        thresholded_frame_boxes = []
        thresholded_frame_scores = []
        thresholded_frame_labels = []

        for model_boxes, model_scores, model_labels in zip(frame_boxes, frame_scores, frame_labels):
            model_thresholded_boxes = []
            model_thresholded_scores = []
            model_thresholded_labels = []

            for box, score, label in zip(model_boxes, model_scores, model_labels):
                if score >= threshold:
                    model_thresholded_boxes.append(box)
                    model_thresholded_scores.append(score)
                    model_thresholded_labels.append(label)

            thresholded_frame_boxes.append(model_thresholded_boxes)
            thresholded_frame_scores.append(model_thresholded_scores)
            thresholded_frame_labels.append(model_thresholded_labels)

        thresholded_boxes.append(thresholded_frame_boxes)
        thresholded_scores.append(thresholded_frame_scores)
        thresholded_labels.append(thresholded_frame_labels)

    return thresholded_boxes, thresholded_scores, thresholded_labels

## Usage:

# img_width = 512  # Replace with your image width
# img_height = 512  # Replace with your image height
#
# boxes_list, scores_list, labels_list = combine_json_results(
#     'Slice4DEDETRpred_dict_BBs_V0BeforeServer.json',
#     'Slice4DEDETRpred_dict_BBs_V185beforePC.json',
#     img_width,
#     img_height
# )
# threshold = 0.5  # Set your desired threshold here
#
# thresholded_boxes, thresholded_scores, thresholded_labels = threshold_detections(
#     boxes_list, scores_list, labels_list, threshold
# )

## Print results (you might want to adjust this based on your needs)
# print("Thresholded boxes:")
# print(thresholded_boxes)
# print("\nThresholded scores:")
# print(thresholded_scores)
# print("\nThresholded labels:")
# print(thresholded_labels)
# print("end")