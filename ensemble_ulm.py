# ---------------------------------------------------------------------------
# Project   : Ensemle learning for ULM
# Author    : Sepideh K. Gharamaleki  |  sepidehkhakzad@gmail.com
# Created   : 2025-05-26
# Version   : 0.1.0
#
# Description
# -----------
# Utility functions for adaptive thresholding, non-maximum suppression,
# weighted-box fusion, and related post-processing of bounding boxes in
# super-resolution ultrasound microbubble localisation.
#
# Disclaimer
# ----------
# This software is provided “AS IS”, without warranty of any kind, express
# or implied. In no event shall the author or copyright holder be liable for
# any claim, damages or other liability arising from, out of, or in connection
# with the software or the use or other dealings in the software.
#
# Citation
# --------
# If you use this code—or any part of it—in academic work or a commercial
# product, **please cite**:
#
# [1] S. K. Gharamaleki, B. Helfeld, and H. Rivaz,
    # “Ensemble Learning for Microbubble Localization in Super-Resolution Ultrasound,”
    # in *Proc. 2025 IEEE 22nd Int. Symp. Biomed. Imaging (ISBI)*,
    # Houston, TX, USA, 14–17 Apr. 2025.
    # doi: 10.1109/ISBI60581.2025.10980786

# [2] S. K. Gharamaleki, **Ensemble‑ULM**  GitHub repository, 2025.
# URL: https://github.com/sepidehkhakzad/EnsembleULM
#
#
# License
# -------
# SPDX-License-Identifier: MIT
#
# The full MIT licence text is included in the project-level **LICENSE** file.
# ---------------------------------------------------------------------------


import os.path
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import savemat
from ensemble_boxes import *
from utils.dynamic_ensemble import *
import tqdm
import numpy as np
import h5py
import re
from utils.json2boxes import combine_json_results, find_max_in_boxes_list, find_min_in_boxes_list
from utils.thresholding import threshold_detections, threshold_one_model
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import scipy.io as sio
from omegaconf import OmegaConf, DictConfig


def save_boxes_to_file_for_method(cfg, method_name, all_boxes, all_scores, all_labels):
    print("Saving boxes for method:", method_name)
    filename = f"boxes_{method_name.replace(' ', '_').lower()}.txt"
    filePath = cfg.output_path + filename
    if not os.path.exists(filePath):
        os.makedirs(os.path.dirname(filePath), exist_ok=True)

    with open(filePath, 'w') as f:
        for frame_idx, (boxes, scores, labels) in enumerate(zip(all_boxes, all_scores, all_labels)):
            f.write(f"Frame {frame_idx + 1}\n")
            for box, score, label in zip(boxes, scores, labels):
                f.write(f"Box: {box}, Score: {score}, Label: {label}\n")
            f.write("\n")

def calculate_kde_density(data2, mapX, mapZ):
    # data2 is a NumPy array or pandas DataFrame and
    # columns 3 (z) and 5 (x) need to be extracted for the calculation.

    # Extracting columns 3 and 5 (z and x values)
    x_values = data2[:, 2]  # Assuming z is in column index 3
    z_values = data2[:, 4]  # Assuming x is in column index 5

    # Stack x and z values for 2D KDE calculation
    values = np.vstack([x_values, z_values])

    # Calculate Gaussian KDE
    kde = gaussian_kde(values)

    # Evaluate the density on a grid defined by mapX and mapZ
    xx, zz = np.meshgrid(mapX, mapZ)
    grid_coords = np.vstack([xx.ravel(), zz.ravel()])
    density = kde(grid_coords).reshape(xx.shape)

    return density

def calculate_KDE(thresholded_boxes_list_for_one_method):
    # region calculate kde
    centers = []
    for frame in thresholded_boxes_list_for_one_method:
        for box in frame:
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            centers.append([center_x, center_y])

    centers = np.array(centers)
    # Calculate KDE
    kde = gaussian_kde(centers.T)

    ## Create a grid of points
    # x, y = np.mgrid[0:1:200j, 0:1:200j]
    # positions = np.vstack([x.ravel(), y.ravel()])
    # z = np.reshape(kde(positions), x.shape)

    # # Plot the KDE
    # fig, ax = plt.subplots(figsize=(10, 10))
    # im = ax.imshow(np.rot90(z), cmap=plt.cm.viridis, extent=[0, 1, 0, 1], aspect='auto')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('KDE of Detected Boxes Across All Frames')
    # plt.colorbar(im)
    #
    # # Optionally, plot the original box centers
    # ax.scatter(centers[:, 1], centers[:, 0], color='red', s=1, alpha=0.1)

    # plt.show()
    return kde
    # endregion

def find_optimal_score_threshold(all_ground_truths, all_predictions, offset_x,
                                  offset_y, all_scores, score_threshold_range, tol):
    """
       Finds the optimal score threshold for binary classification of detections by maximizing the F1 score.
       Also optionally plots the ROC curve for visual performance evaluation.

       For each threshold, it does the following:
            - Adjusts the ground truth positions using offset_x, offset_y
            - Computes the Euclidean distance between each adjusted GT and all predicted positions (using cdist)
            - For each predicted point, if its closest GT point is within tol, it is counted as a true (1); otherwise false (0). This becomes y_true
            - Predictions with score ≥ current threshold are set to 1 in y_pred; others are 0
            - Calculates the F1-score (harmonic mean of precision and recall) for that threshold
            - Keeps track of the threshold that gives the best F1-score


       Args:
           all_ground_truths (list): Ground truth data in the form [(frame_id, ..., x, y), ...].
           all_predictions (list of lists): Predicted coordinates per frame, e.g., [[(x1, y1), (x2, y2), ...], ...].
           offset_x (float): Horizontal offset to apply to ground truth coordinates.
           offset_y (float): Vertical offset to apply to ground truth coordinates.
           all_scores (list of lists): Confidence scores corresponding to predictions per frame.
           score_threshold_range (iterable): Range of score thresholds to evaluate, e.g., np.arange(0.1, 1.0, 0.05).
           tol (float): Distance tolerance (in pixels or units) within which a prediction is considered a match to a ground truth.

       Returns:
           best_threshold (float): Score threshold that yields the highest F1 score.
           best_f1_score (float): Highest F1 score achieved.
       """
    
    
    best_threshold = None
    best_f1_score = 0

    f1_scores = []
    thresholds = []
    all_y_true = []
    all_y_pred = []

    # Iterate over the score thresholds
    for score_threshold in score_threshold_range:
        # Compute y_true and y_pred for the current threshold
        current_y_true = []
        current_y_pred = []
        for i, (predictions, scores) in enumerate(zip(all_predictions, all_scores)):
            gt_pred = [t[-2:] for t in all_ground_truths if t[0] == 400 + i]

            # Add offset to all ground truth
            adjusted_ground_truths = [(x + offset_x, y + offset_y) for x, y in gt_pred]

            # Compute distances and determine y_true
            distances = cdist(np.array(adjusted_ground_truths), np.array(predictions))
            min_distances = np.min(distances, axis=0)
            y_true = (min_distances <= tol).astype(int)

            # Compute y_pred
            y_pred = (np.array(scores) >= score_threshold).astype(int)

            current_y_true.extend(y_true)
            current_y_pred.extend(y_pred)

        all_y_true.extend(current_y_true)
        all_y_pred.extend(current_y_pred)

        # Compute F1 score for the current threshold
        f1 = f1_score(current_y_true, current_y_pred)

        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = score_threshold

        f1_scores.append(f1)
        thresholds.append(score_threshold)

    # # Plot the ROC curve (optional)
    # # Compute ROC values for the range of thresholds
    # fpr, tpr, roc_thresholds = roc_curve(all_y_true, all_y_pred)
    # # Plot the ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(all_y_true, all_y_pred))
    # plt.plot([0, 1], [0, 1], 'k--')
    # # Plot the optimal threshold
    #
    # plt.plot(0, 0, 'ro', markersize=0, label=f'Optimal Threshold (={best_threshold:.2f})')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    return best_threshold, best_f1_score

def plot_instance(gt, pred):
    # Create a figure
    plt.figure(figsize=(12, 5))

# Extract and plot the predicted and ground truth centers
    pred_x = [p[0] for p in pred]
    pred_y = [p[1] for p in pred]
    gt_x = [g[0] for g in gt]
    gt_y = [g[1] for g in gt]

    plt.scatter(pred_x, pred_y, color='red', label='Predicted Center')
    plt.scatter(gt_x, gt_y, color='blue', label='Ground Truth')
    # Add legend and show the plot
    plt.legend()
    plt.tight_layout()
    plt.show()


def calculate_precision_recall(gt_bbox, pred, tol,dx, dz):
    tp = 0
    fp = 0
    fn = 0

    pred_in = pred

    tp_pairs = []
    fp_points = []
    fn_points = []
    squared_errors = []

    for i in range(len(gt_bbox)):
        gt_center_x = gt_bbox[i][0]
        gt_center_y = gt_bbox[i][1]

        min_distance = float('inf')
        min_index = -1
        for j in range(len(pred_in)):
            distance = ((gt_center_x - 2* (dx) - pred_in[j][0]) ** 2 + (
                        gt_center_y - (2 * dz) - pred_in[j][1]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                min_index = j

        if min_distance <= tol:
            tp += 1
            tp_pairs.append((gt_center_x, gt_center_y, pred_in[min_index][0], pred_in[min_index][1]))
            squared_errors.append(min_distance ** 2)
            pred_in.pop(min_index)
        else:
            fn += 1
            fn_points.append((gt_center_x, gt_center_y))

    fp = len(pred_in)
    fp_points = pred_in

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else 0
    # plot_instance(gt_bbox, pred_in)

    return precision, recall, rmse, tp_pairs, fn_points, fp_points

def normalized_to_realworld(normalized_box, image_width, image_height, dx, dz, mapX, mapZ):
    """Convert normalized [xmin, ymin, xmax, ymax] to real-world coordinates"""
    xmin = normalized_box[0] * image_width * dx + mapX[0]
    ymin = normalized_box[1] * image_height * dz + mapZ[0]
    xmax = normalized_box[2] * image_width * dx + mapX[0]
    ymax = normalized_box[3] * image_height * dz + mapZ[0]
    return [xmin, ymin, xmax, ymax]

def bbox_to_center_realworld(bbox):
    """Convert real-world [xmin, ymin, xmax, ymax] to [center_x, center_y]"""
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def calculate_metrics_for_frame(gt_bbox, pred_bboxes, image_width, image_height, dx, dz, mapX, mapZ, tol, scores):
    pred_centers = []
    for bbox in pred_bboxes:
        realworld_bbox = normalized_to_realworld(bbox, image_width, image_height, dx, dz, mapX, mapZ)
        center = bbox_to_center_realworld(realworld_bbox)
        pred_centers.append(center)

    return calculate_precision_recall(gt_bbox, pred_centers, tol, dx, dz)

def calculate_maxAUC(gt_bbox, pred_bboxes, image_width, image_height, dx, dz, mapX, mapZ, tol, scores):

    all_pred_centers = []
    for frame_boxes2 in pred_bboxes:
        pred_centers = []
        for bbox in frame_boxes2:
            realworld_bbox = normalized_to_realworld(bbox, image_width, image_height, dx, dz, mapX, mapZ)
            center = bbox_to_center_realworld(realworld_bbox)
            pred_centers.append(center)
        all_pred_centers.append(pred_centers)


    optimal_threshold, max_auc_roc = find_optimal_score_threshold(gt_bbox, all_pred_centers, 2 * dx,
                                                                  2 * dz, scores, threshold_range, tol)
    return optimal_threshold

def calculate_and_plot_kde(frame_boxes):
    # Convert boxes to center points
    centers = []
    for box in frame_boxes:
        centers.append(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))

    # Create a grid of points
    x, y = np.mgrid[0:1:100j, 0:1:100j]
    positions = np.vstack([x.ravel(), y.ravel()])

    # Calculate KDE
    kde = gaussian_kde(centers)
    z = np.reshape(kde(positions), x.shape)



    # plt.show()
    return z

# Load the YAML file -------------------------------------------------
cfg: DictConfig = OmegaConf.load("config_invivo.yaml")
img_width = cfg.img_width
img_height = cfg.img_height

# -----------------------------------------------------------------------------
# STEP 1 ─ Load predictions from five Deformable DETR models
# -----------------------------------------------------------------------------
# • Each JSON file contains per-frame bounding-box (BB) predictions.
# • combine_json_results() returns 3 parallel lists
#     – boxes_list  : list[length = n_frames] of list[length = n_models] of boxes
#     – scores_list : same shape, but confidence scores
#     – labels_list : same shape, but class labels
#   so that   boxes_list[f][m]  gives the boxes for frame f from model m.

filePath = cfg.predictions_path
fileNames = cfg.predictions_name
files =  [filePath + name for name in fileNames]

boxes_list, scores_list, labels_list = combine_json_results(
    files,
    img_width=img_width,
    img_height=img_height
)

# -----------------------------------------------------------------------------
# STEP 2 ─ Load simulation metadata (pixel-to-physical mapping, centre frequency)
# ---------------------------------------------------------------------------
try:                      # v7.3 (HDF-5)
    metadata = h5py.File(cfg.metadata_path, "r")
    mapX = metadata['PxSet']['mapX'][:]      # x-coordinate grid (mm)
    mapZ = metadata['PxSet']['mapZ'][:]      # z-coordinate grid (mm)
    dx   = metadata['PxSet']['dx'][0][0]     # pixel size in x (mm)
    dz   = metadata['PxSet']['dz'][0][0]     # pixel size in z (mm)
    cf   = metadata['SimSet']['centre_frequency'][0][0]  # probe centre freq (MHz)
    
except (OSError, IOError):  # ≤ v7.2
    metadata = sio.loadmat(cfg.metadata_path, simplify_cells=True)
    mapX = metadata['PxSet']['mapX']
    mapZ = metadata['PxSet']['mapZ']
    dx = metadata['PxSet']['dx']
    dz = metadata['PxSet']['dz']
    cf = metadata['SimSet']['centre_frequency']

# Spatial tolerance: half an acoustic wavelength at the probe’s centre frequency
lambdaa = 1540 / cf            # λ = c / f  (speed of sound ≈ 1540 m s⁻¹)
tol     = lambdaa / 2

# -----------------------------------------------------------------------------
# STEP 3 ─ Extract ground-truth positions for frames ≥ 400 (simulation choice)
# -----------------------------------------------------------------------------
if cfg.simulation:
    # Load the ground truth data.
    # The first column is the frame index, and we only keep frames ≥ 400.
    # The second and third columns are the x and z coordinates of the GT positions.
    # We convert these to tuples for easier processing later.
    gt_data = np.loadtxt(cfg.gt_data_path, delimiter=',')  # x,z positions per frame
    mask  = gt_data[:, 0] > 399            # first column is frame index
    loc2  = gt_data[mask, ::2]             # take x,z pairs (skip z duplicates if any)
    tuples_list = [tuple(row) for row in loc2.tolist()]  # convenient tuple format

# -----------------------------------------------------------------------------
# STEP 4 ─ Threshold each model at its own optimal score   (optional)This is only available for simulation since we need ground truth data
# -----------------------------------------------------------------------------

if cfg.simulation and cfg.threshold_each_model:
    # Thresholding is optional, but if enabled, we find the optimal threshold
    # for each model by maximising the AUC of the ROC curve.
    # This is done by calculate_maxAUC() which returns the optimal threshold.
    # The thresholds are used to filter out low-confidence detections.

    threshold_range = np.linspace(0, 1, 100)

    thresholded_boxes_list  = []  # shape: [n_models][n_frames][n_boxes]
    thresholded_scores_list = []
    thresholded_labels_list = []

    for jsonfile in range(5):
        # Pull the per-model column out of the stacked lists
        boxes  = [entry[jsonfile] for entry in boxes_list]
        scores = [entry[jsonfile] for entry in scores_list]
        labels = [entry[jsonfile] for entry in labels_list]

        # Choose the score threshold that maximises AUC for this model
        optimal_threshold = calculate_maxAUC(
            tuples_list, boxes, img_width, img_height, dx, dz, mapX, mapZ, tol, scores
        )

        # Keep only detections whose score ≥ optimal_threshold
        boxes, scores, labels = threshold_one_model(boxes, scores, labels,
                                                    optimal_threshold)

        # Accumulate results
        thresholded_boxes_list.append(boxes)
        thresholded_scores_list.append(scores)
        thresholded_labels_list.append(labels)
else:
    # If thresholding is not enabled, just copy the original lists
    thresholded_boxes_list  = boxes_list
    thresholded_scores_list = scores_list
    thresholded_labels_list = labels_list

# -----------------------------------------------------------------------------
# STEP 5 ─ Post-process / fuse the five models for every frame
# -----------------------------------------------------------------------------
num_frames = cfg.num_frames                 # total frames in the clip
weights    = cfg.weights                 # model-specific confidence weights

# Dynamic IoU threshold derived from spatial density of GT positions
# Here calculate_KDE() converts density → scalar IoU threshold per frame. This iou_thresh is used for adaptive algorithms.
if cfg.adaptive_iou_thresh:
    iou_thr = calculate_KDE(thresholded_boxes_list[1])
else:
    iou_thr = cfg.iou_thresh                  # fixed IoU threshold for fusion methods

# Hyper-parameters for Soft-NMS & WBF
skip_box_thr = cfg.skip_box_thr  # drop boxes with score < this inside Soft-NMS / WBF
sigma        = cfg.sigma  # Gaussian parameter for Soft-NMS

# Each entry: (reference, user-friendly name, wrapper that matches our call-sig)
methods = [
    (nms, f"{'Adaptive ' if cfg.adaptive_iou_thresh else ''}Non-Maximum Suppression{' After Threshold' if cfg.threshold_each_model  else ''}",
     (lambda b, s, l: nms_dynamic_thresh(b, s, l, sigma=sigma, weights=weights, iou_thr=iou_thr)) if cfg.adaptive_iou_thresh  else
     (lambda boxes, scores, labels: nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr))),

    (soft_nms, f"{'Adaptive ' if cfg.adaptive_iou_thresh  else ''}Soft-NMS{' After Threshold' if cfg.threshold_each_model  else ''}",
     (lambda b, s, l: soft_nms_dynamic(b, s, l, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)) if cfg.adaptive_iou_thresh  else
     (lambda boxes, scores, labels: soft_nms(boxes, scores, labels, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr))),

    (non_maximum_weighted, f"{'Adaptive ' if cfg.adaptive_iou_thresh  else ''}Non-Maximum Weighted{' After Threshold' if cfg.threshold_each_model  else ''}",
     (lambda b, s, l: non_maximum_weighted_dyn(b, s, l, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)) if cfg.adaptive_iou_thresh  else
     (lambda boxes, scores, labels: non_maximum_weighted(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr))),

    (weighted_boxes_fusion, f"{'Adaptive ' if cfg.adaptive_iou_thresh  else ''}Weighted Boxes Fusion{' After Threshold' if cfg.threshold_each_model  else ''}",
     (lambda b, s, l: weighted_boxes_fusion_dyn(b, s, l, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)) if cfg.adaptive_iou_thresh  else
     (lambda boxes, scores, labels: weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)))
]

# Storage for fused boxes from each method
all_method_boxes  = [[] for _ in methods]
all_method_scores = [[] for _ in methods]
all_method_labels = [[] for _ in methods]

# Loop over frames, run every fusion method, collect results
for frame_idx in tqdm.tqdm(range(num_frames), desc="Processing frames"):
    # Apply each fusion technique to the five-model stack for this frame
    for m_idx, (_, _, fuse_fn) in enumerate(methods):
        boxes, scores, labels = fuse_fn(thresholded_boxes_list[frame_idx],
                                        thresholded_scores_list[frame_idx],
                                        thresholded_labels_list[frame_idx])
        all_method_boxes [m_idx].append(boxes)
        all_method_scores[m_idx].append(scores)
        all_method_labels[m_idx].append(labels)

# -----------------------------------------------------------------------------
# STEP 6 ─ Save fused detections to disk (one file per method)
# -----------------------------------------------------------------------------
for m_idx, (_, method_name, _) in enumerate(methods):
    save_boxes_to_file_for_method(
        cfg,
        method_name,
        all_method_boxes [m_idx],
        all_method_scores[m_idx],
        all_method_labels[m_idx]
    )
