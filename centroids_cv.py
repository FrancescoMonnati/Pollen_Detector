# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:04:54 2025

@author: Francesco
"""
import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.signal import find_peaks
import os


def calculate_centroids_from_watershed(markers, min_area=20):

    centroids = []
    unique_labels = np.unique(markers)
    object_labels = unique_labels[(unique_labels > 1)]  
    
    for l in object_labels:
        object_mask = (markers == l).astype(np.uint8)
        area = np.sum(object_mask)
        if area >= min_area:
            cx,cy = get_centroids(object_mask)             
            centroids.append({
                    'label': int(l),
                    'x': float(cx),
                    'y': float(cy),
                    'area': int(area),
                    'method': 'moments'
                })    
    return centroids

def get_centroids(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:  
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return cx,cy
    else:    
        raise ValueError(f"Division by zero")
        
def get_segmentation_mask(img_path, difference_parameter=10):
    
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file does not exist: {img_path}")
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to read image: {img_path}")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB/BGR image with 3 channels, got shape {image.shape}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    color_mask = np.any(np.abs(image_rgb.astype(int) - gray_3channel.astype(int)) > difference_parameter, axis=2)
    color_mask_uint8 = (color_mask.astype(np.uint8) * 255)
    
    return image_rgb, color_mask_uint8

def bg_and_eroding(mask, kernel_param = 3, erosion_size = 25):
    
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, got {type(mask)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array (binary image), got shape {mask.shape}")
    
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 255}):
        raise ValueError(f"mask must be binary (0 and 255), found values: {unique_vals}")
    
    kernel = np.ones((kernel_param, kernel_param), np.uint8)
    bg = cv2.dilate(mask, kernel, iterations=3)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    eroded = cv2.erode(mask, kernel_erode, iterations=1)
    
    return bg, eroded

def get_missing_objects(mask,labels):
    
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, got {type(mask)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array (binary image), got shape {mask.shape}")
    
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 255}):
        raise ValueError(f"mask must be binary (0 and 255), found values: {unique_vals}")
    
    missing_objects = []
    for region in regionprops(labels):
        coords = region.coords
        overlap = mask[coords[:,0], coords[:,1]].sum()
        if overlap == 0:  
            missing_objects.append(region.label)
    return missing_objects


def get_distance_mask(mask, dist_param = 0.38):
    
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, got {type(mask)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array (binary image), got shape {mask.shape}")
    
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 255}):
        raise ValueError(f"mask must be binary (0 and 255), found values: {unique_vals}")

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist_transform, dist_param * dist_transform.max(), 255, 0)
    return fg.astype(np.uint8)
    
def get_labels(mask):
    
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, got {type(mask)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2D array (binary image), got shape {mask.shape}")
    
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 255}):
        raise ValueError(f"mask must be binary (0 and 255), found values: {unique_vals}")
    
    return label(mask)

def mask_merging(missing_objects, labels):
    return  np.isin(labels, missing_objects)


def get_peaks(object_hsv):
    hist = cv2.calcHist([object_hsv], [0], None, [256], [0, 256])
    hist_flat = hist.ravel()
    peaks, _ = find_peaks(hist_flat, height=0)  
    num_modes = len(peaks)
    return peaks, num_modes


def get_markers(mask, city):
    
    if np.sum(mask) == 0:
        return [], False, np.array([])
    if "FIPALL" in city:
        bg, eroded_mask = bg_and_eroding(mask, dist_param = 0.28)
        fg = get_distance_mask(mask)
        mask_labels = get_labels(mask)
        mask_missing_object = get_missing_objects(fg,mask_labels)
        missing_mask = mask_merging(mask_missing_object,mask_labels)
        final_mask = np.logical_or(fg, missing_mask).astype(np.uint8)
    else:
        bg, eroded_mask = bg_and_eroding(mask)
        fg = get_distance_mask(mask)
        mask_labels = get_labels(mask)    
        eroded_mask_labels = get_labels(eroded_mask)                
        eroded_missing_object = get_missing_objects(fg,eroded_mask_labels)
        eroded_missing_mask = mask_merging(eroded_missing_object,eroded_mask_labels)
                
        mask_missing_object = get_missing_objects(fg,mask_labels)
        missing_mask = mask_merging(mask_missing_object,mask_labels)
                
        final_mask = np.logical_or(eroded_missing_mask, fg).astype(np.uint8)
        final_mask = np.logical_or(final_mask, missing_mask).astype(np.uint8)
    num_labels, markers = cv2.connectedComponents(final_mask)
    unknown = cv2.subtract(bg, final_mask)
    
    return unknown, num_labels, markers

def color_analysis(img, num_labels, markers):
    
    for lab in range(num_labels):
        if lab == 0:
            continue
        object_mask = (markers == lab)
        object_colors = img*object_mask[...,None]
        object_hsv = cv2.cvtColor(object_colors, cv2.COLOR_RGB2HSV)
                    
        peaks, num_modes = get_peaks(object_hsv)
        valid_peaks = peaks.copy()
        for p in peaks:
            hsv_mask = (object_hsv[:,:,0] == p)
            object_hsv_masked = object_hsv * hsv_mask[...,None]
            num_labels_peak, labels_peak = cv2.connectedComponents(hsv_mask.astype(np.uint8))
            if num_labels_peak > 2 or object_hsv_masked.flatten().sum() < 50000:
                valid_peaks = peaks[peaks != p]
        peak_centroids =  []
        next_label = markers.max() + 1       
        for i, p in enumerate(valid_peaks):
            hsv_mask = (object_hsv[:,:,0] == p)
            cx,cy = get_centroids(hsv_mask.astype(np.uint8))
            peak_centroids.append({
                        'x': float(cx),
                        'y': float(cy),
                        })
    
            hsv_mask = (object_hsv[:,:,0] == p) & object_mask
            if np.any(hsv_mask):
                if i == 0:
                    markers[hsv_mask] = lab
                else:
                    markers[hsv_mask] = next_label + (i-1)
    return markers            

def watershed(mask,markers):
    
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(mask_color, markers)
    centroids = calculate_centroids_from_watershed(markers_ws, min_area=10)
    
    return centroids




