# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 17:36:20 2025

@author: Francesco
"""

import numpy as np
from datetime import datetime
from pycocotools import mask as maskUtils
import cv2
from PIL import Image



def from_tiff_to_jpeg(file_path,output_path):
  """Convert .tiff image to .jpg"""
  try: 
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.save(output_path, format="JPEG", quality=100)
            return True
  except Exception as e:
      print(f"Error occured while converting {file_path} {e}")
      return False


def create_coco_annotation(detections, image_id, image_shape, category_id, annotation_id_start=1, method = "RLE"):    
    """
    Create COCO format annotations from detections of SAM/Grounding Dino
    
    Args:
        detections: SuperVision detections object with masks, xyxy class id and confidence
        image_id: ID of the image
        image_shape: (height, width) of the image
        classes: List of class names
        annotation_id_start: Starting ID for annotations
        method: method for segmentation, RLE as default, alternative POLYGON
    
    Returns:
        List of COCO annotation dictionaries
    """
    annotations = []
    for i, (xyxy, mask, confidence, class_id, _) in enumerate(detections):

        area = float(np.sum(mask)) #Sum of True
        
        if method=="RLE":
            segmentation = mask_to_rle(mask)
        elif method=="POLYGON":
            segmentation = mask_to_polygon(mask)
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'RLE' or 'POLYGON'.")
            
        xmin, ymin, xmax, ymax = xyxy
        bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]  # [x, y, width, height]  as COCO format requires                
        annotation = {
            "id": annotation_id_start + i,
            "image_id": image_id,
            "category_id": category_id,  # COCO categories start from 1
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation,
            "iscrowd": 0,
            "score": float(confidence)  # Optional: confidence score
        }        
        annotations.append(annotation)       
    return annotations    
            

def mask_to_rle(mask):
       
    """Convert binary mask to RLE format
    Args: Mask --> Ensure mask is in the right format (H, W) and binary
    
    Returns: Run-Lenght Encoding Mask
    """ 
    
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8))) #np.asfortranarray converts an array to Fortran-contiguous order (column-major order),
                                                                     #as opposed to C-contiguous order (row-major order, it moves faster across rows)
                                                                     #it's required by pycocotools.mask function that return a dictionary :  {"size": [height, width],  
                                                                     #"counts": b"encoded_run_length_data"}
        
    rle['counts'] = rle['counts'].decode('utf-8')    # Convert bytes to string for JSON serialization
    return rle
                
def mask_to_polygon(mask):

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Need at least 3 points for a polygon
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:  # Need at least 3 coordinate pairs
                polygons.append(polygon)
    return polygons
                
                
def create_coco_dataset(images_info, all_annotations, classes, dataset_description, method, version = "1.0"):

    categories = []
    for i, class_name in enumerate(classes):
        categories.append({
            "id": i + 1,  # COCO categories start from 1
            "name": class_name,
            "supercategory": "object"
        })    
    coco_dataset = {
        "info": {
            "description": dataset_description,
            "version": version,
            "year": datetime.now().year,
            "contributor": method,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": images_info,
        "annotations": all_annotations,
        "categories": categories
    }    
    return coco_dataset             
                
def topleft_rightbottom_to_yolo(bbox, image_width, image_height):
    """
    Convert tpoleft-rightbottom bounding box coordinates to YOLO format   
    Args:
        bbox: [xmin, ymin, xmax, ymax] in absolute pixels
        image_width
        image_height   
    Returns:
        [x_center, y_center, width, height] in normalized coordinates (0-1)
    """
    xmin, ymin, xmax, ymax = bbox    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def coco_to_yolo(coco_bbox, image_width, image_height):
    """
    Convert COCO format [x, y, width, height] to YOLO format
    
    Args:
        coco_bbox: [x, y, width, height] where x,y is top-left corner
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        [x_center, y_center, width, height] in normalized coordinates
    """
    xmin, ymin, width, height = coco_bbox
    xmax = xmin + width
    ymax = ymin + height
 
    return topleft_rightbottom_to_yolo([xmin, ymin, xmax, ymax], image_width, image_height)


def yolo_to_topleft_rightbottom(yolo_bbox, image_width, image_height):
    """
    Convert YOLO format to absolute bounding box coordinates
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] in normalized coordinates
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        [xmin, ymin, xmax, ymax] in absolute pixels
    """
    x_center_norm, y_center_norm, width_norm, height_norm = yolo_bbox
    

    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    width = width_norm * image_width
    height = height_norm * image_height
    
    xmin = x_center - (width / 2)
    ymin = y_center - (height / 2)
    xmax = x_center + (width / 2)
    ymax = y_center + (height / 2)
    
    return [xmin, ymin, xmax, ymax]


def yolo_to_coco(yolo_bbox, image_width, image_height):
    """
    Convert YOLO format to COCO format
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] in normalized coordinates
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        [x, y, width, height] where x,y is top-left corner in absolute pixels
    """

    xmin, ymin, xmax, ymax = yolo_to_topleft_rightbottom(yolo_bbox, image_width, image_height)
    width = xmax - xmin
    height = ymax - ymin
    
    return [xmin, ymin, width, height]



def mask_to_yolo_segmentation(mask, image_width, image_height):
    """
    Convert binary mask to YOLO segmentation format (normalized polygon coordinates)
    
    Args:
        mask: Binary mask array (H, W)
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of normalized coordinates [x1, y1, x2, y2, ...]
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Use the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 3:
        return []
    
    # Flatten and normalize coordinates
    polygon = largest_contour.flatten()
    normalized_polygon = []
    
    for i in range(0, len(polygon), 2):
        x_norm = polygon[i] / image_width
        y_norm = polygon[i + 1] / image_height
        normalized_polygon.extend([x_norm, y_norm])
    
    return normalized_polygon      

def coco_polygon_to_yolo(coco_polygon, image_width, image_height):
    """
    Convert COCO polygon format to YOLO segmentation format
    
    Args:
        coco_polygon: List of polygon coordinates [x1, y1, x2, y2, ...]
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        List of normalized coordinates for YOLO format
    """
    # Convert to numpy array and reshape
    polygon = np.array(coco_polygon).reshape(-1, 2)
    
    # Normalize coordinates
    normalized_polygon = []
    for x, y in polygon:
        norm_x = x / image_width
        norm_y = y / image_height
        normalized_polygon.extend([norm_x, norm_y])
    
    return normalized_polygon

def yolo_to_coco_polygon(yolo_coords, image_width, image_height):
    """
    Convert YOLO segmentation format to COCO polygon format
    
    Args:
        yolo_coords: List of normalized coordinates [x1, y1, x2, y2, ...]
        image_width: Image width in pixels  
        image_height: Image height in pixels
    
    Returns:
        List of absolute pixel coordinates for COCO format
    """
    coco_polygon = []
    
    # Process coordinates in pairs
    for i in range(0, len(yolo_coords), 2):
        norm_x = yolo_coords[i]
        norm_y = yolo_coords[i + 1]
        
        # Convert to absolute coordinates
        abs_x = norm_x * image_width
        abs_y = norm_y * image_height
        
        coco_polygon.extend([abs_x, abs_y])
    
    return coco_polygon

def coco_rle_to_yolo(rle_annotation, image_width, image_height):
    """
    Convert COCO RLE format to YOLO polygon format
    
    Args:
        rle_annotation: COCO RLE segmentation
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        List of normalized polygon coordinates
    """
    # Decode RLE to binary mask
    if isinstance(rle_annotation, list):
        # Polygon format - convert first polygon
        return coco_polygon_to_yolo(rle_annotation[0], image_width, image_height)
    else:
        # RLE format
        mask = maskUtils.decode(rle_annotation)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return []
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert contour to polygon
        polygon = largest_contour.flatten()
        
        # Normalize coordinates
        normalized_coords = []
        for i in range(0, len(polygon), 2):
            norm_x = polygon[i] / image_width
            norm_y = polygon[i + 1] / image_height
            normalized_coords.extend([norm_x, norm_y])
        
        return normalized_coords

                
                
def create_yolo_annotation(detections, image_width, image_height, category_id):
    """
    Create YOLO format annotations from detections
    
    Args:
        detections: SuperVision detections object with masks, xyxy class id and confidence
        image_width: Width of the image
        image_height: Height of the image
        category_id: Category ID for this detection (will be converted to 0-indexed)
    
    Returns:
        List of YOLO annotation strings
    """
    yolo_annotations = []
    
    for i, (xyxy, mask, confidence, class_id, _) in enumerate(detections):
        # Convert category_id to 0-indexed for YOLO
        yolo_class_id = category_id - 1
        
        # Get normalized polygon coordinates
        segmentation_coords = mask_to_yolo_segmentation(mask, image_width, image_height)
        
        if len(segmentation_coords) >= 6:  # Need at least 3 points (6 coordinates)
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            coords_str = ' '.join([f"{coord:.6f}" for coord in segmentation_coords])
            yolo_line = f"{yolo_class_id} {coords_str}"
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations                
                
                
                
                