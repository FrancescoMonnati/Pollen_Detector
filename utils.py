# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:43:09 2025

@author: Francesco
"""

from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import cv2


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
  
    
    
def save_figure_with_centroids(img_rgb, centroids, save_path, base_name):
    
    img_with_centroids = draw_centroids(img_rgb, centroids)
    
    plt.figure(figsize=(25,16))
    plt.imshow(img_with_centroids)
    plt.axis('off')
    
    fig_filename = f"{base_name}_with_centroids.png"
    fig_path = os.path.join(save_path, fig_filename)
    
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close() 
    
def save_json(destination_path, len_images, total_centroids, img_dictionary):
    
     json_data = {
        'total_images': len_images,
        'total_centroids': total_centroids,
        'images': img_dictionary
    }
     
     json_filename = f"centroids_coordinates.json"
     json_path = os.path.join(destination_path, json_filename)
    
     with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
        

def draw_centroids(img, centroids):
    img_with_centroids = img.copy()
    for centroid in centroids:
        x, y = int(centroid['x']), int(centroid['y'])

        cv2.circle(img_with_centroids, (x, y), 4, (255, 255, 255), -1)  
        cv2.circle(img_with_centroids, (x, y), 7, (0, 0, 0), 2)  
        cv2.line(img_with_centroids, (x-10, y), (x+10, y), (255, 255, 255), 2)
        cv2.line(img_with_centroids, (x, y-10), (x, y+10), (255, 255, 255), 2)
    return img_with_centroids    