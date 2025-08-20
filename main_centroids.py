# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:44:05 2025

@author: Francesco
"""

import utils
import centroids_cv
import os

def main():

    dataset_folder = "E:/Universita/Dottorato/Progetto Polline Tor Vergata/Progetto Immagini/Level 0 Sylva/"
    centroids_path = dataset_folder + "centroids"

    if not os.path.exists(centroids_path):
        os.makedirs(centroids_path)
        
    city_folder_list = [f for f in os.listdir(dataset_folder) if not f.endswith(".txt") and "centroids" not in f]
    for city in city_folder_list[:1]:
        city_folder = os.path.join(dataset_folder,city)
        date_folder_list = os.listdir(city_folder)
        for date in date_folder_list[:1]:
            date_folder = os.path.join(city_folder,date)
            image_folder_list = [f for f in os.listdir(date_folder) if not f.endswith(".zip") and not f.endswith(".txt") ]
            for img_folder in image_folder_list[:1]:
                img_folder_path = os.path.join(date_folder,img_folder,"images")
                json_path = os.path.join(date_folder,img_folder)
                json_dictionary = {}
                img_list = [f for f in os.listdir(img_folder_path) if "SEG" in f]
                counting_img = 0
                counting_centroids = 0
                for img in img_list:
                
                    img_path = os.path.join(img_folder_path,img)
                    image_rgb, color_mask_uint8 = centroids_cv.get_segmentation_mask(img_path)
                    unknown, num_labels, markers = centroids_cv.get_markers(color_mask_uint8, city)
                    if markers.size == 0:
                        continue
                    markers_c = centroids_cv.color_analysis(image_rgb, num_labels, markers)
                    markers_c = markers_c + 1
                    markers_c[unknown == 255] = 0
                    centroids = centroids_cv.watershed(color_mask_uint8, markers_c)
                   #utils.save_figure_with_centroids(image_rgb,centroids,centroids_path,img)
                    counting_img += 1
                    counting_centroids += len(centroids)
                    json_dictionary[img] = [[c["x"],c["y"]] for c in centroids] 
            utils.save_json(json_path,counting_img,counting_centroids, json_dictionary)
            
if __name__ == "__main__":
    main()        