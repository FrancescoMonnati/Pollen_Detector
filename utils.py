# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:43:09 2025

@author: Francesco
"""

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