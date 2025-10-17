import numpy as np
import cv2

def compute_perceived_focal_length(pixel_length, real_length_cm, distance_cm):
    """F = (P * D) / W"""
    return (pixel_length * distance_cm) / real_length_cm

def pixels_to_cm(pixel_length, focal_length, distance_cm):
    """Given perceived focal length F, compute real-world size W = (P * D) / F"""
    return (pixel_length * distance_cm) / focal_length

def estimate_sizes_from_bboxes(image, bboxes, focal_length, camera_height_cm=90):
    """Estimate width and height in cm for each bbox.
    bboxes: list of [x,y,w,h] in pixels.
    Returns list of dicts with estimated width_cm, height_cm, area_cm2.
    """
    H,W = image.shape[:2]
    results = []
    for (x,y,w,h) in bboxes:
        width_cm = pixels_to_cm(w, focal_length, camera_height_cm)
        height_cm = pixels_to_cm(h, focal_length, camera_height_cm)
        area = width_cm * height_cm
        results.append({'bbox':[x,y,w,h],'width_cm':width_cm,'height_cm':height_cm,'area_cm2':area})
    return results

if __name__ == '__main__':
    print('This module provides functions to compute perceived focal length and convert bbox pixels to cm.')
