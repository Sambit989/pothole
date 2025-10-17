# Camera Calibration & Focal Length Estimation

To compute perceived focal length F:
1. Place a ruler or object of known width W (cm) on the ground at a known distance D (cm) from the camera (e.g., 30cm, 60cm, 90cm).
2. Capture an image and measure the object's pixel width P (use `cv2.Canny` + contours or manual inspection).
3. Compute F = (P * D) / W for multiple distances and average the F values.
4. Save the averaged focal length and use it with `dimension_estimation.py` to convert bbox pixels to real-world sizes.

Example script to measure P is provided in `calibrate/measure_pixels.py`.
