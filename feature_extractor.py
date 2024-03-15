# Author: Shajib Ghosh
# Date: 2024-03-14
# Time: 03:34 UTC

import os
import cv2
import numpy as np
from skimage.feature import graycomatrix
from skimage import color as skimage_color
import pandas as pd
from tqdm import tqdm

def yolo_to_polygon(yolo_box, image_size):
    num_points = len(yolo_box)
    polygon_coords = [(int(yolo_box[i] * image_size[0]), int(yolo_box[i+1] * image_size[1])) for i in range(0, num_points, 2)]
    return polygon_coords

def extract_glcm_features(image, distances, angles):
    if len(image.shape) == 3:
        gray_image = skimage_color.rgb2gray(image)
    elif len(image.shape) == 4:  # Check if the image has an alpha channel
        image = image[:, :, :3]  # Drop the alpha channel
        gray_image = skimage_color.rgb2gray(image)
    else:
        gray_image = image

    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    contrast = np.mean([contrast_glcm(glcm, d, a, distances, angles) for d in distances for a in angles])
    correlation = np.mean([correlation_glcm(glcm, d, a, distances, angles) for d in distances for a in angles])
    energy = np.mean([energy_glcm(glcm, d, a, distances, angles) for d in distances for a in angles])
    homogeneity = np.mean([homogeneity_glcm(glcm, d, a, distances, angles) for d in distances for a in angles])

    return {"GLCM_Contrast": contrast, "GLCM_Correlation": correlation, "GLCM_Energy": energy, "GLCM_Homogeneity": homogeneity}

def contrast_glcm(glcm, distance, angle, distances, angles):
    return np.sum((distance * glcm[:, :, distances.index(distance), angles.index(angle)]) ** 2)

def correlation_glcm(glcm, distance, angle, distances, angles):
    mu_x, mu_y = np.mean(glcm[:, :, distances.index(distance), angles.index(angle)]), np.mean(glcm[:, :, distances.index(distance), angles.index(angle)].T)
    sigma_x, sigma_y = np.std(glcm[:, :, distances.index(distance), angles.index(angle)]), np.std(glcm[:, :, distances.index(distance), angles.index(angle)].T)
    return np.mean((glcm[:, :, distances.index(distance), angles.index(angle)] - mu_x) * (glcm[:, :, distances.index(distance), angles.index(angle)].T - mu_y) / (sigma_x * sigma_y))

def energy_glcm(glcm, distance, angle, distances, angles):
    return np.sum(glcm[:, :, distances.index(distance), angles.index(angle)] ** 2)

def homogeneity_glcm(glcm, distance, angle, distances, angles):
    return np.sum(glcm[:, :, distances.index(distance), angles.index(angle)] / (1 + distance))

def extract_texture_features_within_polygon(image, polygon_coords, distances, angles):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [np.array(polygon_coords)], 255)
    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    glcm_features = extract_glcm_features(masked_image, distances, angles)

    eroded_image = cv2.erode(masked_image, None, iterations=3)
    dilated_image = cv2.dilate(masked_image, None, iterations=3)
    tamura_contrast = np.mean(np.abs(masked_image.astype(np.int32) - eroded_image.astype(np.int32)))
    tamura_coarseness = np.mean(np.abs(eroded_image.astype(np.int32) - dilated_image.astype(np.int32)))
    tamura_directionality = np.mean(np.abs(dilated_image.astype(np.int32) - masked_image.astype(np.int32)))

    entropy = -np.sum((masked_image / 255) * np.log2((masked_image / 255) + 1e-10))

    texture_features = {**glcm_features, "Tamura_Contrast": tamura_contrast, "Tamura_Coarseness": tamura_coarseness, "Tamura_Directionality": tamura_directionality, "Entropy": entropy}

    return texture_features

def extract_shape_features_within_polygon(image, polygon_coords):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [np.array(polygon_coords)], 255)
    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)

    # Find contours in the masked image
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize shape features
    area = 0
    perimeter = 0
    circularity = 0
    solidity = 0
    convexity = 0
    eccentricity = 0
    aspect_ratio = 0

    # Iterate through contours
    for contour in contours:
        # Calculate area and perimeter
        if len(contour) >= 5:
            area += cv2.contourArea(contour)
            perimeter += cv2.arcLength(contour, True)

            # Convex hull
            convex_hull = cv2.convexHull(contour)

            # Solidity (area of contour / area of convex hull)
            hull_area = cv2.contourArea(convex_hull)
            if hull_area != 0:
                solidity += cv2.contourArea(contour) / hull_area

            # Convexity (perimeter of contour / perimeter of convex hull)
            hull_perimeter = cv2.arcLength(convex_hull, True)
            if hull_perimeter != 0:
                convexity += cv2.arcLength(contour, True) / hull_perimeter

            # Eccentricity
            _, (w, h), _ = cv2.fitEllipse(contour)
            eccentricity += (w / h) ** 2

            # Aspect ratio
            aspect_ratio += w / h

    # Calculate circularity
    if perimeter != 0 and area != 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Average shape features over all contours
    num_contours = len(contours)
    if num_contours != 0:
        area /= num_contours
        perimeter /= num_contours
        circularity /= num_contours
        solidity /= num_contours
        convexity /= num_contours
        eccentricity /= num_contours
        aspect_ratio /= num_contours

    return {"Area": area, "Perimeter": perimeter, "Circularity": circularity, "Solidity": solidity, "Convexity": convexity, "Eccentricity": eccentricity, "Aspect_Ratio": aspect_ratio}


def extract_color_features_within_polygon(image, polygon_coords, bins=8):
    # Create a mask for the polygon
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, np.array([polygon_coords], dtype=np.int32), 255)

    # Function to calculate color histograms for each channel in a color space
    def calc_color_hist(image, color_space, mask, bins):
        color_features = {}
        for i, col in enumerate(['Channel_1', 'Channel_2', 'Channel_3']):
            hist = cv2.calcHist([image], [i], mask, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            for j in range(bins):
                color_features[f"{color_space}_{col}_hist_bin_{j}"] = hist[j]
        return color_features

    # Convert image to different color spaces and calculate histograms
    color_spaces = {
        'RGB': cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        'HSV': cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        'HLS': cv2.cvtColor(image, cv2.COLOR_BGR2HLS),
        'LAB': cv2.cvtColor(image, cv2.COLOR_BGR2Lab),
        'LUV': cv2.cvtColor(image, cv2.COLOR_BGR2Luv),
        'YCrCb': cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb),
        'XYZ': cv2.cvtColor(image, cv2.COLOR_BGR2XYZ),
    }
    color_features = {}
    for space, img in color_spaces.items():
        color_features.update(calc_color_hist(img, space, mask, bins))

    return color_features

def main():
    parent_folder = os.getcwd()
    images_folder = os.path.join(parent_folder, 'images_fpic')  #images_fpic, images
    annotations_folder = os.path.join(parent_folder, 'smd_fpic_anns_yolo') #smd_anns_fpic, annotations
    output_folder = os.path.join(parent_folder, 'extracted_feats_fpic') #extracted_feats_fpic, extracted_features

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Specify GLCM distances and angles
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Iterate through each image and its associated annotation
    for filename in tqdm(os.listdir(images_folder), desc="Processing Images"):
        if filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            annotation_path = os.path.join(annotations_folder, filename.replace(".png", ".txt"))

            # Read image and associated YOLO annotations
            image = cv2.imread(image_path)
            with open(annotation_path, 'r') as file:
                lines = file.readlines()

            # Initialize a list to store the extracted features for this image
            features_list = []

            # Iterate through each YOLO annotation in the file
            for i, line in enumerate(lines):
                yolo_box = [float(coord) for coord in line.strip().split()]
                class_id = int(yolo_box[0])  # Assuming class ID is the first element
                polygon_coords = yolo_to_polygon(yolo_box[1:], image.shape[:2])  # Exclude the class ID

                # Extract texture features
                texture_features = extract_texture_features_within_polygon(image, polygon_coords, distances, angles)

                # Extract shape features
                shape_features = extract_shape_features_within_polygon(image, polygon_coords)

                # Extract color features
                color_features = extract_color_features_within_polygon(image, polygon_coords)

                # Combine texture and shape features
                combined_features = {"Component_ID": i + 1, "Class_ID": class_id, **color_features, **texture_features, **shape_features}
                features_list.append(combined_features)

            # Create a DataFrame for this image and save it to CSV
            output_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_features.csv")
            df = pd.DataFrame(features_list)
            # Save the DataFrame to CSV
            df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    main()
