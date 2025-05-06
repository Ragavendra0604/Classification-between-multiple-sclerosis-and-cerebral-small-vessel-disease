import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from tqdm import tqdm
import gc

def extract_features(slices, hog_orientations=8, hog_pix_cell=(16, 16), hog_block=(1, 1)):
    features = []

    for idx, slice_img in enumerate(tqdm(slices, desc="Extracting Features")):
        try:
            # Normalize to 0–255 and convert to uint8
            slice_img = np.clip(slice_img * 255, 0, 255).astype(np.uint8)

            # --- HOG Features ---
            hog_feat = hog(
                slice_img,
                orientations=hog_orientations,
                pixels_per_cell=hog_pix_cell,
                cells_per_block=hog_block,
                block_norm='L2-Hys',
                visualize=False,
                feature_vector=True
            )

            # GLCM Features 
            glcm = graycomatrix(slice_img, distances=[1], angles=[0],
                                levels=256, symmetric=True, normed=True)

            glcm_features = [
                graycoprops(glcm, prop)[0, 0]
                for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            ]

            # HOG + GLCM
            combined = np.hstack((hog_feat, glcm_features)).astype(np.float32)
            features.append(combined)

            if idx % 100 == 0:
                gc.collect()

        except Exception as e:
            print(f"Slice {idx} failed: {e}")
            continue

    return np.array(features, dtype=np.float32)
