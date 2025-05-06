import numpy as np
import nibabel as nib
from skimage import morphology
from scipy.ndimage import binary_fill_holes
import cv2
import gc

def resize_and_skull_strip(preprocessed_slices, output_size=(256, 256), verbose=True):
    skull_stripped_slices = []

    for idx, slice_2d in enumerate(preprocessed_slices):
        try:
            slice_2d = slice_2d.astype(np.float32, copy=False)

            # Resize
            resized = cv2.resize(slice_2d, output_size, interpolation=cv2.INTER_LINEAR)

            # Normalize to 0–255
            img_uint8 = (resized * 255).astype(np.uint8)

            # Otsu's Thresholding for mask
            _, binary_mask = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bin_bool = binary_mask.astype(bool)

            # Morphological cleaning
            bin_bool = morphology.remove_small_objects(bin_bool, min_size=64)
            bin_bool = morphology.binary_closing(bin_bool, morphology.disk(3))
            bin_bool = binary_fill_holes(bin_bool)

            # Apply mask
            final_mask = bin_bool.astype(np.float32)
            skull_stripped = resized * final_mask

            skull_stripped_slices.append(skull_stripped)

            if idx % 100 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error processing slice #{idx}: {e}")

    if verbose:
        print(f"Skull-stripped {len(skull_stripped_slices)} slices.")

    return skull_stripped_slices