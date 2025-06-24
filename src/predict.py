import tensorflow as tf
import numpy as np
import cv2
import os
import time
import psutil

class PatchPredictor:
    """Patch‑wise predictor for large images.

    This class performs segmentation prediction on a given image (NumPy array)
    using a pre-trained Keras model. It does not handle file I/O.
    """

    def __init__(self, model_path, patch_size: int = 256, overlap: float = 0.5):
        self.PATCH_SIZE = patch_size
        self.OVERLAP = overlap
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "focal_dice_loss": self.focal_dice_loss,
                "dice_coef": self.dice_coef,
            },
        )
        self.process = psutil.Process(os.getpid())

    # ---------------------------------------------------------------------
    #  Metrics & loss (These remain unchanged as they are model-specific)
    # ---------------------------------------------------------------------
    def dice_coef(self, y_true, y_pred, smooth: int = 1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (
            (2.0 * intersection + smooth) /
            (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
        )

    def focal_dice_loss(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = 1 - self.dice_coef(y_true, y_pred)
        return bce + dice



    # ---------------------------------------------------------------------
    #  Public API (Simplified to only accept and return image arrays)
    # ---------------------------------------------------------------------
    def predict_mask(self, image_array: np.ndarray) -> np.ndarray:
        """Predicts a binary mask for a single grayscale image (NumPy array).

        Args:
            image_array (np.ndarray): The input grayscale image as a NumPy array.
                                      Expected shape (H, W). Values should be 0-255.

        Returns:
            np.ndarray: The predicted binary mask as a NumPy array (0 or 255).
                        Returns None if the input image_array is invalid.
        """
        start_time = time.time()
        start_mem = self.process.memory_info().rss

        if image_array is None or image_array.ndim != 2:
            print(" Error: Input image_array must be a 2D grayscale NumPy array.")
            return None

        stride = int(self.PATCH_SIZE * (1 - self.OVERLAP))
        img_h, img_w = image_array.shape
        full_image_normalized = image_array.astype("float32") / 255.0

        pad_h = (stride - ((img_h - self.PATCH_SIZE) % stride)) % stride
        pad_w = (stride - ((img_w - self.PATCH_SIZE) % stride)) % stride

        padded_image = np.pad(full_image_normalized, ((0, pad_h), (0, pad_w)), mode="constant")
        padded_prediction = np.zeros(padded_image.shape, dtype=np.float32)
        padded_overlap_count = np.zeros(padded_image.shape, dtype=np.float32)

        for y in range(0, padded_image.shape[0] - self.PATCH_SIZE + 1, stride):
            for x in range(0, padded_image.shape[1] - self.PATCH_SIZE + 1, stride):
                patch = padded_image[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE]
                # Ensure the patch has the correct dimensions (batch, height, width, channels)
                patch_expanded = np.expand_dims(np.expand_dims(patch, axis=-1), axis=0)
                pred_patch = self.model.predict(patch_expanded, verbose=0)[0, :, :, 0]
                padded_prediction[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE] += pred_patch
                padded_overlap_count[y : y + self.PATCH_SIZE, x : x + self.PATCH_SIZE] += 1

        # Avoid division by zero for any regions that might not have been covered (though unlikely with overlap)
        padded_overlap_count[padded_overlap_count == 0] = 1e-8
        final_padded_prediction = padded_prediction / padded_overlap_count
        final_prediction = final_padded_prediction[0:img_h, 0:img_w]
        final_mask = (final_prediction > 0.1).astype(np.uint8) * 255

        return final_mask

    # Removed save_comparison as file saving will be handled externally
    # Removed predict_from_path as file I/O and path management are external

    def predict_all(self, items, output_dir: str):
        """
        Predicts masks for a list of (name, image) pairs and saves them to output_dir.

        Args:
            items: List of tuples (name: str, image: np.ndarray)
            output_dir: Destination folder for mask PNGs
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for name, img in items:
            if img is None or img.ndim != 2:
                print(f" Skipping {name}: not a valid 2D grayscale image.")
                continue

            mask = self.predict_mask(img)
            if mask is not None:
                out_path = os.path.join(output_dir, f"{name}_mask.png")
                cv2.imwrite(out_path, mask)
                print("  saved", os.path.basename(out_path))
                results.append((name, mask))

        print(f"\n All masks are inside {output_dir}")
        return results
