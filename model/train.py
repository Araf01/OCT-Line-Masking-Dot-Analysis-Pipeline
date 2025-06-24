import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION PARAMETERS ---

# Image and Patch Dimensions
IMG_HEIGHT = 1024
IMG_WIDTH = 1999
PATCH_SIZE = 256
OVERLAP = 0.5 # 50% overlap

# Paths to your data
# Organize your data as follows:
# /data/
#   /images/
#     image_1.png
#     image_2.png
#     ...
#   /masks/
#     image_1.png  (mask corresponding to image_1.png)
#     image_2.png
#     ...
IMAGE_DIR = '/kaggle/input/oct9-inverted-final/inverted_images'
MASK_DIR = '/kaggle/input/oct9-inverted-final/masks'

# Training Parameters
BATCH_SIZE = 4
EPOCHS = 50
VALIDATION_SPLIT = 0.20 # Use 15% of data for validation

# --- 2. U-NET MODEL ARCHITECTURE ---

def build_unet(input_shape):
    """
    Builds the U-Net model architecture.
    """
    inputs = Input(input_shape)

    # Encoder (Contracting Path)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder (Expansive Path)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# --- 3. CUSTOM LOSS FUNCTIONS & METRICS ---

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)
    
    # Calculate Focal Loss
    loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

def focal_dice_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


# --- 4. DATA GENERATOR & AUGMENTATION ---

def augment_patch(image_patch, mask_patch):
    """Applies data augmentation to a single image and mask patch."""
    # Random horizontal flip
    if random.random() > 0.5:
        image_patch = cv2.flip(image_patch, 1)
        mask_patch = cv2.flip(mask_patch, 1)

    # Random rotation
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((PATCH_SIZE // 2, PATCH_SIZE // 2), angle, 1)
        image_patch = cv2.warpAffine(image_patch, M, (PATCH_SIZE, PATCH_SIZE))
        mask_patch = cv2.warpAffine(mask_patch, M, (PATCH_SIZE, PATCH_SIZE))

    # Add synthetic noise (to make model robust to existing noise)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, image_patch.shape).astype('float32')
        image_patch = image_patch + noise
        image_patch = np.clip(image_patch, 0., 1.)
        
    return image_patch, mask_patch


def data_generator(image_ids, image_dir, mask_dir, batch_size=BATCH_SIZE, augment=True):
    stride = int(PATCH_SIZE * (1 - OVERLAP))
    while True:
        random.shuffle(image_ids)
        batch_images, batch_masks = [], []
        for image_id in image_ids:
            img_path = os.path.join(image_dir, image_id)
            mask_path = os.path.join(mask_dir, image_id)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise FileNotFoundError(f"Failed to load image: {img_path}")
            if mask is None:
                raise FileNotFoundError(f"Failed to load mask: {mask_path}")

            img = img.astype('float32') / 255.0
            mask = (mask > 0).astype('float32')


            # Extract patches
            for y in range(0, IMG_HEIGHT - PATCH_SIZE + 1, stride):
                for x in range(0, IMG_WIDTH - PATCH_SIZE + 1, stride):
                    img_patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    
                    # Augment if specified
                    if augment:
                        img_patch, mask_patch = augment_patch(img_patch, mask_patch)

                    batch_images.append(img_patch)
                    batch_masks.append(mask_patch)
                    
                    # Yield a batch when it's full
                    if len(batch_images) == batch_size:
                        yield (np.expand_dims(np.array(batch_images), axis=-1), 
                               np.expand_dims(np.array(batch_masks), axis=-1))
                        batch_images = []
                        batch_masks = []

# --- 5. MAIN TRAINING SCRIPT ---

if __name__ == '__main__':
    # Get all image IDs
    image_ids = next(os.walk(IMAGE_DIR))[2]
    
    # Split IDs into training and validation sets
    train_ids, val_ids = train_test_split(image_ids, test_size=VALIDATION_SPLIT, random_state=42)

    print(f"Total images: {len(image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Create generators
    train_gen = data_generator(train_ids, IMAGE_DIR, MASK_DIR, batch_size=BATCH_SIZE, augment=True)
    val_gen = data_generator(val_ids, IMAGE_DIR, MASK_DIR, batch_size=BATCH_SIZE, augment=False) # No augmentation for validation

    # Calculate steps per epoch
    # Note: This is an approximation. A more precise way would be to count all patches beforehand.
    stride = int(PATCH_SIZE * (1 - OVERLAP))
    patches_per_image = ( (IMG_HEIGHT - PATCH_SIZE) // stride + 1 ) * ( (IMG_WIDTH - PATCH_SIZE) // stride + 1 )
    
    steps_per_epoch = (len(train_ids) * patches_per_image) // BATCH_SIZE
    validation_steps = (len(val_ids) * patches_per_image) // BATCH_SIZE

    print(f"Approx. patches per image: {patches_per_image}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Build and compile the model
    model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 1))
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss=focal_dice_loss, 
                  metrics=[dice_coef])
    
    model.summary()

    # Define callbacks
    callbacks = [
        ModelCheckpoint('oct_segmentation_best.keras', verbose=1, save_best_only=True, monitor='val_dice_coef', mode='max'),
        EarlyStopping(patience=10, verbose=1, monitor='val_dice_coef', mode='max')
    ]

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    print("Training finished.")

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], label='Training Dice Coef')
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef')
    plt.title('Dice Coefficient')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.show()

# --- 6. INFERENCE SCRIPT (Example) ---

def predict_full_image(model, image_path):
    """
    Performs patch-based prediction on a full image and stitches the result.
    """
    stride = int(PATCH_SIZE * (1 - OVERLAP))

    # Load and normalize the image
    full_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_h, img_w = full_image.shape
    full_image_normalized = full_image.astype('float32') / 255.0

    # Create placeholders for the final mask and a counter for averaging overlaps
    full_prediction = np.zeros((img_h, img_w), dtype=np.float32)
    overlap_count = np.zeros((img_h, img_w), dtype=np.float32)

    # Pad the image to be a multiple of the patch size if needed
    pad_h = (stride - (img_h - PATCH_SIZE) % stride) % stride
    pad_w = (stride - (img_w - PATCH_SIZE) % stride) % stride
    padded_image = np.pad(full_image_normalized, ((0, pad_h), (0, pad_w)), mode='constant')
    
    padded_h, padded_w = padded_image.shape
    
    # Iterate over the padded image and predict on patches
    for y in range(0, padded_h - PATCH_SIZE + 1, stride):
        for x in range(0, padded_w - PATCH_SIZE + 1, stride):
            patch = padded_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_expanded = np.expand_dims(np.expand_dims(patch, axis=-1), axis=0) # (1, 256, 256, 1)
            
            # Predict
            pred_patch = model.predict(patch_expanded)[0, :, :, 0]
            
            # Add the prediction to the full mask
            full_prediction[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred_patch
            overlap_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Average the predictions in the overlapping regions
    # Add a small epsilon to avoid division by zero
    final_prediction = full_prediction / (overlap_count + 1e-8)
    
    # Crop back to original image size
    final_prediction = final_prediction[0:img_h, 0:img_w]
    
    # Binarize the final mask
    final_mask = (final_prediction > 0.1).astype(np.uint8) * 255

    return full_image, final_mask

