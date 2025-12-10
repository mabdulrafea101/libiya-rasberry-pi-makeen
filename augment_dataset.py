"""
Dataset Augmentation Script for Bottle Classification
======================================================
This script:
1. Balances classes by augmenting underrepresented classes
2. Applies realistic augmentations (noise, blur, brightness, rotation)
3. Creates a new dataset with 70% train, 20% valid, 10% test split
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
from collections import defaultdict

# Configuration
SOURCE_DATASET = "bottle classification.v7-version-5.folder"
OUTPUT_DATASET = "bottle_classification_augmented"
TARGET_IMAGES_PER_CLASS = 300  # Target number of images per class
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Augmentation functions
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to simulate camera noise."""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_motion_blur(image, size=5):
    """Add motion blur to simulate camera shake."""
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def adjust_brightness(image, factor=None):
    """Adjust brightness randomly or by a factor."""
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor=None):
    """Adjust contrast."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def rotate_image(image, angle=None):
    """Rotate image by a random angle."""
    if angle is None:
        angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def add_shadow(image):
    """Add random shadow effect."""
    h, w = image.shape[:2]
    top_y = random.randint(0, h // 2)
    bottom_y = random.randint(h // 2, h)
    shadow = image.copy().astype(np.float32)
    shadow[top_y:bottom_y, :] *= random.uniform(0.4, 0.7)
    return np.clip(shadow, 0, 255).astype(np.uint8)

def apply_random_augmentation(image, augmentation_strength='medium'):
    """Apply random combination of augmentations."""
    augmented = image.copy()
    
    # Randomly select augmentations
    augmentations = []
    
    if augmentation_strength in ['medium', 'strong']:
        if random.random() > 0.5:
            augmentations.append(('brightness', lambda img: adjust_brightness(img)))
        if random.random() > 0.5:
            augmentations.append(('contrast', lambda img: adjust_contrast(img)))
        if random.random() > 0.6:
            augmentations.append(('rotation', lambda img: rotate_image(img)))
    
    if augmentation_strength == 'strong':
        if random.random() > 0.4:
            augmentations.append(('noise', lambda img: add_gaussian_noise(img, sigma=random.randint(10, 30))))
        if random.random() > 0.5:
            augmentations.append(('blur', lambda img: add_motion_blur(img, size=random.choice([3, 5]))))
        if random.random() > 0.6:
            augmentations.append(('shadow', lambda img: add_shadow(img)))
    
    # Apply selected augmentations
    for name, aug_func in augmentations:
        augmented = aug_func(augmented)
    
    return augmented

def load_images_from_class(class_path):
    """Load all images from a class directory."""
    images = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for ext in extensions:
        images.extend(list(class_path.glob(f'*{ext}')))
        images.extend(list(class_path.glob(f'*{ext.upper()}')))
    
    return images

def augment_class(class_name, source_images, target_count, output_dir):
    """Augment a class to reach target count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_count = len(source_images)
    needed = target_count - current_count
    
    print(f"\n  Class: {class_name}")
    print(f"  Current: {current_count} images")
    print(f"  Target: {target_count} images")
    print(f"  Generating: {needed} augmented images")
    
    all_images = []
    
    # Copy original images
    for idx, img_path in enumerate(tqdm(source_images, desc=f"  Copying originals")):
        img = cv2.imread(str(img_path))
        if img is not None:
            output_path = output_dir / f"{class_name}_orig_{idx:04d}.jpg"
            cv2.imwrite(str(output_path), img)
            all_images.append(output_path)
    
    # Generate augmented images
    if needed > 0:
        augmented_count = 0
        pbar = tqdm(total=needed, desc=f"  Augmenting")
        
        while augmented_count < needed:
            # Randomly select source image
            source_img_path = random.choice(source_images)
            img = cv2.imread(str(source_img_path))
            
            if img is not None:
                # Determine augmentation strength based on how many we need
                strength = 'strong' if needed > current_count * 2 else 'medium'
                augmented_img = apply_random_augmentation(img, strength)
                
                output_path = output_dir / f"{class_name}_aug_{augmented_count:04d}.jpg"
                cv2.imwrite(str(output_path), augmented_img)
                all_images.append(output_path)
                
                augmented_count += 1
                pbar.update(1)
        
        pbar.close()
    
    return all_images

def split_dataset(all_images_by_class, output_base_dir, train_ratio, valid_ratio, test_ratio):
    """Split dataset into train/valid/test sets."""
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    splits = {
        'train': train_ratio,
        'valid': valid_ratio,
        'test': test_ratio
    }
    
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for class_name, images in all_images_by_class.items():
        # Shuffle images
        random.shuffle(images)
        total = len(images)
        
        # Calculate split indices
        train_end = int(total * train_ratio)
        valid_end = train_end + int(total * valid_ratio)
        
        split_images = {
            'train': images[:train_end],
            'valid': images[train_end:valid_end],
            'test': images[valid_end:]
        }
        
        # Copy images to respective directories
        for split_name, split_images_list in split_images.items():
            split_dir = output_base_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images_list:
                dest_path = split_dir / img_path.name
                shutil.copy2(img_path, dest_path)
                split_counts[split_name][class_name] += 1
    
    # Print summary
    print(f"\n{'Split':<10} {'Class':<20} {'Count':<10}")
    print("-" * 60)
    for split_name in ['train', 'valid', 'test']:
        for class_name in sorted(split_counts[split_name].keys()):
            count = split_counts[split_name][class_name]
            print(f"{split_name:<10} {class_name:<20} {count:<10}")
        print("-" * 60)
    
    return split_counts

def main():
    print("="*60)
    print("BOTTLE CLASSIFICATION DATASET AUGMENTATION")
    print("="*60)
    
    source_path = Path(SOURCE_DATASET)
    temp_dir = Path("temp_augmented_images")
    output_path = Path(OUTPUT_DATASET)
    
    # Clean up previous runs
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    temp_dir.mkdir(exist_ok=True)
    
    # Collect all original images from train, valid, and test
    all_original_images = defaultdict(list)
    
    for split in ['train', 'valid', 'test']:
        split_path = source_path / split
        if split_path.exists():
            for class_dir in split_path.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    class_name = class_dir.name
                    images = load_images_from_class(class_dir)
                    all_original_images[class_name].extend(images)
    
    print("\nOriginal Dataset Summary:")
    print("-" * 60)
    for class_name, images in sorted(all_original_images.items()):
        print(f"{class_name:<20} {len(images):>5} images")
    print("-" * 60)
    
    # Augment each class
    print("\n" + "="*60)
    print("AUGMENTING CLASSES")
    print("="*60)
    
    all_augmented_images = {}
    
    for class_name, original_images in all_original_images.items():
        class_output_dir = temp_dir / class_name
        augmented = augment_class(
            class_name, 
            original_images, 
            TARGET_IMAGES_PER_CLASS, 
            class_output_dir
        )
        all_augmented_images[class_name] = augmented
    
    # Split into train/valid/test
    split_counts = split_dataset(
        all_augmented_images,
        output_path,
        TRAIN_RATIO,
        VALID_RATIO,
        TEST_RATIO
    )
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f"\n✅ New dataset created at: {output_path}")
    print(f"\n📊 Final Statistics:")
    print(f"   - Images per class: {TARGET_IMAGES_PER_CLASS}")
    print(f"   - Total images: {TARGET_IMAGES_PER_CLASS * len(all_original_images)}")
    print(f"   - Train/Valid/Test: {TRAIN_RATIO*100:.0f}% / {VALID_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")
    print(f"\n💡 Next Steps:")
    print(f"   1. Update your notebook to use: '{OUTPUT_DATASET}'")
    print(f"   2. Change: dataset_path = Path('{OUTPUT_DATASET}')")
    print(f"   3. Retrain the model with the balanced dataset")
    print("="*60)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    main()
