import os
import numpy as np
import shutil
import sys
import cv2
import torch
import json
import rawpy
import exifread
from datetime import datetime
from tqdm import tqdm
from glob import glob
from PIL import Image
from preprocess_NEF import process_nef_folder
from get_mask import lazy_load_birefnet, extract_leaves_and_masks, get_birefnet_mask, align_leaf_orientation
from scipy.ndimage import find_objects
from skimage.measure import label
from utils import (
    resize_with_padding, 
    get_nb_stage,
    process_normal,
    depadding,
    normal_to_rgb
)
sys.path.append('./Uni-MS-PS')  # Add the Uni-MS-PS folder to path
from utils import load_model
from run import run

def compress_raw_data(base_folder: str, leaf_resolution=1920):
    """
    Process all subfolders in base_folder, each representing a blender object asset.
    
    Step 1: 
    - Look for existing 'raw' folders in each asset folder
    - Create 'compressed' folder and convert NEF files to PNG
    - If no 'raw' folder exists, look for NEF files in root and create 'raw' folder
    
    Args:
        base_folder: Path to folder containing object asset folders
        leaf_resolution: Target resolution for compressed images
    """
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(object_folders)} object folders to process")
    
    # Progress bar for processing object folders
    with tqdm(object_folders, desc="Compressing RAW data", unit="folder") as pbar:
        for folder_name in pbar:
            pbar.set_postfix_str(f"Processing {folder_name}")
            folder_path = os.path.join(base_folder, folder_name)
            raw_folder = os.path.join(folder_path, 'raw')
            compressed_folder = os.path.join(folder_path, 'compressed')
            
            # Check if 'raw' folder already exists
            if os.path.exists(raw_folder):
                # Find NEF files in the raw folder
                nef_files = glob(os.path.join(raw_folder, '*.NEF'))
                nef_files.extend(glob(os.path.join(raw_folder, '*.nef')))
                
                if not nef_files:
                    pbar.set_postfix_str(f"{folder_name}: No NEF files in raw")
                    continue
                
                pbar.set_postfix_str(f"{folder_name}: Found {len(nef_files)} NEF files")
                
            else:
                # Find NEF files in the current folder root
                nef_files = glob(os.path.join(folder_path, '*.NEF'))
                nef_files.extend(glob(os.path.join(folder_path, '*.nef')))
                
                if not nef_files:
                    pbar.set_postfix_str(f"{folder_name}: No NEF files found")
                    continue
                
                # Create 'raw' folder and move NEF files there
                os.makedirs(raw_folder, exist_ok=True)
                
                for nef_file in nef_files:
                    destination = os.path.join(raw_folder, os.path.basename(nef_file))
                    shutil.move(nef_file, destination)
                
                # Update nef_files list to point to new locations
                nef_files = glob(os.path.join(raw_folder, '*.NEF'))
                nef_files.extend(glob(os.path.join(raw_folder, '*.nef')))
                
                pbar.set_postfix_str(f"{folder_name}: Moved {len(nef_files)} NEF files to raw/")
            
            # Check if compressed folder already exists and has content
            if os.path.exists(compressed_folder):
                existing_png_files = glob(os.path.join(compressed_folder, '*.png'))
                if existing_png_files and len(existing_png_files) >= len(nef_files):
                    pbar.set_postfix_str(f"{folder_name}: Already processed")
                    continue
            
            # Create 'compressed' folder if it doesn't exist
            os.makedirs(compressed_folder, exist_ok=True)
            
            try:
                pbar.set_postfix_str(f"{folder_name}: Converting to PNG...")
                # Use the preprocess_NEF function to convert files
                process_nef_folder(
                    input_folder=raw_folder,
                    target_resolution=leaf_resolution,
                    output_subfolder='../compressed'
                )
                
                # Verify the output
                output_files = glob(os.path.join(compressed_folder, '*.png'))
                pbar.set_postfix_str(f"{folder_name}: Created {len(output_files)} PNG files")
                
            except Exception as e:
                pbar.set_postfix_str(f"{folder_name}: Error - {str(e)}")
                continue

def extract_leaves_from_compressed(base_folder: str):
    """
    Step 2: Process compressed folders to extract individual leaves with accurate scaling
    """
    
    # Load the BiRefNet model once
    print("Loading BiRefNet model...")
    mask_model = lazy_load_birefnet()
    print("Model loaded successfully!")
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(object_folders)} object folders to process for leaf extraction")
    
    # Progress bar for processing object folders
    with tqdm(object_folders[12:], desc="Extracting leaves", unit="folder") as folder_pbar:
        for folder_name in folder_pbar:
            folder_pbar.set_postfix_str(f"Processing {folder_name}")
            folder_path = os.path.join(base_folder, folder_name)
            compressed_folder = os.path.join(folder_path, 'compressed')
            
            if not os.path.exists(compressed_folder):
                folder_pbar.set_postfix_str(f"{folder_name}: No compressed folder")
                continue
            
            # Find _diffuse.png image
            diffuse_files = glob(os.path.join(compressed_folder, '*_diffuse.png'))
            
            if not diffuse_files:
                folder_pbar.set_postfix_str(f"{folder_name}: No diffuse image")
                os.error(f"{folder_name}: No diffuse image found. Make sure one image is named _diffuse.png or .NEF")
                sys.exit(1)
            
            # Use the first diffuse file found
            diffuse_path = diffuse_files[0]
            diffuse_image = Image.open(diffuse_path)
            
            # Generate mask and get leaf extraction data
            mask = get_birefnet_mask(diffuse_image.convert('RGB'), mask_model)
            labeled = label(mask)
            slices = find_objects(labeled)
            
            if not slices:
                folder_pbar.set_postfix_str(f"{folder_name}: No leaves detected")
                continue
            
            diffuse_rgba = diffuse_image.convert('RGBA')
            diffuse_np = np.array(diffuse_rgba)
            
            # Get all images in compressed folder (except full_photo.png)
            all_images = []
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                all_images.extend(glob(os.path.join(compressed_folder, f'*{ext}')))
            
            # Filter out full_photo.png files
            filtered_images = [img for img in all_images 
                                if not os.path.basename(img).endswith('full_photo.png')]
            
            folder_pbar.set_postfix_str(f"{folder_name}: Processing {len([s for s in slices if s is not None])} leaves")
            
            # Process each detected leaf using the SAME extraction logic
            leaf_count = 0
            
            # Progress bar for processing leaves within this folder
            for idx, slc in enumerate(slices):
                if slc is None:
                    continue
                    
                leaf_mask = (labeled[slc] == (idx + 1)).astype(np.uint8) * 255
                if leaf_mask.sum() < 100:  # Skip very small detected regions
                    continue
                    
                leaf_count += 1
                leaf_folder = os.path.join(compressed_folder, f'leaf_{leaf_count}')
                os.makedirs(leaf_folder, exist_ok=True)
                
                # Apply the EXACT SAME processing as in extract_leaves_and_masks
                crop_diffuse = diffuse_np[slc].copy()
                crop_mask = leaf_mask
                
                # Set alpha channel to mask
                crop_diffuse[..., 3] = crop_mask
                
                # Apply same transformations (rotation, alignment)
                crop_diffuse, crop_mask, _ = align_leaf_orientation(crop_diffuse, crop_mask, None)
                
                # Save the processed diffuse and mask
                leaf_img = Image.fromarray(crop_diffuse)
                mask_img = Image.fromarray(crop_mask, mode='L')
                
                leaf_img.save(os.path.join(leaf_folder, f'leaf_{leaf_count}_diffuse.png'))
                mask_img.save(os.path.join(leaf_folder, f'leaf_{leaf_count}_mask.png'))
                
                # Process other images with progress bar
                other_images = [img for img in filtered_images if img != diffuse_path]
                for img_path in other_images:
                    # Load the image at full resolution
                    img = Image.open(img_path)
                    
                    # Ensure image is same size as diffuse image
                    if img.size != diffuse_image.size:
                        img = img.resize(diffuse_image.size, Image.Resampling.LANCZOS)
                    
                    # Convert to numpy and apply EXACT SAME cropping
                    if img.mode != 'RGBA':
                        img = img.convert('RGB')
                    img_np = np.array(img)
                    
                    # Apply the same slice cropping
                    crop_img = img_np[slc].copy()
                    
                    # Apply the SAME orientation transformation AND GET THE TRANSFORMED MASK
                    crop_img, transformed_mask, _ = align_leaf_orientation(crop_img, leaf_mask, None)
                    
                    # Apply mask to the cropped image using the transformed mask
                    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:  # RGB
                        # Create masked version with white background
                        mask_3d = np.stack([transformed_mask, transformed_mask, transformed_mask], axis=2) > 128
                        crop_img[~mask_3d] = 255  # White background where mask is 0
                        final_img = Image.fromarray(crop_img, mode='RGB')
                    elif len(crop_img.shape) == 2:  # Grayscale
                        crop_img[transformed_mask <= 128] = 255  # White background
                        final_img = Image.fromarray(crop_img, mode='L')
                    else:
                        final_img = Image.fromarray(crop_img)
                    
                    # Save with descriptive name
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    output_name = f'leaf_{leaf_count}_{base_name}.png'
                    output_path = os.path.join(leaf_folder, output_name)
                    final_img.save(output_path)
            
            folder_pbar.set_postfix_str(f"{folder_name}: Extracted {leaf_count} leaves")

def generate_normal_maps(base_folder: str):
    """
    Step 3: Generate normal maps for each leaf folder using Uni-MS-PS
    """
    
    # Load the Uni-MS-PS model once
    print("Loading Uni-MS-PS model...")
    try:
        model = load_model(
            path_weight='./Uni-MS-PS/weights',
            cuda=True,
            mode_inference=True,
            calibrated=False
        )
        print("Uni-MS-PS model loaded successfully!")
    except Exception as e:
        print(f"Error loading Uni-MS-PS model: {str(e)}")
        return
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    # Count total leaf folders for progress tracking
    total_leaf_folders = 0
    folder_leaf_counts = {}
    
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        compressed_folder = os.path.join(folder_path, 'compressed')
        
        if os.path.exists(compressed_folder):
            leaf_folders = [d for d in os.listdir(compressed_folder) 
                           if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
            folder_leaf_counts[folder_name] = len(leaf_folders)
            total_leaf_folders += len(leaf_folders)
    
    print(f"Found {total_leaf_folders} leaf folders across {len(object_folders)} objects")
    
    # Progress bar for all leaf folders
    with tqdm(total=total_leaf_folders, desc="Generating normal maps", unit="leaf") as pbar:
        for folder_name in object_folders:
            folder_path = os.path.join(base_folder, folder_name)
            compressed_folder = os.path.join(folder_path, 'compressed')
            
            if not os.path.exists(compressed_folder):
                continue
            
            # Find all leaf_X folders
            leaf_folders = [d for d in os.listdir(compressed_folder) 
                           if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
            
            if not leaf_folders:
                continue
            
            for leaf_folder_name in leaf_folders:
                pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}")
                leaf_folder_path = os.path.join(compressed_folder, leaf_folder_name)
                
                try:
                    # Check if we have enough images (excluding diffuse and mask)
                    all_files = [f for f in os.listdir(leaf_folder_path) 
                               if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
                    
                    # Filter out diffuse and mask files
                    image_files = [f for f in all_files 
                                 if not f.endswith('_diffuse.png') and not f.endswith('_mask.png')]
                    
                    if len(image_files) < 3:
                        pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Not enough images")
                        pbar.update(1)
                        continue
                    
                    # Check if normal map already exists
                    normal_map_path = os.path.join(leaf_folder_path, f'{leaf_folder_name}_normal.png')
                    if os.path.exists(normal_map_path):
                        pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Already exists")
                        pbar.update(1)
                        continue
                    
                    # Generate normal map
                    imgs, mask, padding, zoom_coord, original_shape = load_imgs_mask_custom(
                        path=leaf_folder_path,
                        nb_img=-1,
                        max_size=512,
                        calibrated=False,
                        exclude_files=['_diffuse.png', '_mask.png']
                    )
                    
                    if imgs is None:
                        pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Failed to load")
                        pbar.update(1)
                        continue
                    
                    normal = process_normal(model=model, imgs=imgs, mask=mask)
                    
                    # Post-process the normal map
                    normal_resize = depadding(normal, padding=padding)
                    normal_resize = torch.from_numpy(normal_resize)
                    normal_resize = torch.nn.functional.normalize(normal_resize, 2, -1).numpy()
                    
                    # Restore original size
                    pad_x_min = np.zeros((zoom_coord[0], normal_resize.shape[1], 3))
                    pad_x_max = np.zeros((zoom_coord[1], normal_resize.shape[1], 3))
                    normal_resize = np.concatenate((pad_x_min, normal_resize, pad_x_max), axis=0)
                            
                    pad_y_min = np.zeros((normal_resize.shape[0], zoom_coord[2], 3))
                    pad_y_max = np.zeros((normal_resize.shape[0], zoom_coord[3], 3))
                    normal_resize = np.concatenate((pad_y_min, normal_resize, pad_y_max), axis=1)
                    
                    normal_resize = cv2.resize(normal_resize,
                                            (original_shape[1], original_shape[0]),
                                            interpolation=cv2.INTER_LINEAR)
                    normal_resize = np.clip(normal_resize, -1, 1)
                    normal_resize_rgb = normal_to_rgb(normal_resize)
                    
                    # Save the normal map
                    cv2.imwrite(normal_map_path, normal_resize_rgb[:,:,::-1])
                    pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Generated")
                    
                except Exception as e:
                    pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Error")
                
                pbar.update(1)

def organize_final_structure(base_folder: str):
    """
    Step 4: Organize final structure
    """
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    # Count total leaf folders for progress tracking
    total_leaf_folders = 0
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        compressed_folder = os.path.join(folder_path, 'compressed')
        
        if os.path.exists(compressed_folder):
            leaf_folders = [d for d in os.listdir(compressed_folder) 
                           if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
            total_leaf_folders += len(leaf_folders)
    
    print(f"Organizing {total_leaf_folders} leaf folders across {len(object_folders)} objects")
    
    # Progress bar for organizing folders
    with tqdm(total=total_leaf_folders, desc="Organizing structure", unit="leaf") as pbar:
        for folder_name in object_folders:
            folder_path = os.path.join(base_folder, folder_name)
            compressed_folder = os.path.join(folder_path, 'compressed')
            
            if not os.path.exists(compressed_folder):
                continue
            
            # Find all leaf_X folders in compressed directory
            leaf_folders = [d for d in os.listdir(compressed_folder) 
                           if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
            
            if not leaf_folders:
                continue
            
            for leaf_folder_name in leaf_folders:
                pbar.set_postfix_str(f"Moving {folder_name}/{leaf_folder_name}")
                old_leaf_path = os.path.join(compressed_folder, leaf_folder_name)
                new_leaf_path = os.path.join(folder_path, leaf_folder_name)
                
                try:
                    # Move the leaf folder to root level
                    if os.path.exists(new_leaf_path):
                        shutil.rmtree(new_leaf_path)
                    
                    shutil.move(old_leaf_path, new_leaf_path)
                    
                    # Create 'processed' subfolder
                    processed_folder = os.path.join(new_leaf_path, 'processed')
                    os.makedirs(processed_folder, exist_ok=True)
                    
                    # Define files to move to processed folder
                    files_to_move = [
                        f'{leaf_folder_name}_mask.png',
                        f'{leaf_folder_name}_diffuse.png', 
                        f'{leaf_folder_name}_normal.png'
                    ]
                    
                    # Move specified files to processed folder
                    moved_count = 0
                    for file_name in files_to_move:
                        source_path = os.path.join(new_leaf_path, file_name)
                        dest_path = os.path.join(processed_folder, file_name)
                        
                        if os.path.exists(source_path):
                            shutil.move(source_path, dest_path)
                            moved_count += 1
                    
                    pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Moved {moved_count} files")
                    
                except Exception as e:
                    pbar.set_postfix_str(f"{folder_name}/{leaf_folder_name}: Error")
                
                pbar.update(1)

def load_imgs_mask_custom(path, nb_img, max_size=None, calibrated=False, exclude_files=None):
    """
    Custom version of load_imgs_mask that excludes specific files
    """
    
    if exclude_files is None:
        exclude_files = []
    
    # Get all image files
    possible_files = os.listdir(path)
    temp = []
    
    for file in possible_files:
        # Check if it's an image file and not in exclude list
        if any(ext in file.lower() for ext in [".png", ".jpg", ".jpeg", ".tif"]):
            if not any(exclude in file for exclude in exclude_files):
                if "mask" not in file.lower() and "normal" not in file.lower():
                    temp.append(file)
    
    if len(temp) < 3:
        return None, None, None, None, None
    
    # Load mask
    mask_files = [f for f in os.listdir(path) if f.endswith('_mask.png')]
    if mask_files:
        mask_path = os.path.join(path, mask_files[0])
        mask = cv2.imread(mask_path)
    else:
        # Create a default mask from the first image
        first_img_path = os.path.join(path, temp[0])
        first_img = cv2.imread(first_img_path)
        mask = np.ones(first_img.shape, dtype=np.uint8) * 255
    
    original_shape = mask.shape
    
    if max_size is not None:
        if mask.shape[0] > max_size or mask.shape[1] > max_size:
            mask = cv2.resize(mask, (max_size, max_size))
    
    # Find bounding box of mask
    coord = np.argwhere(mask[:,:,0] > 0)
    if len(coord) == 0:
        return None, None, None, None, None
        
    x_min, x_max = np.min(coord[:,0]), np.max(coord[:,0])
    y_min, y_max = np.min(coord[:,1]), np.max(coord[:,1])
    
    x_max_pad = mask.shape[0] - x_max
    y_max_pad = mask.shape[1] - y_max
    
    mask = mask[x_min:x_max, y_min:y_max]
    
    # Prepare for model input
    nb_stage = get_nb_stage(mask.shape)
    size_img = 32 * 2**(nb_stage-1)
    
    mask, _ = resize_with_padding(mask, expected_size=(size_img, size_img))
    mask = (mask > 0)
    mask = mask[:,:,0]
    
    # Load and process images
    imgs = []
    
    if nb_img is None or nb_img >= len(temp) or nb_img == -1:
        files = np.array(temp)
    else:
        files = np.random.choice(temp, nb_img, replace=False)
    
    for file in files:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
            img = np.concatenate((img, img, img), axis=-1)
        
        if max_size is not None:
            if img.shape[0] > max_size or img.shape[1] > max_size:
                img = cv2.resize(img, (max_size, max_size))
        
        img = img[x_min:x_max, y_min:y_max]
        img, padding = resize_with_padding(img=img, expected_size=(size_img, size_img))
        
        img = img.astype(np.float32)
        mean_img = np.mean(img, -1)
        mean_img = mean_img.flatten()
        mean_img1 = np.mean(mean_img[mask.flatten()])
        img = img / mean_img1
        
        imgs.append(img)
    
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs, -1, 0)
    imgs = torch.from_numpy(imgs).unsqueeze(0).float()
    
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    return imgs, mask, padding, [x_min, x_max_pad, y_min, y_max_pad], original_shape

def calculate_leaf_sizes(base_folder: str):
    """
    Step 5: Calculate leaf sizes in cm and save as JSON files
    
    - Read camera parameters from NEF files (focal length)
    - Load and convert NEF to full-resolution PNG for measurements
    - Calculate pixel to cm conversion factor using focal length and distance
    - Measure leaf sizes from original full-resolution masks
    - Save leaf size data as JSON files in each object folder
    
    Args:
        base_folder: Path to folder containing object asset folders
    """
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(object_folders)} object folders to calculate leaf sizes")
    
    # Initialize summary data for the entire dataset
    dataset_summary = {
        "dataset_info": {
            "base_folder": base_folder,
            "processing_date": str(datetime.now()),
            "total_objects": len(object_folders),
            "total_leaves": 0
        },
        "objects": {}
    }
    
    # Progress bar for calculating leaf sizes
    with tqdm(object_folders, desc="Calculating leaf sizes", unit="object") as pbar:
        for folder_name in pbar:
            pbar.set_postfix_str(f"Processing {folder_name}")
            folder_path = os.path.join(base_folder, folder_name)
            raw_folder = os.path.join(folder_path, 'raw')
            compressed_folder = os.path.join(folder_path, 'compressed')
            
            if not os.path.exists(raw_folder) or not os.path.exists(compressed_folder):
                pbar.set_postfix_str(f"{folder_name}: Missing folders")
                continue
            
            # Initialize object data
            object_data = {
                "object_name": folder_name,
                "camera_parameters": {},
                "leaves": {},
                "leaf_count": 0
            }
            
            try:
                # Get NEF files
                nef_files = glob(os.path.join(raw_folder, '*.NEF')) + glob(os.path.join(raw_folder, '*.nef'))
                if not nef_files:
                    pbar.set_postfix_str(f"{folder_name}: No NEF files")
                    continue
                
                # Find the diffuse NEF file
                diffuse_files = glob(os.path.join(compressed_folder, '*_diffuse.png'))
                if not diffuse_files:
                    pbar.set_postfix_str(f"{folder_name}: No diffuse image")
                    continue
                
                # Get the base name of the diffuse file to find corresponding NEF
                diffuse_basename = os.path.basename(diffuse_files[0]).replace('_diffuse.png', '')
                diffuse_nef = None
                
                for nef_file in nef_files:
                    nef_basename = os.path.splitext(os.path.basename(nef_file))[0]
                    if nef_basename == diffuse_basename:
                        diffuse_nef = nef_file
                        break
                
                if diffuse_nef is None:
                    diffuse_nef = nef_files[0]
                
                pbar.set_postfix_str(f"{folder_name}: Reading EXIF data")
                
                # Read EXIF data from NEF file for focal length
                with open(diffuse_nef, 'rb') as f:
                    tags = exifread.process_file(f)
                    
                    # Get focal length
                    if 'EXIF FocalLength' in tags:
                        f_mm = float(tags['EXIF FocalLength'].printable)
                    else:
                        f_mm = 85.0
                
                pbar.set_postfix_str(f"{folder_name}: Converting NEF to full resolution")
                
                with rawpy.imread(diffuse_nef) as raw:
                    # Process RAW to RGB without downscaling
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        output_bps=8,
                        use_camera_wb=True,
                        half_size=False,
                        four_color_rgb=False,
                        dcb_enhance=False,
                        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                        noise_thr=None
                    )
                
                # Convert to PIL Image
                diffuse_image = Image.fromarray(rgb)
                res_w, res_h = diffuse_image.size
                
                # Camera sensor specifications
                sensor_w_mm = 23.5
                sensor_h_mm = 15.6
                z_mm = 280.0
                
                # Determine orientation and map sensor dimensions correctly
                if res_w > res_h:  # Landscape orientation
                    px_w_mm = sensor_w_mm / res_w
                    px_h_mm = sensor_h_mm / res_h
                else:  # Portrait orientation
                    px_w_mm = sensor_h_mm / res_w
                    px_h_mm = sensor_w_mm / res_h
                
                # Calculate pixels per cm
                def pixels_per_cm(f_mm, z_mm, px_mm):
                    return (f_mm * 10.0) / ((z_mm - f_mm) * px_mm)
                
                ppcm_w = pixels_per_cm(f_mm, z_mm, px_w_mm)
                ppcm_h = pixels_per_cm(f_mm, z_mm, px_h_mm)
                
                # Store camera parameters
                object_data["camera_parameters"] = {
                    "focal_length_mm": f_mm,
                    "distance_mm": z_mm,
                    "sensor_width_mm": sensor_w_mm,
                    "sensor_height_mm": sensor_h_mm,
                    "image_width": res_w,
                    "image_height": res_h,
                    "orientation": "landscape" if res_w > res_h else "portrait",
                    "pixel_size_w_mm": round(px_w_mm, 6),
                    "pixel_size_h_mm": round(px_h_mm, 6),
                    "pixels_per_cm_horizontal": round(ppcm_w, 2),
                    "pixels_per_cm_vertical": round(ppcm_h, 2)
                }
                
                pbar.set_postfix_str(f"{folder_name}: Generating mask")
                
                # Load BiRefNet model and generate mask
                mask_model = lazy_load_birefnet()
                mask = get_birefnet_mask(diffuse_image.convert('RGB'), mask_model)
                labeled = label(mask)
                slices = find_objects(labeled)
                
                if not slices:
                    pbar.set_postfix_str(f"{folder_name}: No leaves detected")
                    continue
                
                # Find all leaf folders to match with detected leaves
                leaf_folders = [d for d in os.listdir(folder_path) 
                               if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('leaf_')]
                
                if not leaf_folders:
                    pbar.set_postfix_str(f"{folder_name}: No leaf folders")
                    continue
                
                pbar.set_postfix_str(f"{folder_name}: Measuring {len(leaf_folders)} leaves")
                
                # Process each detected leaf
                leaf_count = 0
                for idx, slc in enumerate(slices):
                    if slc is None:
                        continue
                        
                    leaf_mask = (labeled[slc] == (idx + 1)).astype(np.uint8) * 255
                    if leaf_mask.sum() < 100:
                        continue
                    
                    leaf_count += 1
                    leaf_folder_name = f'leaf_{leaf_count}'
                    leaf_folder_path = os.path.join(folder_path, leaf_folder_name)
                    
                    if not os.path.exists(leaf_folder_path):
                        continue
                    
                    # Calculate leaf dimensions from mask at FULL resolution
                    crop_mask_original = leaf_mask.copy()
                    
                    # Get the cropped region from FULL resolution image
                    diffuse_rgba = diffuse_image.convert('RGBA')
                    diffuse_np = np.array(diffuse_rgba)
                    crop_diffuse = diffuse_np[slc].copy()
                    crop_diffuse[..., 3] = crop_mask_original
                    
                    # Apply same transformations to get final oriented mask
                    crop_diffuse_transformed, crop_mask_transformed, _ = align_leaf_orientation(
                        crop_diffuse, crop_mask_original, None
                    )
                    
                    # Calculate dimensions from transformed mask
                    mask_coords = np.argwhere(crop_mask_transformed > 128)
                    if len(mask_coords) == 0:
                        continue
                    
                    # Get bounding box in pixels
                    y_min, x_min = mask_coords.min(axis=0)
                    y_max, x_max = mask_coords.max(axis=0)
                    
                    # Calculate dimensions in pixels
                    width_pixels = x_max - x_min
                    height_pixels = y_max - y_min
                    
                    # Convert to cm using focal length-based pixel per cm ratios
                    width_cm = width_pixels / ppcm_w
                    height_cm = height_pixels / ppcm_h
                    
                    # Calculate area (precise count of mask pixels)
                    area_pixels = np.sum(crop_mask_transformed > 128)
                    area_cm2 = area_pixels / (ppcm_w * ppcm_h)
                    
                    # Create leaf data
                    leaf_data = {
                        "leaf_id": leaf_folder_name,
                        "object_name": folder_name,
                        "dimensions": {
                            "width_cm": round(width_cm, 3),
                            "height_cm": round(height_cm, 3),
                            "area_cm2": round(area_cm2, 3)
                        },
                        "pixels": {
                            "width_pixels": int(width_pixels),
                            "height_pixels": int(height_pixels),
                            "area_pixels": int(area_pixels)
                        },
                        "processing_info": {
                            "measured_at_full_resolution": True,
                            "original_image_size": [res_w, res_h],
                            "transformations_applied": ["cropping", "orientation_alignment"],
                            "measurement_method": "focal_length_distance_based",
                            "source": "full_resolution_NEF"
                        }
                    }
                    
                    # Add leaf to object data
                    object_data["leaves"][leaf_folder_name] = leaf_data
                
                object_data["leaf_count"] = leaf_count
                dataset_summary["objects"][folder_name] = object_data
                dataset_summary["dataset_info"]["total_leaves"] += leaf_count
                
                # Save individual object JSON file
                object_json_path = os.path.join(folder_path, f'{folder_name}_leaves_data.json')
                with open(object_json_path, 'w') as f:
                    json.dump(object_data, f, indent=2)
                
                pbar.set_postfix_str(f"{folder_name}: Completed {leaf_count} leaves")
                
            except Exception as e:
                pbar.set_postfix_str(f"{folder_name}: Error - {str(e)}")
                continue
    
    # Save dataset summary
    summary_json_path = os.path.join(base_folder, 'dataset_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    print(f"\n✓ Saved dataset summary to: dataset_summary.json")
    print(f"Total objects processed: {dataset_summary['dataset_info']['total_objects']}")
    print(f"Total leaves measured: {dataset_summary['dataset_info']['total_leaves']}")

def main():
    """
    Main function - specify your base folder path here
    """
    # UPDATE THIS PATH TO YOUR FOLDER
    base_folder = '/mnt/e/projects/raw_datasets/lalweco/sugarbeets/nikon_camera/unprocessed/08-08-2025'
    # base_folder = '/mnt/e/projects/raw_datasets/lalweco/sugarbeets/nikon_camera/test_dataset'
    # Larger image size slows down the NORMAL generation process. 
    single_leaf_image_size = 1920
    
    print(f"Starting processing of folders in: {base_folder}")
    
    # Step 1: Compress raw data
    compress_raw_data(base_folder, leaf_resolution=single_leaf_image_size)
    
    # Step 2: Extract leaves from compressed images
    extract_leaves_from_compressed(base_folder)
    
    # Step 3: Generate normal maps for each leaf
    generate_normal_maps(base_folder)
    
    # Step 4: Organize final structure
    organize_final_structure(base_folder)
    
    # Step 5: Calculate leaf sizes in cm
    calculate_leaf_sizes(base_folder)
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()