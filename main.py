import os
import numpy as np
import shutil
import sys
import cv2
import torch
from glob import glob
from PIL import Image
from preprocess_NEF import process_nef_folder
from get_mask import lazy_load_birefnet, extract_leaves_and_masks, get_birefnet_mask, align_leaf_orientation
from scipy.ndimage import find_objects
from skimage.measure import label
from utils import resize_with_padding, get_nb_stage
sys.path.append('./Uni-MS-PS')  # Add the Uni-MS-PS folder to path
from utils import load_model
from run import run

def compress_raw_data(base_folder: str, leaf_resolution=1920):
    """
    Process all subfolders in base_folder, each representing a blender object asset.
    
    Step 1: 
    - Create 'raw' folder and move NEF files there
    - Create 'compressed' folder and convert NEF files to PNG
    
    Args:
        base_folder: Path to folder containing object asset folders
    """
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(object_folders)} object folders to process")
    
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        print(f"\nProcessing object folder: {folder_name}")
        
        # Find NEF files in the current folder
        nef_files = glob(os.path.join(folder_path, '*.NEF'))
        nef_files.extend(glob(os.path.join(folder_path, '*.nef')))
        
        if not nef_files:
            print(f"  No NEF files found in {folder_name}, skipping...")
            continue
        
        print(f"  Found {len(nef_files)} NEF files")
        
        # Step 1a: Create 'raw' folder and move NEF files there
        raw_folder = os.path.join(folder_path, 'raw')
        os.makedirs(raw_folder, exist_ok=True)
        
        print(f"  Moving NEF files to 'raw' folder...")
        for nef_file in nef_files:
            destination = os.path.join(raw_folder, os.path.basename(nef_file))
            shutil.move(nef_file, destination)
            print(f"    Moved: {os.path.basename(nef_file)}")
        
        # Step 1b: Create 'compressed' folder and convert NEF to PNG
        compressed_folder = os.path.join(folder_path, 'compressed')
        os.makedirs(compressed_folder, exist_ok=True)
        
        print(f"  Converting NEF files to PNG in 'compressed' folder...")
        try:
            # Use the preprocess_NEF function to convert files
            process_nef_folder(
                input_folder=raw_folder,
                target_resolution=leaf_resolution,
                output_subfolder='../compressed'  # Relative path to go up and into compressed
            )
            print(f"  ✓ Successfully processed {folder_name}")
        except Exception as e:
            print(f"  ✗ Error processing {folder_name}: {str(e)}")
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
    
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        compressed_folder = os.path.join(folder_path, 'compressed')
        
        if not os.path.exists(compressed_folder):
            print(f"  No compressed folder found in {folder_name}, skipping...")
            continue
            
        print(f"\nProcessing compressed folder: {folder_name}")
        
        # Find _diffuse.png image
        diffuse_files = glob(os.path.join(compressed_folder, '*_diffuse.png'))
        
        if not diffuse_files:
            print(f"  No _diffuse.png file found in {folder_name}/compressed, skipping...")
            continue
        
        # Use the first diffuse file found
        diffuse_path = diffuse_files[0]
        print(f"  Found diffuse image: {os.path.basename(diffuse_path)}")

        # Load the diffuse image
        diffuse_image = Image.open(diffuse_path)
        
        # Generate mask and get leaf extraction data
        mask = get_birefnet_mask(diffuse_image.convert('RGB'), mask_model)
        labeled = label(mask)
        slices = find_objects(labeled)
        
        if not slices:
            print(f"  No leaves detected in {folder_name}")
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
        
        print(f"  Found {len(filtered_images)} images to crop")
        
        # Process each detected leaf using the SAME extraction logic
        leaf_count = 0
        for idx, slc in enumerate(slices):
            if slc is None:
                continue
            leaf_mask = (labeled[slc] == (idx + 1)).astype(np.uint8) * 255
            if leaf_mask.sum() < 100:
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
            
            # Save the processed diffuse and mask (this matches the temp output)
            leaf_img = Image.fromarray(crop_diffuse)
            mask_img = Image.fromarray(crop_mask, mode='L')
            
            leaf_img.save(os.path.join(leaf_folder, f'leaf_{leaf_count}_diffuse.png'))
            mask_img.save(os.path.join(leaf_folder, f'leaf_{leaf_count}_mask.png'))
            
            # Now process all other images using the SAME slice and transformations
            for img_path in filtered_images:
                if img_path == diffuse_path:  # Skip the diffuse image we already processed
                    continue
                
                # Load the image at full resolution
                img = Image.open(img_path)
                
                # Ensure image is same size as diffuse image
                if img.size != diffuse_image.size:
                    print(f"    Warning: {os.path.basename(img_path)} size {img.size} doesn't match diffuse size {diffuse_image.size}")
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
                
            print(f"    Created leaf_{leaf_count} folder with cropped images")
        print(f"  ✓ Successfully processed {folder_name} - extracted {leaf_count} leaves")

def generate_normal_maps(base_folder: str):
    """
    Step 3: Generate normal maps for each leaf folder using Uni-MS-PS
    
    - Find all leaf_X folders in compressed directories
    - Load all images except _diffuse.png (including _mask.png)
    - Generate normal maps using the photometric stereo model
    - Save normal maps in the same leaf folder
    
    Args:
        base_folder: Path to folder containing object asset folders
    """
    
    # Load the Uni-MS-PS model once
    print("Loading Uni-MS-PS model...")
    try:
        model = load_model(
            path_weight='./Uni-MS-PS/weights',  # Adjust path as needed
            cuda=True,  # Set to False if no CUDA
            mode_inference=True,
            calibrated=False  # Set to True if you have calibrated light directions
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
    
    print(f"Found {len(object_folders)} object folders to process for normal map generation")
    
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        compressed_folder = os.path.join(folder_path, 'compressed')
        
        if not os.path.exists(compressed_folder):
            print(f"  No compressed folder found in {folder_name}, skipping...")
            continue
            
        print(f"\nProcessing compressed folder: {folder_name}")
        
        # Find all leaf_X folders
        leaf_folders = [d for d in os.listdir(compressed_folder) 
                       if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
        
        if not leaf_folders:
            print(f"  No leaf folders found in {folder_name}/compressed, skipping...")
            continue
        
        print(f"  Found {len(leaf_folders)} leaf folders")
        
        for leaf_folder_name in leaf_folders:
            leaf_folder_path = os.path.join(compressed_folder, leaf_folder_name)
            print(f"    Processing {leaf_folder_name}...")
            
            try:
                # Check if we have enough images (excluding diffuse and mask)
                all_files = [f for f in os.listdir(leaf_folder_path) 
                           if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))]
                
                # Filter out diffuse and mask files, but keep everything else
                image_files = [f for f in all_files 
                             if not f.endswith('_diffuse.png') and not f.endswith('_mask.png')]
                
                if len(image_files) < 3:
                    print(f"      Not enough images in {leaf_folder_name} (found {len(image_files)}, need at least 3)")
                    continue
                
                print(f"      Found {len(image_files)} images for photometric stereo")
                
                # Check if normal map already exists
                normal_map_path = os.path.join(leaf_folder_path, f'{leaf_folder_name}_normal.png')
                if os.path.exists(normal_map_path):
                    print(f"      Normal map already exists for {leaf_folder_name}, skipping...")
                    continue
                
                # Create a custom load_imgs_mask function for our specific needs
                from utils import load_imgs_mask
                
                # Temporarily modify the folder to only contain the images we want
                imgs, mask, padding, zoom_coord, original_shape = load_imgs_mask_custom(
                    path=leaf_folder_path,
                    nb_img=-1,  # Use all available images
                    max_size=512,  # Smaller resolution for faster processing
                    calibrated=False,
                    exclude_files=['_diffuse.png', '_mask.png']
                )
                
                if imgs is None:
                    print(f"      Failed to load images for {leaf_folder_name}")
                    continue
                
                # Generate normal map using the model
                from utils import process_normal, depadding, normal_to_rgb
                import torch
                import cv2
                
                normal = process_normal(model=model, imgs=imgs, mask=mask)
                
                # Post-process the normal map (same as in run.py)
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
                print(f"      ✓ Generated normal map: {leaf_folder_name}_normal.png")
                
            except Exception as e:
                print(f"      ✗ Error processing {leaf_folder_name}: {str(e)}")
                continue
        
        print(f"  ✓ Completed normal map generation for {folder_name}")

def organize_final_structure(base_folder: str):
    """
    Step 4: Organize final structure
    
    - Move all leaf_X folders to the root of each object folder
    - Create 'processed' subfolder in each leaf folder
    - Move _mask.png, _diffuse.png, _normal.png to the processed subfolder
    
    Args:
        base_folder: Path to folder containing object asset folders
    """
    
    # Find all subdirectories in the base folder
    object_folders = [d for d in os.listdir(base_folder) 
                     if os.path.isdir(os.path.join(base_folder, d))]
    
    if not object_folders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(object_folders)} object folders to organize")
    
    for folder_name in object_folders:
        folder_path = os.path.join(base_folder, folder_name)
        compressed_folder = os.path.join(folder_path, 'compressed')
        
        if not os.path.exists(compressed_folder):
            print(f"  No compressed folder found in {folder_name}, skipping...")
            continue
            
        print(f"\nOrganizing folder: {folder_name}")
        
        # Find all leaf_X folders in compressed directory
        leaf_folders = [d for d in os.listdir(compressed_folder) 
                       if os.path.isdir(os.path.join(compressed_folder, d)) and d.startswith('leaf_')]
        
        if not leaf_folders:
            print(f"  No leaf folders found in {folder_name}/compressed, skipping...")
            continue
        
        print(f"  Found {len(leaf_folders)} leaf folders to move")
        
        for leaf_folder_name in leaf_folders:
            old_leaf_path = os.path.join(compressed_folder, leaf_folder_name)
            new_leaf_path = os.path.join(folder_path, leaf_folder_name)
            
            try:
                # Move the leaf folder to root level
                if os.path.exists(new_leaf_path):
                    print(f"    Removing existing {leaf_folder_name} in root...")
                    shutil.rmtree(new_leaf_path)
                
                shutil.move(old_leaf_path, new_leaf_path)
                print(f"    Moved {leaf_folder_name} to root level")
                
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
                moved_files = []
                for file_name in files_to_move:
                    source_path = os.path.join(new_leaf_path, file_name)
                    dest_path = os.path.join(processed_folder, file_name)
                    
                    if os.path.exists(source_path):
                        shutil.move(source_path, dest_path)
                        moved_files.append(file_name)
                
                if moved_files:
                    print(f"    Moved to processed/: {', '.join(moved_files)}")
                else:
                    print(f"    No processed files found to move for {leaf_folder_name}")
                
            except Exception as e:
                print(f"    ✗ Error organizing {leaf_folder_name}: {str(e)}")
                continue
        
        print(f"  ✓ Completed organization for {folder_name}")

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
        print(f"    Not enough images found in {path} after filtering")
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

def main():
    """
    Main function - specify your base folder path here
    """
    # UPDATE THIS PATH TO YOUR FOLDER
    base_folder = '/mnt/e/projects/raw_datasets/lalweco/sugarbeets/nikon_camera/test_dataset'
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
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()