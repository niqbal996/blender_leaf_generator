import os
import json
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
from glob import glob
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from torchvision import transforms
from PIL import Image
import tempfile
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_binary_mask(image_with_alpha: Image.Image, save_path: str):
    """
    Save a binary mask (white foreground, black background) from the alpha channel of an RGBA image.
    """
    alpha = np.array(image_with_alpha.split()[-1])
    binary_mask = (alpha > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary_mask, mode='L')
    mask_img.save(save_path)

def preprocess_image(input, mask_model='birefnet', resolution=1024):
    # if has alpha channel, use it directly
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, -1]
        if not np.all(alpha == 255):
            has_alpha = True
    
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        
        # Load BiRefNet model if not already loaded
        
        # Get mask using BiRefNet
        mask = get_birefnet_mask(input, mask_model)
        
        # Convert input to RGBA and apply mask
        input_rgba = input.convert('RGBA')
        input_array = np.array(input_rgba)
        input_array[:, :, 3] = mask * 255  # Apply mask to alpha channel
        output = Image.fromarray(input_array)

    # Process the output image
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    
    # Find bounding box of non-transparent pixels
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:  # Handle case where no foreground is detected
        return input.convert('RGB')
    
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    
    # Calculate and apply crop bbox
    bbox = (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2)
    )
    
    # Ensure bbox is within image bounds
    bbox = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(output.width, bbox[2]),
        min(output.height, bbox[3])
    )
    
    output = output.crop(bbox)
    
    # Add padding to maintain aspect ratio
    width, height = output.size
    if width > height:
        new_height = width
        padding = (width - height) // 2
        padded_output = Image.new('RGBA', (width, new_height), (0, 0, 0, 0))
        padded_output.paste(output, (0, padding))
    else:
        new_width = height
        padding = (height - width) // 2
        padded_output = Image.new('RGBA', (new_width, height), (0, 0, 0, 0))
        padded_output.paste(output, (padding, 0))
    
    # Resize padded image to target size
    padded_output = padded_output.resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    # Final processing
    output = np.array(padded_output).astype(np.float32) / 255
    output = np.dstack((
        output[:, :, :3] * output[:, :, 3:4],  # RGB channels premultiplied by alpha
        output[:, :, 3]                         # Original alpha channel
    ))
    output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
    
    return output

def lazy_load_birefnet():
    """Lazy loading of the BiRefNet model"""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForImageSegmentation
    birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    ).to(DEVICE)
    return birefnet_model.eval()
    
def get_birefnet_mask(image: Image.Image, model) -> np.ndarray:
    """Get object mask using BiRefNet"""
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    mask_np = np.array(mask)

    # Convert to binary mask: foreground (object) as white (255), background as black (0)
    binary_mask = (mask_np > 128).astype(np.uint8) * 255
    mask_np = binary_mask

    return (mask_np > 128).astype(np.uint8)

def apply_leaf_orientation(img, mask):
    """Rotate and flip img and mask so the leaf is upright and stem is at the bottom."""
    from skimage.measure import regionprops, label
    # 1. Align major axis vertically
    mask_bool = mask > 0
    lbl = label(mask_bool)
    props = regionprops(lbl)
    if not props:
        return img, mask
    angle = np.rad2deg(props[0].orientation)
    rotate_angle = 90 - angle
    img = np.array(Image.fromarray(img).rotate(rotate_angle, expand=True))
    mask = np.array(Image.fromarray(mask).rotate(rotate_angle, expand=True))

    # 2. If major axis is vertical but broad side is on left/right, rotate 90°
    h, w = mask.shape
    left_sum = np.count_nonzero(mask[:, :w//2])
    right_sum = np.count_nonzero(mask[:, w//2:])
    if right_sum > left_sum:
        img = np.array(Image.fromarray(img).rotate(90, expand=True))
        mask = np.array(Image.fromarray(mask).rotate(90, expand=True))
    elif left_sum > right_sum:
        img = np.array(Image.fromarray(img).rotate(-90, expand=True))
        mask = np.array(Image.fromarray(mask).rotate(-90, expand=True))

    # 3. Ensure stem is at the bottom (narrowest part at bottom)
    h, w = mask.shape
    vertical_profile = mask.sum(axis=1)
    top_sum = vertical_profile[:h//10].sum()
    bottom_sum = vertical_profile[-h//10:].sum()
    if top_sum < bottom_sum:
        img = np.array(Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM))
        mask = np.array(Image.fromarray(mask).transpose(Image.FLIP_TOP_BOTTOM))
    return img, mask

def align_leaf_orientation(crop_img, crop_mask, crop_normal):
    # 1. Align major axis vertically
    mask_bool = crop_mask > 0
    lbl = label(mask_bool)
    props = regionprops(lbl[0])
    if not props:
        return crop_img, crop_mask  # fallback
    angle = np.rad2deg(props[0].orientation)
    rotate_angle = 90 - angle
    crop_img = np.array(Image.fromarray(crop_img).rotate(rotate_angle, expand=True))
    crop_mask = np.array(Image.fromarray(crop_mask).rotate(rotate_angle, expand=True))

    # 2. If major axis is vertical but broad side is on left/right, rotate 90°
    h, w = crop_mask.shape
    left_sum = np.count_nonzero(crop_mask[:, :w//2])
    right_sum = np.count_nonzero(crop_mask[:, w//2:])
    if right_sum > left_sum:
        crop_img = np.array(Image.fromarray(crop_img).rotate(90, expand=True))
        crop_mask = np.array(Image.fromarray(crop_mask).rotate(90, expand=True))
        crop_normal = np.array(Image.fromarray(crop_normal).rotate(90, expand=True))
    elif left_sum > right_sum:
        crop_img = np.array(Image.fromarray(crop_img).rotate(-90, expand=True))
        crop_mask = np.array(Image.fromarray(crop_mask).rotate(-90, expand=True))
        crop_normal = np.array(Image.fromarray(crop_normal).rotate(-90, expand=True))

    # 3. Ensure stem is at the bottom (narrowest part at bottom)
    h, w = crop_mask.shape
    vertical_profile = crop_mask.sum(axis=1)
    top_sum = vertical_profile[:h//10].sum()
    bottom_sum = vertical_profile[-h//10:].sum()
    # Flip if stem is at the top (top_sum < bottom_sum)
    if top_sum < bottom_sum:
        crop_img = np.array(Image.fromarray(crop_img).transpose(Image.FLIP_TOP_BOTTOM))
        crop_mask = np.array(Image.fromarray(crop_mask).transpose(Image.FLIP_TOP_BOTTOM))
        crop_normal = np.array(Image.fromarray(crop_normal).transpose(Image.FLIP_TOP_BOTTOM))
        
    return crop_img, crop_mask, crop_normal

def get_long_axis_angle(mask):
    """Returns angle (in degrees) to rotate so the longest axis is vertical."""
    from skimage.measure import regionprops, label
    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return 0
    orientation = props[0].orientation  # in radians, CCW from horizontal axis
    angle = np.rad2deg(orientation)
    # If angle is negative, rotate CCW; if positive, rotate CW
    # We want the major axis vertical, so rotate by (90 - angle)
    return 90 - angle

def find_stem_side(mask):
    """
    Compares the number of nonzero values between left and right half of the mask.
    Returns:
        'left'  - if left half has more (rotate 90 deg CW)
        'right' - if right half has more (rotate 90 deg CCW)
        'none'  - if equal or ambiguous
    """
    w = mask.shape[1]
    left_sum = np.count_nonzero(mask[:, :w//2])
    right_sum = np.count_nonzero(mask[:, w//2:])
    if right_sum > left_sum:
        return 'right'
    elif left_sum > right_sum:
        return 'left'
    else:
        return 'none'

def extract_leaves_and_masks(
    input_image: Image.Image,
    mask_model,
    resolution=1024,
    out_dir='./output',
    normal_map_path: str = None,
    pixels_per_cm: float = 60.14, # 0.01663 sampling distance cm/pixels i.e. 60.14 pixels per cm
):
    import skimage.transform
    os.makedirs(out_dir, exist_ok=True)
    leaf_physical_sizes = {}
    mask = get_birefnet_mask(input_image.convert('RGB'), mask_model)
    labeled, num_features = label(mask)
    slices = find_objects(labeled)
    input_rgba = input_image.convert('RGBA')
    np_img = np.array(input_rgba)

    # --- Load and resize normal map if provided ---
    normal_np = None
    if normal_map_path is not None:
        normal_img = Image.open(normal_map_path).convert('RGBA')
        normal_np = np.array(normal_img)

    # --- Find the largest leaf's bounding box size ---
    max_h, max_w = 0, 0
    for slc in slices:
        if slc is None:
            continue
        leaf_mask = (labeled[slc] == (np.max(labeled[slc]))).astype(np.uint8)
        if leaf_mask.sum() < 100:
            continue
        h, w = leaf_mask.shape
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w
    max_dim = max(max_h, max_w)

    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        leaf_mask = (labeled[slc] == (idx + 1)).astype(np.uint8) * 255
        if leaf_mask.sum() < 100:
            continue
        crop_img = np_img[slc].copy()
        crop_mask = leaf_mask
        crop_normal = normal_np[slc].copy() if normal_np is not None else None

        # Set alpha channel to mask
        crop_img[..., 3] = crop_mask

        # --- Rotate so stem is at the bottom (mask drives all transforms) ---
        # If normal map is present, pass it to align_leaf_orientation, but only mask is used for logic
        crop_img, crop_mask, crop_normal = align_leaf_orientation(crop_img, crop_mask, crop_normal)

        # --- Maintain relative size on fixed-size canvas ---
        leaf_h, leaf_w = crop_mask.shape
        max_physical_dim_cm = max(leaf_h, leaf_w) / pixels_per_cm
        scale = min(resolution / max_dim, 1.0)
        new_size = (int(leaf_w * scale), int(leaf_h * scale))
        leaf_img = Image.fromarray(crop_img).resize(new_size, Image.Resampling.LANCZOS)
        leaf_mask_img = Image.fromarray(crop_mask).resize(new_size, Image.Resampling.NEAREST)

        # Center on transparent canvas
        canvas = Image.new('RGBA', (resolution, resolution), (0, 0, 0, 0))
        mask_canvas = Image.new('L', (resolution, resolution), 0)
        offset = ((resolution - new_size[0]) // 2, (resolution - new_size[1]) // 2)
        canvas.paste(leaf_img, offset, leaf_mask_img)
        mask_canvas.paste(leaf_mask_img, offset)
        # Save RGB and mask
        # Save scale factor to text
        leaf_physical_sizes[f'leaf_{idx+1}'] = {
            "height_cm": leaf_h / pixels_per_cm,
            "width_cm": leaf_w / pixels_per_cm
        }
        canvas.save(os.path.join(out_dir, f'leaf_{idx+1}_diffuse.png'))
        mask_canvas.save(os.path.join(out_dir, f'leaf_{idx+1}_mask.png'))

        # --- Save normal map for this leaf if available ---
        if crop_normal is not None:
            # Apply the same transformation as mask (already done above)
            crop_normal_img = Image.fromarray(crop_normal).resize(new_size, Image.Resampling.LANCZOS)
            normal_canvas = Image.new('RGB', (resolution, resolution), (0, 0, 0))
            normal_canvas.paste(crop_normal_img, offset)
            normal_canvas.save(os.path.join(out_dir, f'leaf_{idx+1}_normal.png'))

    # --- Save the complete image with background removed ---
    full_mask = (mask * 255).astype(np.uint8)
    full_rgba = np.array(input_image.convert('RGBA'))
    with open(os.path.join(out_dir, f'leaf_scales.json'), 'w') as f:
        json.dump(leaf_physical_sizes, f, indent=2)
    full_rgba[..., 3] = full_mask
    full_img = Image.fromarray(full_rgba, mode='RGBA')
    full_img.save(os.path.join(out_dir, 'all_leaves_no_bg.png'))

    # --- Save the binary mask of the entire image ---
    mask_img = Image.fromarray(full_mask, mode='L')
    mask_img.save(os.path.join(out_dir, 'all_leaves_mask.png'))

def resize_and_pad_to_match(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize img to fit inside target_size, keeping aspect ratio, and pad with black."""
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    if img_ratio > target_ratio:
        # Fit width
        new_w = target_size[0]
        new_h = int(new_w / img_ratio)
    else:
        # Fit height
        new_h = target_size[1]
        new_w = int(new_h * img_ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2)
    new_img.paste(img_resized, offset)
    return new_img

def leaves():
    image = Image.open('/home/niqbal/git/aa_blender/leaf_data/plant_1/blender_assets/IMG_9728_diffused.JPEG')
    image_paths = glob('./image_pool/*.JPEG')  # Adjust the path to your image pool
    image_pool = [Image.open(p) for p in image_paths]
    model = lazy_load_birefnet()
    extract_leaves_and_masks(image, 
                             mask_model=model, 
                             resolution=1024, 
                             out_dir='/home/niqbal/git/aa_blender/leaf_data/plant_1/leaf_assets/',
                             normal_map_path='/home/niqbal/git/aa_blender/leaf_data/plant_1/blender_assets/IMG_9728_normal.png'
                             )

def main():
    image = Image.open('image_pool/IMG_9728_diffused.JPEG')  # Replace with your input image path
    model = lazy_load_birefnet()
    mask_image = preprocess_image(image, mask_model=model, resolution=1024)
    # mask_image.save('/mnt/e/projects/raw_datasets/lalweco/sugarbeets/leaves/IMG_9728_mask.png')  # Save the output image
    save_binary_mask(
        mask_image,
        '/mnt/e/projects/raw_datasets/lalweco/sugarbeets/leaves/IMG_9728_mask.png'
    )
if __name__ == "__main__":
    # main()
    leaves()