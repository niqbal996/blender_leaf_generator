import os
import rawpy
import imageio
from PIL import Image
from glob import glob

def resize_to_fit(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image to fit within target_size while maintaining aspect ratio.
    """
    # Calculate scaling factor to fit the image within target size
    scale = min(target_size / image.width, target_size / image.height)
    
    # Calculate new size
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    
    # Resize the image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized

def process_nef_folder(
    input_folder: str,
    target_resolution: int = 1024,
    output_subfolder: str = 'processed'
):
    """
    Process all NEF files in a folder, resize them and save as PNG.
    
    Args:
        input_folder: Path to folder containing NEF files
        target_resolution: Maximum dimension for resized images
        output_subfolder: Name of subfolder to create for processed images
    """
    
    # Create output directory
    output_folder = os.path.join(input_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all NEF files
    nef_files = glob(os.path.join(input_folder, '*.NEF'))
    nef_files.extend(glob(os.path.join(input_folder, '*.nef')))
    
    if not nef_files:
        print(f"No NEF files found in {input_folder}")
        return
    
    print(f"Found {len(nef_files)} NEF files to process")
    
    for nef_path in nef_files:
        try:
            print(f"Processing: {os.path.basename(nef_path)}")
            
            # Read NEF file using rawpy
            with rawpy.imread(nef_path) as raw:
                # Process the raw image
                rgb = raw.postprocess(no_auto_bright=False, 
                                      use_camera_wb=True,
                                      output_bps=8)

            # Convert numpy array to PIL Image
            image = Image.fromarray(rgb)
            original_size = image.size
            # print(f"  Original size: {original_size}")
            
            # Resize while maintaining aspect ratio
            resized_image = resize_to_fit(image, target_resolution)
            new_size = resized_image.size
            # print(f"  Resized to: {new_size}")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(nef_path))[0]
            output_path = os.path.join(output_folder, f'{base_name}.png')
            
            # Save as PNG
            resized_image.save(output_path, 'PNG')
            print(f"  Saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(nef_path)}: {str(e)}")
            continue

def process_nef_folder_square(
    input_folder: str,
    target_resolution: int = 1024,
    output_subfolder: str = 'processed',
    background_color: tuple = (255, 255, 255)  # White background
):
    """
    Process all NEF files in a folder, resize them to square format and save as PNG.
    
    Args:
        input_folder: Path to folder containing NEF files
        target_resolution: Square dimension for output images
        output_subfolder: Name of subfolder to create for processed images
        background_color: RGB tuple for background color when padding to square
    """
    
    # Create output directory
    output_folder = os.path.join(input_folder, output_subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all NEF files
    nef_files = glob(os.path.join(input_folder, '*.NEF'))
    nef_files.extend(glob(os.path.join(input_folder, '*.nef')))
    
    if not nef_files:
        print(f"No NEF files found in {input_folder}")
        return
    
    print(f"Found {len(nef_files)} NEF files to process")
    
    for nef_path in nef_files:
        try:
            print(f"Processing: {os.path.basename(nef_path)}")
            
            # Read NEF file using rawpy
            with rawpy.imread(nef_path) as raw:
                # Process the raw image
                rgb = raw.postprocess()
            
            # Convert numpy array to PIL Image
            image = Image.fromarray(rgb)
            original_size = image.size
            print(f"  Original size: {original_size}")
            
            # Resize while maintaining aspect ratio
            scale = min(target_resolution / image.width, target_resolution / image.height)
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create square canvas with specified background color
            canvas = Image.new('RGB', (target_resolution, target_resolution), background_color)
            
            # Calculate position to center the resized image
            x_offset = (target_resolution - new_width) // 2
            y_offset = (target_resolution - new_height) // 2
            
            # Paste the resized image onto the canvas
            canvas.paste(resized_image, (x_offset, y_offset))
            
            print(f"  Resized to: {target_resolution}x{target_resolution} (centered)")
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(nef_path))[0]
            output_path = os.path.join(output_folder, f'{base_name}.png')
            
            # Save as PNG
            canvas.save(output_path, 'PNG')
            print(f"  Saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(nef_path)}: {str(e)}")
            continue

def main():
    """
    Example usage
    """
    input_folder = '/mnt/e/projects/raw_datasets/lalweco/sugarbeets/nikon_camera/unprocessed/08-08-2025/weed_1'
    
    # Option 1: Resize maintaining original aspect ratio
    process_nef_folder(
        input_folder=input_folder,
        target_resolution=1024
    )
    
    # Option 2: Resize to square format with padding
    # process_nef_folder_square(
    #     input_folder=input_folder,
    #     target_resolution=1024,
    #     background_color=(255, 255, 255)  # White background
    # )

if __name__ == "__main__":
    main()