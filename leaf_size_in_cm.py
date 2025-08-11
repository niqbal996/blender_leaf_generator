import exifread
with open('/mnt/e/projects/raw_datasets/lalweco/sugarbeets/nikon_camera/test_dataset/weed_7/raw/DSC_0335.NEF','rb') as f:
    tags = exifread.process_file(f)
    f_mm = float(tags.get('EXIF FocalLength').printable)
    
z_mm = 280.0       # 28 cm -> 280 mm (your measured distance)
sensor_w_mm = 23.5
sensor_h_mm = 15.6
res_w = 6016
res_h = 4016

# pixel size (mm per pixel), average of width/height pixel pitch
px_mm = (sensor_w_mm/res_w + sensor_h_mm/res_h)/2.0

def pixels_per_cm(f_mm, z_mm, px_mm):
    return (f_mm * 10.0) / ((z_mm - f_mm) * px_mm)

ppcm = pixels_per_cm(f_mm, z_mm, px_mm)
print("pixels per cm:", ppcm)

# if you measured the leaf as N pixels in the image:
pixels_measured = 1000.0
leaf_cm = pixels_measured / ppcm
print("leaf size (cm):", leaf_cm)