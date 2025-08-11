import bpy
import bmesh
from mathutils import Vector
import numpy as np
from PIL import Image
import os
import json 
from glob import glob

def create_leaf_material(name, diffuse_path, normal_path, mask_path):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.use_backface_culling = False
    mat.show_transparent_back = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    add_shader = nodes.new("ShaderNodeAddShader")

    tex_diffuse = nodes.new("ShaderNodeTexImage")
    tex_diffuse.image = bpy.data.images.load(diffuse_path)

    tex_normal = nodes.new("ShaderNodeTexImage")
    tex_normal.image = bpy.data.images.load(normal_path)
    tex_normal.image.colorspace_settings.name = 'Non-Color'

    tex_mask = nodes.new("ShaderNodeTexImage")
    tex_mask.image = bpy.data.images.load(mask_path)
    tex_mask.image.colorspace_settings.name = 'Non-Color'

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    normal_map = nodes.new("ShaderNodeNormalMap")

    # Mask normals
    normal_mask_mult = nodes.new("ShaderNodeMixRGB")
    normal_mask_mult.blend_type = 'MULTIPLY'
    normal_mask_mult.inputs[0].default_value = 1.0

    # Connect Diffuse
    links.new(tex_diffuse.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_diffuse.outputs['Color'], hue_sat.inputs['Color'])
    links.new(hue_sat.outputs['Color'], translucent.inputs['Color'])

    # Masked Normal Mapping
    links.new(tex_normal.outputs['Color'], normal_mask_mult.inputs[1])
    links.new(tex_mask.outputs['Color'], normal_mask_mult.inputs[2])
    links.new(normal_mask_mult.outputs['Color'], normal_map.inputs['Color'])

    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    links.new(normal_map.outputs['Normal'], translucent.inputs['Normal'])

    # Transparency
    links.new(tex_mask.outputs['Color'], bsdf.inputs['Alpha'])

    # Shader Combination
    links.new(bsdf.outputs['BSDF'], add_shader.inputs[0])
    links.new(translucent.outputs['BSDF'], add_shader.inputs[1])
    links.new(add_shader.outputs['Shader'], output.inputs['Surface'])

    return mat

def create_contour_based_mesh(image_path, subdivisions=10, mesh_name="ContourMesh", scale=1.0, collection=None):
    try:
        import cv2
    except ImportError:
        print("OpenCV not available. Install with: pip install opencv-python")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    height, width = img.shape

    aspect_ratio = width / height
    scale_x = scale if width > height else scale * aspect_ratio
    scale_y = scale / aspect_ratio if width > height else scale

    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in image")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    mesh = bpy.data.meshes.new(mesh_name)
    obj = bpy.data.objects.new(mesh_name, mesh)

    if collection:
        collection.objects.link(obj)
        if bpy.context.collection.objects.get(obj.name):
            bpy.context.collection.objects.unlink(obj)
    else:
        bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    contour_verts = []

    for point in simplified_contour:
        x_pixel, y_pixel = point[0]
        x_world = (x_pixel / width - 0.5) * scale_x
        y_world = (y_pixel / height - 0.5) * scale_y
        z_world = 0.0
        vert = bm.verts.new((x_world, y_world, z_world))
        contour_verts.append(vert)

    bm.verts.ensure_lookup_table()

    # Create a face from the outer contour
    if len(contour_verts) >= 3:
        try:
            face = bm.faces.new(contour_verts)
        except ValueError:
            print("Face creation failed; likely non-manifold or intersecting edges.")

    # Triangulate to get inner structure
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Subdivide all edges to add internal vertices
    bmesh.ops.subdivide_edges(
        bm,
        edges=bm.edges[:],
        cuts=3,  # Increase for more resolution
        use_grid_fill=True
    )

    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()

    obj.data.update()
    # Create UV map
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    # Use original image width and height for UV normalization
    img_height, img_width = img.shape

    uv_layer = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            vert_index = obj.data.loops[loop_index].vertex_index
            vert = obj.data.vertices[vert_index].co

            # Convert vertex x/y back to image pixel coordinates
            x_pixel = ((vert.x / scale_x) + 0.5) * img_width
            y_pixel = ((vert.y / scale_y) + 0.5) * img_height

            # Normalize to UV range and flip Y
            u = x_pixel / img_width
            v = 1.0 - (y_pixel / img_height)

            uv_layer[loop_index].uv = (u, v)
            
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    return obj

def load_leaf_size_data(asset_folder_path):
    """Load leaf size data from JSON file in asset folder"""
    json_files = glob(os.path.join(asset_folder_path, '*_leaves_data.json'))
    if json_files:
        with open(json_files[0], 'r') as f:
            return json.load(f)
    return None

# === Main Execution ===

# UPDATE THIS PATH TO YOUR DATASET
base_folder = r"F:\projects\raw_datasets\lalweco\sugarbeets\nikon_camera\test_dataset"

# Find all asset folders (containing leaf_ subfolders)
asset_folders = [d for d in os.listdir(base_folder) 
                if os.path.isdir(os.path.join(base_folder, d))]

bpy.context.scene.render.engine = 'CYCLES'

# Create main collection for all leaf meshes
collection_name = "LeafMeshes"
if collection_name not in bpy.data.collections:
    main_leaf_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(main_leaf_collection)
else:
    main_leaf_collection = bpy.data.collections[collection_name]

y_offset = 0.0
spacing = 0.5  # space between leaves along Y-axis
total_leaf_count = 0

print(f"Found {len(asset_folders)} asset folders to process")

for asset_idx, asset_folder_name in enumerate(sorted(asset_folders)):
    asset_folder_path = os.path.join(base_folder, asset_folder_name)
    
    # Find all leaf_ folders in this asset folder
    leaf_folders = [d for d in os.listdir(asset_folder_path) 
                   if os.path.isdir(os.path.join(asset_folder_path, d)) and d.startswith('leaf_')]
    
    if not leaf_folders:
        print(f"No leaf folders found in {asset_folder_name}, skipping...")
        continue
    
    print(f"\nProcessing asset: {asset_folder_name} with {len(leaf_folders)} leaves")
    
    # Create a collection for this asset's leaves
    asset_collection_name = f"Asset_{asset_idx+1}_{asset_folder_name}"
    if asset_collection_name not in bpy.data.collections:
        asset_collection = bpy.data.collections.new(asset_collection_name)
        main_leaf_collection.children.link(asset_collection)
    else:
        asset_collection = bpy.data.collections[asset_collection_name]
    
    # Load leaf size data for this asset (optional - for physical scaling)
    leaf_size_data = load_leaf_size_data(asset_folder_path)
    
    for leaf_idx, leaf_folder_name in enumerate(sorted(leaf_folders)):
        leaf_folder_path = os.path.join(asset_folder_path, leaf_folder_name)
        
        # Find the processed images in the leaf folder
        processed_folder = os.path.join(leaf_folder_path, 'processed')
        
        if os.path.exists(processed_folder):
            # Look for images in the processed folder
            mask_files = glob(os.path.join(processed_folder, '*_mask.png'))
            diffuse_files = glob(os.path.join(processed_folder, '*_diffuse.png'))
            normal_files = glob(os.path.join(processed_folder, '*_normal.png'))
        else:
            # Fallback: look directly in the leaf folder
            mask_files = glob(os.path.join(leaf_folder_path, '*_mask.png'))
            diffuse_files = glob(os.path.join(leaf_folder_path, '*_diffuse.png'))
            normal_files = glob(os.path.join(leaf_folder_path, '*_normal.png'))
        
        if not (mask_files and diffuse_files and normal_files):
            print(f"  Missing files for {leaf_folder_name}, skipping...")
            continue
        
        # Use the first found file (should only be one of each type)
        mask_path = mask_files[0]
        diffuse_path = diffuse_files[0]
        normal_path = normal_files[0]
        
        # Extract leaf number for naming
        leaf_number = leaf_folder_name.replace('leaf_', '')
        
        # Create unique mesh name: asset_folder_numeric_leaf_numeric
        mesh_name = f"{asset_folder_name}_{asset_idx+1}_{leaf_folder_name}_{leaf_number}"
        
        total_leaf_count += 1
        
        print(f"  Creating mesh {mesh_name}")
        print(f"    Mask: {os.path.basename(mask_path)}")
        print(f"    Diffuse: {os.path.basename(diffuse_path)}")
        print(f"    Normal: {os.path.basename(normal_path)}")
        
        # Create the mesh object
        obj = create_contour_based_mesh(
            image_path=mask_path,
            subdivisions=3,
            mesh_name=mesh_name,
            scale=2.0,
            collection=asset_collection
        )
        
        if obj:
            # Ensure active object
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Load and show image in UV/Image Editor
            try:
                img = bpy.data.images.load(diffuse_path)
                for area in bpy.context.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.spaces.active.image = img
            except:
                print(f"    Warning: Could not load image for UV editor")
            
            # Create and assign material
            material_name = f"Material_{asset_folder_name}_{leaf_folder_name}"
            material = create_leaf_material(
                name=material_name,
                diffuse_path=diffuse_path,
                normal_path=normal_path,
                mask_path=mask_path
            )

            obj.data.materials.clear()
            obj.data.materials.append(material)

            # Generate UV map
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=66)
            bpy.ops.object.mode_set(mode='OBJECT')

            # --- POSITIONING AND ROTATION ---
            # 1. Rotate 180° around X-axis
            obj.rotation_euler[0] = np.radians(180)
            bpy.context.view_layer.update()

            # 2. Get bounding box after rotation to calculate dimensions
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
            
            # Calculate dimensions
            y_min = min(v.y for v in bbox)
            y_max = max(v.y for v in bbox)
            z_min = min(v.z for v in bbox)
            
            leaf_height = y_max - y_min  # Height along Y-axis
            
            # 3. Apply physical scaling if size data is available
#            if leaf_size_data and 'leaves' in leaf_size_data and leaf_folder_name in leaf_size_data['leaves']:
#                leaf_data = leaf_size_data['leaves'][leaf_folder_name]
#                physical_height_cm = leaf_data['dimensions']['height_cm']
#                
#                # Convert cm to Blender units (assuming 1 Blender unit = 1 meter)
#                physical_height_blender = physical_height_cm / 100.0
#                
#                # Calculate scale factor
#                scale_factor = physical_height_blender / leaf_height
#                obj.scale = (scale_factor, scale_factor, scale_factor)
#                bpy.context.view_layer.update()
#                
#                # Recalculate dimensions after scaling
#                eval_obj = obj.evaluated_get(depsgraph)
#                bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
#                y_min = min(v.y for v in bbox)
#                y_max = max(v.y for v in bbox)
#                z_min = min(v.z for v in bbox)
#                leaf_height = y_max - y_min
#                
#                print(f"    Scaled to physical size: {physical_height_cm}cm ({physical_height_blender:.3f}m)")
            
            # 4. Position the object
            obj.location.x = 0.0  # Keep at X=0
            obj.location.y = y_offset + (leaf_height / 2)  # Center leaf at current Y offset
            obj.location.z = -z_min  # Move so lowest point is at Z=0
            
            bpy.context.view_layer.update()
            
            # 5. Update Y offset for next leaf
            y_offset += leaf_height + spacing
            
            print(f"    Positioned at Y={obj.location.y:.3f}, height: {leaf_height:.3f}")
        
        else:
            print(f"    Failed to create mesh for {leaf_folder_name}")

print(f"\nCompleted processing!")
print(f"Total assets processed: {len(asset_folders)}")
print(f"Total leaf meshes created: {total_leaf_count}")
print(f"Collections created under '{collection_name}'")