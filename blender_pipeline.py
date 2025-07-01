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

# === Main Execution ===

image_dir = r"\\wsl.localhost\Ubuntu-20.04\home\niqbal\git\aa_blender\leaf_data\plant_1\leaf_assets"
mask_images = sorted(glob(os.path.join(image_dir, 'leaf_*_mask.png')))
diffuse_images = sorted(glob(os.path.join(image_dir, 'leaf_*_diffuse.png')))
normal_images = sorted(glob(os.path.join(image_dir, 'leaf_*_normal.png')))
leaf_scales = os.path.join(image_dir, 'leaf_scales.json')
# Load leaf physical sizes from JSON once before the loop
with open(leaf_scales, 'r') as f:
    leaf_physical_sizes = json.load(f)
bpy.context.scene.render.engine = 'CYCLES'

collection_name = "LeafMeshes"
if collection_name not in bpy.data.collections:
    leaf_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(leaf_collection)
else:
    leaf_collection = bpy.data.collections[collection_name]

x_offset = 0.0
spacing = 0.5  # space between leaves

for idx, mask_path in enumerate(mask_images):
    mesh_name = f"Leaf_{idx+1}_mesh"
    print(f"Creating mesh {mesh_name} from {mask_path}")

    obj = create_contour_based_mesh(
        image_path=mask_path,
        subdivisions=3,
        mesh_name=mesh_name,
        scale=2.0,
        collection=leaf_collection
    )
    # Ensure active object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Load and show image in UV/Image Editor
    img = bpy.data.images.load(diffuse_images[idx])
    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.spaces.active.image = img
    if obj:
        # Assign material
        if idx < len(diffuse_images) and idx < len(normal_images):
            material = create_leaf_material(
                name=f"LeafMaterial_{idx+1}",
                diffuse_path=diffuse_images[idx],
                normal_path=normal_images[idx],
                mask_path=mask_images[idx]
            )

            obj.data.materials.clear()
            obj.data.materials.append(material)

            # Generate UV map
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=66)
            bpy.ops.object.mode_set(mode='OBJECT')

    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    width = max(v.x for v in bbox) - min(v.x for v in bbox)
    height = max(v.y for v in bbox) - min(v.y for v in bbox)

    # --- SCALE TO REAL WORLD SIZE ---
    # Convert physical size from cm to meters
    leaf_data = leaf_physical_sizes.get(f'leaf_{idx+1}', {"height_cm": 10.0, "width_cm": 10.0})
    target_size_cm = max(leaf_data.get("height_cm", 10.0), leaf_data.get("width_cm", 10.0))
    target_size_m = target_size_cm / 100.0  # convert cm to meters

    # Get bounding box in world coordinates
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]

    # Calculate max dimension of the mesh bounding box
    dims = [max(v[i] for v in bbox) - min(v[i] for v in bbox) for i in range(3)]
    max_mesh_dim = max(dims)

    # Calculate scale factor and apply
    scale_factor = target_size_m / max_mesh_dim
    obj.scale = (scale_factor,) * 3

    bpy.context.view_layer.update()
    # Apply -90° X rotation
    obj.rotation_euler[0] = np.radians(-90)
    bpy.context.view_layer.update()

    # Evaluate object with all transforms
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    # Get actual world-space Z values of bounding box corners
    bbox = [eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box]
    z_min = min(v.z for v in bbox)

    # Shift object up so lowest point sits at z = 0
    obj.location.z -= z_min
    bpy.context.view_layer.update()