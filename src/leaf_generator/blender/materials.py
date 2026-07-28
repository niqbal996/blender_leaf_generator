"""Build leaf shader materials from a side's PBR map set.

Each side (oberseite / unterseite) gets its own Principled BSDF + Translucent
combo (so backlighting still reads correctly, as in the original single-sided
material). When both sides are available, the two side-shaders are mixed by
`Geometry > Backfacing`, so a single mesh shows the correct texture set
depending on which face the camera sees, without needing two separate
objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import bpy

REQUIRED_FOR_SHADING = ("albedo", "mask")


def build_leaf_material(
    name: str,
    front_maps: Dict[str, Path],
    back_maps: Optional[Dict[str, Path]] = None,
) -> bpy.types.Material:
    """front_maps/back_maps: {"albedo": Path, "normal": Path, "height": Path,
    "roughness": Path, "mask": Path} -- normal/height/roughness are optional,
    missing ones are simply skipped.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.use_backface_culling = False
    mat.show_transparent_back = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)

    front_shader = _build_side_shader(nodes, links, front_maps, suffix="front", y=250)

    if back_maps:
        back_shader = _build_side_shader(nodes, links, back_maps, suffix="back", y=-250)
        geometry = nodes.new("ShaderNodeNewGeometry")
        geometry.location = (0, -600)
        mix_shader = nodes.new("ShaderNodeMixShader")
        mix_shader.location = (400, 0)
        links.new(front_shader, mix_shader.inputs[1])
        links.new(back_shader, mix_shader.inputs[2])
        links.new(geometry.outputs["Backfacing"], mix_shader.inputs["Fac"])
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    else:
        links.new(front_shader, output.inputs["Surface"])

    return mat


def _load_image(path: Path, non_color: bool = False) -> bpy.types.Image:
    img = bpy.data.images.load(str(path))
    if non_color:
        img.colorspace_settings.name = 'Non-Color'
    return img


def _build_side_shader(nodes, links, maps: Dict[str, Path], suffix: str, y: float):
    x0 = -800

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = f"bsdf_{suffix}"
    bsdf.location = (0, y)
    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    translucent.location = (0, y - 150)
    add_shader = nodes.new("ShaderNodeAddShader")
    add_shader.location = (200, y - 75)
    links.new(bsdf.outputs["BSDF"], add_shader.inputs[0])
    links.new(translucent.outputs["BSDF"], add_shader.inputs[1])

    albedo_path = maps.get("albedo")
    if albedo_path is not None:
        tex_albedo = nodes.new("ShaderNodeTexImage")
        tex_albedo.name = f"albedo_{suffix}"
        tex_albedo.location = (x0, y + 200)
        tex_albedo.image = _load_image(albedo_path)

        hue_sat = nodes.new("ShaderNodeHueSaturation")
        hue_sat.location = (x0 + 300, y + 100)

        links.new(tex_albedo.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(tex_albedo.outputs["Color"], hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], translucent.inputs["Color"])
    else:
        print(f"[leaf_generator] Warning: no albedo map for side '{suffix}'; using flat gray base color.")
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)

    mask_path = maps.get("mask")
    tex_mask = None
    if mask_path is not None:
        tex_mask = nodes.new("ShaderNodeTexImage")
        tex_mask.name = f"mask_{suffix}"
        tex_mask.location = (x0, y - 200)
        tex_mask.image = _load_image(mask_path, non_color=True)
        links.new(tex_mask.outputs["Color"], bsdf.inputs["Alpha"])
    else:
        print(f"[leaf_generator] Warning: no mask map for side '{suffix}'; material will be fully opaque.")

    normal_output_socket = None
    normal_path = maps.get("normal")
    if normal_path is not None:
        tex_normal = nodes.new("ShaderNodeTexImage")
        tex_normal.name = f"normal_{suffix}"
        tex_normal.location = (x0, y)
        tex_normal.image = _load_image(normal_path, non_color=True)

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (x0 + 600, y)

        if tex_mask is not None:
            normal_mask_mult = nodes.new("ShaderNodeMixRGB")
            normal_mask_mult.name = f"normal_mask_mult_{suffix}"
            normal_mask_mult.location = (x0 + 300, y)
            normal_mask_mult.blend_type = 'MULTIPLY'
            normal_mask_mult.inputs[0].default_value = 1.0
            links.new(tex_normal.outputs["Color"], normal_mask_mult.inputs[1])
            links.new(tex_mask.outputs["Color"], normal_mask_mult.inputs[2])
            links.new(normal_mask_mult.outputs["Color"], normal_map.inputs["Color"])
        else:
            links.new(tex_normal.outputs["Color"], normal_map.inputs["Color"])

        normal_output_socket = normal_map.outputs["Normal"]

    height_path = maps.get("height")
    if height_path is not None:
        tex_height = nodes.new("ShaderNodeTexImage")
        tex_height.name = f"height_{suffix}"
        tex_height.location = (x0, y - 400)
        tex_height.image = _load_image(height_path, non_color=True)

        bump = nodes.new("ShaderNodeBump")
        bump.location = (x0 + 600, y - 400)
        bump.inputs["Strength"].default_value = 0.3
        bump.inputs["Distance"].default_value = 0.05
        links.new(tex_height.outputs["Color"], bump.inputs["Height"])
        if normal_output_socket is not None:
            links.new(normal_output_socket, bump.inputs["Normal"])
        normal_output_socket = bump.outputs["Normal"]

    roughness_path = maps.get("roughness")
    if roughness_path is not None:
        tex_roughness = nodes.new("ShaderNodeTexImage")
        tex_roughness.name = f"roughness_{suffix}"
        tex_roughness.location = (x0, y - 600)
        tex_roughness.image = _load_image(roughness_path, non_color=True)
        links.new(tex_roughness.outputs["Color"], bsdf.inputs["Roughness"])

    if normal_output_socket is not None:
        links.new(normal_output_socket, bsdf.inputs["Normal"])
        links.new(normal_output_socket, translucent.inputs["Normal"])

    return add_shader.outputs["Shader"]
