"""
Convex-decompose each object OBJ into per-hull pieces, export under convex/,
and print the XML geom block to paste into push_scene.xml.
"""
import os
import numpy as np
import trimesh
from vhacdx import compute_vhacd

OBJECTS = ["game_controller", "hammer", "painters_tape", "red_bowl", "spring_clamp"]
MAX_HULLS = 8   # adjust per object if needed

# 10 visually distinct colours (RGBA)
PALETTE = [
    "0.95 0.25 0.25 1",  # red
    "0.25 0.55 0.95 1",  # blue
    "0.25 0.85 0.35 1",  # green
    "0.95 0.75 0.10 1",  # yellow
    "0.80 0.30 0.90 1",  # purple
    "0.95 0.50 0.10 1",  # orange
    "0.15 0.85 0.85 1",  # cyan
    "0.90 0.40 0.65 1",  # pink
    "0.50 0.85 0.20 1",  # lime
    "0.60 0.40 0.20 1",  # brown
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "convex")
os.makedirs(OUT_DIR, exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def export_hull(verts, faces, path):
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.export(path)


xml_lines = []

for obj_name in OBJECTS:
    src = os.path.join(SCRIPT_DIR, f"{obj_name}.obj")
    mesh = trimesh.load(src, force="mesh")

    hulls = compute_vhacd(
        mesh.vertices.astype(np.float64),
        mesh.faces.astype(np.uint32),
        maxConvexHulls=MAX_HULLS,
    )
    print(f"{obj_name}: {len(hulls)} hulls")

    asset_lines = []
    geom_lines  = []

    for i, (v, f) in enumerate(hulls):
        piece_name = f"{obj_name}_cvx{i:02d}"
        obj_path   = os.path.join(OUT_DIR, f"{piece_name}.obj")
        export_hull(v, f, obj_path)

        rel_path = f"../objects/convex/{piece_name}.obj"
        color    = PALETTE[i % len(PALETTE)]

        asset_lines.append(f'    <mesh name="{piece_name}" file="{rel_path}"/>')
        geom_lines.append(
            f'      <geom type="mesh" mesh="{piece_name}" '
            f'contype="1" conaffinity="1" friction="0.3 0.003 0.0001" rgba="{color}"/>'
        )

    xml_lines.append(f"\n    <!-- {obj_name} -->")
    xml_lines.extend(asset_lines)
    xml_lines.append(f'    <!-- body for {obj_name} -->')
    xml_lines.append(f'    <body name="{obj_name}" pos="0 0 0"><freejoint/>')
    xml_lines.extend(geom_lines)
    xml_lines.append('    </body>')

print("\n\n=== ASSET BLOCK (paste inside <asset>) ===")
for l in xml_lines:
    if "<mesh" in l:
        print(l)

print("\n=== BODY BLOCK (replace current object bodies) ===")
in_body = False
for l in xml_lines:
    if "body for" in l or "<body" in l or "<geom" in l or "</body>" in l or ("<!--" in l and "body" not in l):
        print(l)
