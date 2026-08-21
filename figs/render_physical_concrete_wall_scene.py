#!/usr/bin/env python3
"""Render Figure 1's physical evidence plate from a single world model.

This file deliberately uses no image sprites or texture photographs.  The
concrete, mortar, bricks, basketball panels, contact patches, trajectories,
lighting and shadows are all procedural.  Every visible state shares one
orthographic camera and one world coordinate system (metres).

The two opaque basketballs are the measured maximum-compression states at the
floor and the wall.  Semi-transparent balls are equal-time samples.  They are
a time-composite, so only the two opaque event states contribute to the common
shadow map; every sample is nevertheless shaded by the same light.

Run from the paper root:
    python figs/render_physical_concrete_wall_scene.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image


OUT_W, OUT_H = 2078, 1000
SUPERSAMPLE = 2
RENDER_W, RENDER_H = OUT_W * SUPERSAMPLE, OUT_H * SUPERSAMPLE
SHADOW_SIZE = 4096

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "assets" / "physical_concrete_wall_scene_v4.png"


# ---------------------------------------------------------------------------
# Physical model (SI units)
# ---------------------------------------------------------------------------
# Calibrated adult basketball: 24 cm nominal diameter.  SCENE_SCALE converts
# the original unit composition to metres without changing a single projected
# pixel.  Dynamic similarity under fixed terrestrial gravity requires
# v -> sqrt(s) v and t -> sqrt(s) t.
G = 9.81
REFERENCE_R = 0.45
R = 0.12
SCENE_SCALE = R / REFERENCE_R
TIME_SCALE = math.sqrt(SCENE_SCALE)
INV_SCENE_SCALE = 1.0 / SCENE_SCALE
DELTA_G = 0.05 * R
DELTA_W = 0.035 * R
E_G = 0.78
E_W = 0.72
V_X = 5.00 * TIME_SCALE
V_Y_OUT = 6.30 * TIME_SCALE
V_Y_IN = -V_Y_OUT / E_G
X_GROUND = -1.35 * SCENE_SCALE
X_WALL = 3.00 * SCENE_SCALE  # collision plane: left face of the fixed masonry wall
Z_MOTION = 0.12 * SCENE_SCALE
X_WALL_CONTACT = X_WALL - R + DELTA_W
T_WALL = (X_WALL_CONTACT - X_GROUND) / V_X
Y_WALL_CONTACT = R + V_Y_OUT * T_WALL - 0.5 * G * T_WALL**2
V_Y_WALL = V_Y_OUT - G * T_WALL

FLOOR_PATCH_RADIUS = math.sqrt(2.0 * R * DELTA_G - DELTA_G**2)
WALL_PATCH_RADIUS = math.sqrt(2.0 * R * DELTA_W - DELTA_W**2)

# Real single-wythe masonry dimensions.
BRICK_LENGTH = 0.215
BRICK_HEIGHT = 0.065
BRICK_DEPTH = 0.1025
MORTAR_GAP = 0.009
MORTAR_RECESS = 0.004
BRICK_BEVEL = 0.0015
WALL_HEIGHT = 3.90 * SCENE_SCALE
WALL_Z_MIN = -0.45 * SCENE_SCALE
WALL_Z_MAX = 0.95 * SCENE_SCALE
WALL_THICKNESS = 0.40 * SCENE_SCALE


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1.0e-12 else v


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    out = np.eye(4, dtype=np.float32)
    out[0, :3] = s
    out[1, :3] = u
    out[2, :3] = -f
    out[0, 3] = -np.dot(s, eye)
    out[1, 3] = -np.dot(u, eye)
    out[2, 3] = np.dot(f, eye)
    return out


def ortho(left: float, right: float, bottom: float, top: float,
          near: float, far: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = 2.0 / (right - left)
    out[1, 1] = 2.0 / (top - bottom)
    out[2, 2] = -2.0 / (far - near)
    out[0, 3] = -(right + left) / (right - left)
    out[1, 3] = -(top + bottom) / (top - bottom)
    out[2, 3] = -(far + near) / (far - near)
    return out


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return out


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return out


def translation(x: float, y: float, z: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, 3] = (x, y, z)
    return out


def gl_bytes(m: np.ndarray) -> bytes:
    return np.asarray(m, dtype="f4").T.tobytes()


def project_pixel(point, view_proj: np.ndarray) -> tuple[int, int]:
    p = np.array([*point, 1.0], dtype=np.float64)
    clip = view_proj.astype(np.float64) @ p
    ndc = clip[:3] / clip[3]
    px = int(round((0.5 * ndc[0] + 0.5) * (OUT_W - 1)))
    py = int(round((0.5 - 0.5 * ndc[1]) * (OUT_H - 1)))
    return px, py


# ---------------------------------------------------------------------------
# Mesh construction.  Vertex layout: position, normal, material-coordinate,
# per-vertex colour (12 float32 values).
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    vbo: moderngl.Buffer
    ibo: moderngl.Buffer
    vao: moderngl.VertexArray
    depth_vao: moderngl.VertexArray
    count: int


def pack_vertex(pos, normal, local, colour) -> list[float]:
    return [*pos, *normal, *local, *colour]


def append_box(vertices: list[float], indices: list[int], lo, hi, colour) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [
        ((-1, 0, 0), [(x0, y0, z1), (x0, y0, z0), (x0, y1, z0), (x0, y1, z1)]),
        ((1, 0, 0), [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)]),
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 1, 0), [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)]),
        ((0, 0, -1), [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
    ]
    for normal, corners in faces:
        base = len(vertices) // 12
        for p in corners:
            vertices.extend(pack_vertex(p, normal, p, colour))
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])


def append_facet(vertices: list[float], indices: list[int], points,
                 normal, colour) -> None:
    """Append a planar convex triangle/quad with verified outward winding."""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    n = normalize(np.asarray(normal, dtype=np.float64))
    if np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), n) < 0.0:
        pts.reverse()
    base = len(vertices) // 12
    for p in pts:
        vertices.extend(pack_vertex(p, n, p, colour))
    if len(pts) == 3:
        indices.extend([base, base + 1, base + 2])
    elif len(pts) == 4:
        indices.extend([base, base + 1, base + 2,
                        base, base + 2, base + 3])
    else:
        raise ValueError("facet must be a triangle or quad")


def append_bevelled_box(vertices: list[float], indices: list[int], lo, hi,
                        colour, bevel: float) -> None:
    """Closed solid chamfered cuboid with flat 1--2 mm masonry bevels."""
    x0, y0, z0 = map(float, lo)
    x1, y1, z1 = map(float, hi)
    b = min(float(bevel), 0.24 * min(x1 - x0, y1 - y0, z1 - z0))
    if b <= 1.0e-8:
        append_box(vertices, indices, lo, hi, colour)
        return

    # Six principal faces, inset by b from every adjoining edge.
    append_facet(vertices, indices,
                 [(x0, y0+b, z0+b), (x0, y1-b, z0+b),
                  (x0, y1-b, z1-b), (x0, y0+b, z1-b)],
                 (-1, 0, 0), colour)
    append_facet(vertices, indices,
                 [(x1, y0+b, z0+b), (x1, y0+b, z1-b),
                  (x1, y1-b, z1-b), (x1, y1-b, z0+b)],
                 (1, 0, 0), colour)
    append_facet(vertices, indices,
                 [(x0+b, y0, z0+b), (x0+b, y0, z1-b),
                  (x1-b, y0, z1-b), (x1-b, y0, z0+b)],
                 (0, -1, 0), colour)
    append_facet(vertices, indices,
                 [(x0+b, y1, z0+b), (x1-b, y1, z0+b),
                  (x1-b, y1, z1-b), (x0+b, y1, z1-b)],
                 (0, 1, 0), colour)
    append_facet(vertices, indices,
                 [(x0+b, y0+b, z0), (x1-b, y0+b, z0),
                  (x1-b, y1-b, z0), (x0+b, y1-b, z0)],
                 (0, 0, -1), colour)
    append_facet(vertices, indices,
                 [(x0+b, y0+b, z1), (x0+b, y1-b, z1),
                  (x1-b, y1-b, z1), (x1-b, y0+b, z1)],
                 (0, 0, 1), colour)

    # Twelve bevel strips, grouped by their edge direction.
    for sy in (-1, 1):  # edges parallel X
        yf = y0 if sy < 0 else y1
        for sz in (-1, 1):
            zf = z0 if sz < 0 else z1
            append_facet(vertices, indices,
                         [(x0+b, yf, zf-sz*b), (x1-b, yf, zf-sz*b),
                          (x1-b, yf-sy*b, zf), (x0+b, yf-sy*b, zf)],
                         (0, sy, sz), colour)
    for sx in (-1, 1):  # edges parallel Y
        xf = x0 if sx < 0 else x1
        for sz in (-1, 1):
            zf = z0 if sz < 0 else z1
            append_facet(vertices, indices,
                         [(xf, y0+b, zf-sz*b), (xf, y1-b, zf-sz*b),
                          (xf-sx*b, y1-b, zf), (xf-sx*b, y0+b, zf)],
                         (sx, 0, sz), colour)
    for sx in (-1, 1):  # edges parallel Z
        xf = x0 if sx < 0 else x1
        for sy in (-1, 1):
            yf = y0 if sy < 0 else y1
            append_facet(vertices, indices,
                         [(xf, yf-sy*b, z0+b), (xf, yf-sy*b, z1-b),
                          (xf-sx*b, yf, z1-b), (xf-sx*b, yf, z0+b)],
                         (sx, sy, 0), colour)

    # Eight triangular corner facets close the solid.
    for sx in (-1, 1):
        xf = x0 if sx < 0 else x1
        for sy in (-1, 1):
            yf = y0 if sy < 0 else y1
            for sz in (-1, 1):
                zf = z0 if sz < 0 else z1
                append_facet(vertices, indices,
                             [(xf, yf-sy*b, zf-sz*b),
                              (xf-sx*b, yf, zf-sz*b),
                              (xf-sx*b, yf-sy*b, zf)],
                             (sx, sy, sz), colour)


def build_bevelled_boxes(boxes, bevel: float):
    vertices: list[float] = []
    indices: list[int] = []
    for seed, (lo, hi, colour) in enumerate(boxes):
        first_vertex = len(vertices) // 12
        append_bevelled_box(vertices, indices, lo, hi, colour, bevel)
        last_vertex = len(vertices) // 12
        centre = 0.5 * (np.asarray(lo, dtype=np.float64)
                        + np.asarray(hi, dtype=np.float64))
        size = np.maximum(np.asarray(hi, dtype=np.float64)
                          - np.asarray(lo, dtype=np.float64), 1.0e-9)
        # Brick-local normalized coordinates prevent a continuous wall-sized
        # scan.  The integer seed is packed in local.x at stride 2 and decoded
        # in GLSL to generate a stable per-brick UV offset/rotation.
        for vi in range(first_vertex, last_vertex):
            base = vi * 12
            p = np.asarray(vertices[base:base+3], dtype=np.float64)
            q = (p - centre) / size
            vertices[base+6] = float(q[0] + 2.0 * seed)
            vertices[base+7] = float(q[1])
            vertices[base+8] = float(q[2])
    return (np.asarray(vertices, dtype="f4").reshape(-1, 12),
            np.asarray(indices, dtype="u4"))


def build_boxes(boxes: list[tuple[tuple[float, float, float],
                                  tuple[float, float, float],
                                  tuple[float, float, float]]]):
    vertices: list[float] = []
    indices: list[int] = []
    for lo, hi, colour in boxes:
        append_box(vertices, indices, lo, hi, colour)
    return np.asarray(vertices, dtype="f4").reshape(-1, 12), np.asarray(indices, dtype="u4")


def build_truncated_ball(delta: float, wall_contact: bool = False,
                         latitudes: int = 72, longitudes: int = 112):
    """Sphere with only its contact cap replaced by the exact tangent disk.

    The canonical truncation removes the -Y cap.  For wall contact it is
    rotated +90 degrees around Z, mapping -Y to +X.  Material coordinates keep
    the uncompressed spherical direction, so the panel seams do not stretch.
    """
    centre_height = R - delta
    theta_cut = math.acos(-centre_height / R)
    pos: list[np.ndarray] = []
    local: list[np.ndarray] = []

    # One top vertex followed by latitude rings including the exact cut ring.
    pos.append(np.array([0.0, R, 0.0], dtype=np.float64))
    local.append(pos[-1].copy())
    for i in range(1, latitudes + 1):
        theta = theta_cut * i / latitudes
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(longitudes):
            phi = 2.0 * math.pi * j / longitudes
            p = np.array([R * st * math.cos(phi), R * ct,
                          R * st * math.sin(phi)], dtype=np.float64)
            pos.append(p)
            local.append(p.copy())

    indices: list[int] = []
    first = 1
    for j in range(longitudes):
        indices.extend([0, first + j, first + (j + 1) % longitudes])
    for i in range(latitudes - 1):
        a = 1 + i * longitudes
        b = a + longitudes
        for j in range(longitudes):
            jn = (j + 1) % longitudes
            indices.extend([a + j, b + j, b + jn, a + j, b + jn, a + jn])

    # Duplicate the contact ring so the flat patch has a physically correct
    # plane normal instead of smoothing into the spherical shell.
    disk_centre = len(pos)
    pos.append(np.array([0.0, -centre_height, 0.0], dtype=np.float64))
    local.append(np.array([0.0, -R, 0.0], dtype=np.float64))
    disk_ring = len(pos)
    a = math.sqrt(2.0 * R * delta - delta * delta)
    for j in range(longitudes):
        phi = 2.0 * math.pi * j / longitudes
        p = np.array([a * math.cos(phi), -centre_height,
                      a * math.sin(phi)], dtype=np.float64)
        pos.append(p)
        local.append(normalize(p) * R)
    for j in range(longitudes):
        # Clockwise from below: disk outward normal is -Y.
        indices.extend([disk_centre, disk_ring + (j + 1) % longitudes, disk_ring + j])

    pos_a = np.asarray(pos, dtype=np.float64)
    local_a = np.asarray(local, dtype=np.float64)
    if wall_contact:
        # +90 degrees about Z: (x, y, z) -> (-y, x, z), hence -Y -> +X.
        pos_a = np.column_stack((-pos_a[:, 1], pos_a[:, 0], pos_a[:, 2]))
        local_a = np.column_stack((-local_a[:, 1], local_a[:, 0], local_a[:, 2]))

    # Accumulate area-weighted normals, preserving the duplicated flat disk.
    idx = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    normals = np.zeros_like(pos_a)
    for tri in idx:
        p0, p1, p2 = pos_a[tri]
        n = np.cross(p1 - p0, p2 - p0)
        normals[tri] += n
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    colour = np.ones_like(pos_a)
    vertices = np.hstack((pos_a, normals, local_a, colour)).astype("f4")
    return vertices, idx.astype("u4").reshape(-1)


def build_sphere(latitudes: int = 64, longitudes: int = 96):
    return build_truncated_ball(0.0, wall_contact=False, latitudes=latitudes,
                                longitudes=longitudes)


def build_tube(points: np.ndarray, radius: float, colour,
               sides: int = 10):
    points = np.asarray(points, dtype=np.float64)
    vertices: list[float] = []
    indices: list[int] = []
    n = len(points)
    for i, p in enumerate(points):
        if i == 0:
            tangent = normalize(points[1] - points[0])
        elif i == n - 1:
            tangent = normalize(points[-1] - points[-2])
        else:
            tangent = normalize(points[i + 1] - points[i - 1])
        b1 = normalize(np.cross(tangent, np.array([0.0, 0.0, 1.0])))
        if np.linalg.norm(b1) < 0.1:
            b1 = np.array([0.0, 1.0, 0.0])
        b2 = normalize(np.cross(tangent, b1))
        for j in range(sides):
            a = 2.0 * math.pi * j / sides
            normal = math.cos(a) * b1 + math.sin(a) * b2
            q = p + radius * normal
            vertices.extend(pack_vertex(q, normal, q, colour))
    for i in range(n - 1):
        for j in range(sides):
            jn = (j + 1) % sides
            a, b = i * sides + j, (i + 1) * sides + j
            c, d = (i + 1) * sides + jn, i * sides + jn
            indices.extend([a, b, c, a, c, d])
    return np.asarray(vertices, dtype="f4").reshape(-1, 12), np.asarray(indices, dtype="u4")


def build_ao_disk(centre, radius: float, plane: str, segments: int = 96):
    """Soft radial contact-occlusion decal in a physical contact plane."""
    c = np.asarray(centre, dtype=np.float64)
    vertices: list[float] = []
    indices: list[int] = []
    normal = np.array([0.0, 1.0, 0.0]) if plane == "floor" else np.array([-1.0, 0.0, 0.0])
    vertices.extend(pack_vertex(c, normal, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    for j in range(segments):
        a = 2.0 * math.pi * j / segments
        u, v = math.cos(a), math.sin(a)
        if plane == "floor":
            p = c + np.array([radius * u, 0.0, radius * v])
        else:
            p = c + np.array([0.0, radius * u, radius * v])
        vertices.extend(pack_vertex(p, normal, (u, v, 0.0), (1.0, 1.0, 1.0)))
    for j in range(segments):
        indices.extend([0, 1 + (j + 1) % segments, 1 + j])
    return (np.asarray(vertices, dtype="f4").reshape(-1, 12),
            np.asarray(indices, dtype="u4"))


# ---------------------------------------------------------------------------
# GLSL
# ---------------------------------------------------------------------------
VERTEX_SHADER = r"""
#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_local;
in vec3 in_colour;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat4 u_light_view_proj;

out vec3 v_world;
out vec3 v_normal;
out vec3 v_local;
out vec3 v_colour;
out vec4 v_light_clip;

void main() {
    vec4 world = u_model * vec4(in_pos, 1.0);
    mat3 normal_matrix = transpose(inverse(mat3(u_model)));
    v_world = world.xyz;
    v_normal = normalize(normal_matrix * in_normal);
    v_local = in_local;
    v_colour = in_colour;
    v_light_clip = u_light_view_proj * world;
    gl_Position = u_view_proj * world;
}
"""

FRAGMENT_SHADER = r"""
#version 330
in vec3 v_world;
in vec3 v_normal;
in vec3 v_local;
in vec3 v_colour;
in vec4 v_light_clip;

uniform sampler2D u_shadow;
uniform sampler2D u_brick_tex;
uniform sampler2D u_concrete_tex;
uniform sampler2D u_ball_tex;
uniform vec3 u_light_dir;
uniform vec3 u_view_pos;
uniform vec3 u_base;
uniform vec3 u_rim;
uniform float u_rim_strength;
uniform float u_alpha;
uniform float u_roughness;
uniform float u_inv_scene_scale;
uniform int u_material;
uniform int u_receive_shadow;

out vec4 frag;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float value_noise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + vec3(0,0,0));
    float n100 = hash31(i + vec3(1,0,0));
    float n010 = hash31(i + vec3(0,1,0));
    float n110 = hash31(i + vec3(1,1,0));
    float n001 = hash31(i + vec3(0,0,1));
    float n101 = hash31(i + vec3(1,0,1));
    float n011 = hash31(i + vec3(0,1,1));
    float n111 = hash31(i + vec3(1,1,1));
    return mix(mix(mix(n000,n100,f.x),mix(n010,n110,f.x),f.y),
               mix(mix(n001,n101,f.x),mix(n011,n111,f.x),f.y),f.z);
}

vec3 srgb_to_linear(vec3 c) {
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

vec3 perturb_normal(vec3 surf_pos, vec3 surf_normal,
                    float height_signal, float amplitude_m) {
    vec3 dpdx = dFdx(surf_pos);
    vec3 dpdy = dFdy(surf_pos);
    float dhdx = dFdx(height_signal);
    float dhdy = dFdy(height_signal);
    vec3 r1 = cross(dpdy, surf_normal);
    vec3 r2 = cross(surf_normal, dpdx);
    float det = dot(dpdx, r1);
    vec3 gradient = (r1 * dhdx + r2 * dhdy)
                  * (sign(det) / max(abs(det), 1.0e-9));
    return normalize(surf_normal - amplitude_m * gradient);
}

float shadow_visibility(vec3 n) {
    if (u_receive_shadow == 0) return 1.0;
    vec3 q = v_light_clip.xyz / v_light_clip.w;
    q = q * 0.5 + 0.5;
    if (q.x <= 0.0 || q.x >= 1.0 || q.y <= 0.0 || q.y >= 1.0 || q.z >= 1.0)
        return 1.0;
    float ndl = max(dot(n, u_light_dir), 0.0);
    float bias = max(0.00082, 0.00165 * (1.0 - ndl));
    // Kernel footprint corresponds to the finite angular radius of the long
    // ceiling panel, yielding one coherent soft penumbra for every object.
    vec2 texel = 8.0 / vec2(textureSize(u_shadow, 0));
    float visible = 0.0;
    float weights = 0.0;
    for (int x = -3; x <= 3; ++x) {
        for (int y = -3; y <= 3; ++y) {
            float w = 1.0 - 0.10 * length(vec2(x,y));
            float d = texture(u_shadow, q.xy + vec2(x,y) * texel).r;
            visible += w * ((q.z - bias <= d) ? 1.0 : 0.0);
            weights += w;
        }
    }
    return visible / weights;
}

void main() {
    vec3 n = normalize(v_normal);
    vec3 albedo = u_base * v_colour;
    float rough = u_roughness;
    float spec_scale = 1.0;

    if (u_material == 0) { // high-roughness cast-concrete floor
        vec2 uv = v_world.xz / 0.72;
        vec3 scan = srgb_to_linear(texture(u_concrete_tex, uv).rgb);
        float lum = dot(scan, vec3(0.2126, 0.7152, 0.0722));
        float coarse = value_noise(v_world * 1.6);
        albedo *= scan * 2.15 * (0.91 + 0.14 * coarse);
        n = perturb_normal(v_world, n, lum, 0.00062);
        rough = clamp(0.84 + 0.13 * (1.0 - lum), 0.84, 0.97);
    } else if (u_material == 1) { // formed rear concrete with panel joints
        vec2 uv = vec2(v_world.x, v_world.y) / 0.54 + vec2(0.17, 0.31);
        vec3 scan = srgb_to_linear(texture(u_concrete_tex, uv).rgb);
        float lum = dot(scan, vec3(0.2126, 0.7152, 0.0722));
        vec2 panel_pitch = vec2(0.62, 0.44);
        vec2 panel_q = fract((vec2(v_world.x, v_world.y) + vec2(0.19, 0.0)) / panel_pitch);
        vec2 edge_d = min(panel_q, 1.0 - panel_q) * panel_pitch;
        float groove = 1.0 - smoothstep(0.0015, 0.0045, min(edge_d.x, edge_d.y));
        vec2 panel_id = floor((vec2(v_world.x, v_world.y) + vec2(0.19, 0.0)) / panel_pitch);
        float panel_tone = 0.93 + 0.11 * hash31(vec3(panel_id, 3.7));
        albedo *= scan * 2.12 * panel_tone;
        albedo = mix(albedo, albedo * 0.46, groove);
        n = perturb_normal(v_world, n, lum - 2.0 * groove, 0.00042);
        rough = clamp(0.87 + 0.10 * (1.0 - lum), 0.87, 0.98);
    } else if (u_material == 2) { // recessed cement-lime mortar
        vec2 uv = vec2(v_world.z, v_world.y) / 0.16;
        vec3 scan = srgb_to_linear(texture(u_concrete_tex, uv).rgb);
        float lum = dot(scan, vec3(0.2126, 0.7152, 0.0722));
        albedo *= scan * 2.05;
        n = perturb_normal(v_world, n, lum, 0.00034);
        rough = 0.96;
    } else if (u_material == 3) { // individually UV-mapped fired-clay brick
        float brick_seed = floor((v_local.x + 1.0) / 2.0);
        vec3 q = v_local;
        q.x -= 2.0 * brick_seed;
        vec2 uv;
        if (abs(n.x) > 0.62)
            uv = vec2(q.z, q.y);
        else if (abs(n.z) > 0.62)
            uv = vec2(q.x, q.y);
        else
            uv = vec2(q.z, q.x);
        float angle = (hash31(vec3(brick_seed, 1.7, 8.3)) - 0.5) * 0.34;
        mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        vec2 offset = vec2(hash31(vec3(brick_seed, 2.1, 5.7)),
                           hash31(vec3(brick_seed, 9.2, 0.4))) * 2.7;
        uv = rot * (uv * vec2(1.08, 0.92)) + offset;
        vec3 scan = srgb_to_linear(texture(u_brick_tex, uv).rgb);
        float lum = dot(scan, vec3(0.2126, 0.7152, 0.0722));
        scan = mix(scan, vec3(lum), 0.16);
        albedo *= scan * 1.12;
        n = perturb_normal(v_world, n, lum, 0.00058);
        rough = clamp(0.76 + 0.14 * (1.0 - lum), 0.76, 0.92);
        spec_scale = 0.58;
    } else if (u_material == 4) { // pebbled rubber basketball shell
        vec3 p = normalize(v_local);
        vec2 sphere_uv = vec2(atan(p.z, p.x) / 6.28318530718 + 0.5,
                              asin(clamp(p.y, -1.0, 1.0)) / 3.14159265359 + 0.5);
        sphere_uv *= vec2(5.5, 2.75);
        vec3 scan = srgb_to_linear(texture(u_ball_tex, sphere_uv).rgb);
        float scan_lum = dot(scan, vec3(0.2126, 0.7152, 0.0722));
        float seam = min(abs(p.x), abs(p.z));
        seam = min(seam, abs(dot(p, normalize(vec3(0.68, 0.70, 0.22)))));
        float seam_mask = 1.0 - smoothstep(0.006, 0.014, seam);
        // Preserve the scan's colour variation while compressing rare bright
        // texels so they cannot masquerade as a pasted-on white highlight.
        scan = min(scan, vec3(0.74, 0.19, 0.060));
        albedo *= scan * 1.02;
        albedo = mix(albedo, vec3(0.006, 0.004, 0.003), seam_mask);
        // A restrained 0.2--0.3 mm scan-derived micro-normal plus a separate
        // approximately 1.2 mm channel recess avoids hard sticker-like spots.
        n = perturb_normal(v_world, n, scan_lum, 0.00023);
        n = perturb_normal(v_world, n, -seam_mask, 0.00115);
        rough = mix(clamp(0.62 + 0.07 * (1.0 - scan_lum), 0.62, 0.69),
                    0.82, seam_mask);
        spec_scale = 0.30;
    }

    // The trajectory tubes are visual measurements, not luminous objects in
    // the physical scene; a small view-independent term keeps them legible.
    if (u_material == 5 || u_material == 6) {
        frag = vec4(albedo, u_alpha);
        return;
    }
    if (u_material == 7) {
        float radial = length(v_local.xy);
        float fade = pow(1.0 - smoothstep(0.10, 1.0, radial), 2.2);
        frag = vec4(vec3(0.015, 0.017, 0.019), u_alpha * fade);
        return;
    }

    float visibility = shadow_visibility(n);
    float ndl = max(dot(n, u_light_dir), 0.0);
    vec3 view_dir = normalize(u_view_pos - v_world);
    vec3 half_dir = normalize(u_light_dir + view_dir);
    float exponent = mix(110.0, 7.0, rough);
    if (u_material == 4) exponent = 15.0;
    float spec = pow(max(dot(n, half_dir), 0.0), exponent)
               * (1.0 - rough) * 0.34 * spec_scale;

    // Cool room fill and a warmer overhead area key share the same geometric
    // normal; only the latter is shadowed by the physical shadow map.
    float hemi = 0.30 + 0.12 * max(n.y, 0.0);
    vec3 cool_fill = vec3(0.67, 0.76, 0.88) * hemi;
    vec3 warm_key = vec3(1.12, 0.91, 0.70) * (1.24 * ndl * visibility);
    vec3 linear = albedo * (cool_fill + warm_key)
                + vec3(1.0, 0.86, 0.68) * spec * visibility;
    float rim = pow(1.0 - max(dot(n, view_dir), 0.0), 3.1) * u_rim_strength;
    linear = mix(linear, u_rim, clamp(rim, 0.0, 0.72));

    linear = vec3(1.0) - exp(-linear * 1.18);
    linear = pow(linear, vec3(1.0 / 2.2));
    frag = vec4(linear, u_alpha);
}
"""

DEPTH_VERTEX = r"""
#version 330
in vec3 in_pos;
uniform mat4 u_model;
uniform mat4 u_light_view_proj;
void main() {
    gl_Position = u_light_view_proj * u_model * vec4(in_pos, 1.0);
}
"""

DEPTH_FRAGMENT = r"""
#version 330
void main() {}
"""


def make_mesh(ctx, program, depth_program, vertices, indices) -> Mesh:
    vertices = np.asarray(vertices, dtype="f4")
    indices = np.asarray(indices, dtype="u4")
    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(
        program,
        [(vbo, "3f 3f 3f 3f", "in_pos", "in_normal", "in_local", "in_colour")],
        ibo,
    )
    depth_vao = ctx.vertex_array(depth_program, [(vbo, "3f 36x", "in_pos")], ibo)
    return Mesh(vbo, ibo, vao, depth_vao, len(indices))


@dataclass
class Draw:
    mesh: Mesh
    model: np.ndarray
    material: int
    base: tuple[float, float, float]
    roughness: float = 0.8
    alpha: float = 1.0
    rim: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rim_strength: float = 0.0
    casts_shadow: bool = True
    receives_shadow: bool = True


def set_uniform(program, name, value):
    if name not in program:
        return
    u = program[name]
    if isinstance(value, np.ndarray) and value.shape == (4, 4):
        u.write(gl_bytes(value))
    else:
        u.value = value


def load_albedo_texture(ctx: moderngl.Context, path: Path) -> moderngl.Texture:
    """Load a repeatable albedo scan; lighting remains entirely in GLSL."""
    image = Image.open(path).convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    tex = ctx.texture(image.size, 3, image.tobytes())
    tex.repeat_x = True
    tex.repeat_y = True
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    try:
        tex.anisotropy = min(8.0, float(ctx.max_anisotropy))
    except Exception:
        pass
    return tex


def floor_y_out(t: float) -> float:
    return R + V_Y_OUT * t - 0.5 * G * t * t


def floor_y_in(t: float) -> float:
    return R + V_Y_IN * t - 0.5 * G * t * t


def wall_y(tau: float) -> float:
    return Y_WALL_CONTACT + V_Y_WALL * tau - 0.5 * G * tau * tau


def ball_model(x: float, y: float, z: float, spin: float) -> np.ndarray:
    return translation(x, y, z) @ rotation_z(spin) @ rotation_y(0.23)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context()
    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    depth_program = ctx.program(vertex_shader=DEPTH_VERTEX, fragment_shader=DEPTH_FRAGMENT)
    material_dir = HERE / "assets" / "materials"
    brick_tex = load_albedo_texture(ctx, material_dir / "brick_clay_albedo_v1.png")
    concrete_tex = load_albedo_texture(ctx, material_dir / "concrete_albedo_v1.png")
    ball_tex = load_albedo_texture(ctx, material_dir / "basketball_rubber_albedo_v1.png")

    # Single orthographic camera: affine projection preserves the exact
    # parabolic form of the coplanar ballistic centre trajectory.
    target = (np.array([-0.05, 1.35, 0.38], dtype=np.float32)
              * SCENE_SCALE)
    # The camera is on the incoming side of the collision plane.  Its modest
    # yaw reveals the brick faces at x=X_WALL without sacrificing the affine
    # parabolic projection; positive world X consequently reads left-to-right.
    view_dir = normalize(np.array([0.58, -0.140, -0.813], dtype=np.float32))
    eye = target - (14.0 * SCENE_SCALE) * view_dir
    view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    half_h = 2.70 * SCENE_SCALE
    half_w = half_h * OUT_W / OUT_H
    projection = ortho(-half_w, half_w, -half_h, half_h,
                       0.1 * SCENE_SCALE, 40.0 * SCENE_SCALE)
    view_proj = projection @ view

    # One broad overhead-left source.  The same vector drives the shadow map,
    # diffuse shading, highlights, contact shadows, and wall cast shadow.
    light_target = (np.array([-0.10, 1.60, 0.55], dtype=np.float32)
                    * SCENE_SCALE)
    fixture_centre = (np.array([-2.51, 3.825, 1.01], dtype=np.float32)
                      * SCENE_SCALE)
    light_dir = normalize(fixture_centre - light_target)
    light_eye = light_target + (14.0 * SCENE_SCALE) * light_dir
    light_view = look_at(light_eye, light_target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    light_proj = ortho(-7.2 * SCENE_SCALE, 7.2 * SCENE_SCALE,
                       -5.7 * SCENE_SCALE, 5.7 * SCENE_SCALE,
                       0.1 * SCENE_SCALE, 32.0 * SCENE_SCALE)
    light_view_proj = light_proj @ light_view

    # Shared reusable ball meshes.
    sphere_mesh = make_mesh(ctx, program, depth_program, *build_sphere())
    ground_ball_mesh = make_mesh(ctx, program, depth_program,
                                 *build_truncated_ball(DELTA_G))
    wall_ball_mesh = make_mesh(ctx, program, depth_program,
                               *build_truncated_ball(DELTA_W, wall_contact=True))

    draws: list[Draw] = []

    floor_v, floor_i = build_boxes([
        ((-55.0 * SCENE_SCALE, -0.10 * SCENE_SCALE, -55.0 * SCENE_SCALE),
         (55.0 * SCENE_SCALE, 0.0, 55.0 * SCENE_SCALE), (1.0, 1.0, 1.0)),
    ])
    floor_mesh = make_mesh(ctx, program, depth_program, floor_v, floor_i)
    draws.append(Draw(floor_mesh, np.eye(4, dtype=np.float32), 0,
                      (0.52, 0.50, 0.47), roughness=0.92,
                      casts_shadow=False))

    rear_v, rear_i = build_boxes([
        ((-7.5 * SCENE_SCALE, 0.0, -3.66 * SCENE_SCALE),
         (7.5 * SCENE_SCALE, 4.9 * SCENE_SCALE, -3.55 * SCENE_SCALE),
         (1.0, 1.0, 1.0)),
    ])
    rear_mesh = make_mesh(ctx, program, depth_program, rear_v, rear_i)
    draws.append(Draw(rear_mesh, np.eye(4, dtype=np.float32), 1,
                      (0.61, 0.59, 0.56), roughness=0.93))

    # Continuous load-bearing mortar/backing closes every joint.  Its exposed
    # face is recessed 4 mm from the common brick collision plane.
    mortar_v, mortar_i = build_boxes([
        ((X_WALL + MORTAR_RECESS, 0.0, WALL_Z_MIN),
         (X_WALL + WALL_THICKNESS, WALL_HEIGHT, WALL_Z_MAX),
         (1.0, 1.0, 1.0)),
    ])
    mortar_mesh = make_mesh(ctx, program, depth_program, mortar_v, mortar_i)
    draws.append(Draw(mortar_mesh, np.eye(4, dtype=np.float32), 2,
                      (0.58, 0.535, 0.47), roughness=0.96))

    # Standard 215 x 65 x 102.5 mm real solids in running bond.  Every brick
    # is a closed chamfered mesh; partial boundary units are sawn solids, never
    # open shells.  A seeded neutral multiplier supplies batch variation while
    # the scan UV receives its own stable per-brick offset and rotation.
    brick_boxes = []
    rng = np.random.default_rng(20260815)
    row_pitch = BRICK_HEIGHT + MORTAR_GAP
    col_pitch = BRICK_LENGTH + MORTAR_GAP
    row = 0
    y0 = 0.0
    while y0 + BRICK_HEIGHT <= WALL_HEIGHT + 1.0e-9:
        offset = -0.5 * col_pitch if row % 2 else 0.0
        z0 = WALL_Z_MIN + offset
        while z0 < WALL_Z_MAX:
            bz0 = max(z0, WALL_Z_MIN)
            bz1 = min(z0 + BRICK_LENGTH, WALL_Z_MAX)
            if bz1 - bz0 > 2.5 * BRICK_BEVEL:
                variation = float(rng.uniform(0.88, 1.10))
                tint = (variation, variation * 0.985, variation * 0.965)
                brick_boxes.append(((X_WALL, y0, bz0),
                                    (X_WALL + BRICK_DEPTH,
                                     y0 + BRICK_HEIGHT, bz1), tint))
            z0 += col_pitch
        row += 1
        y0 = row * row_pitch
    brick_v, brick_i = build_bevelled_boxes(brick_boxes, BRICK_BEVEL)
    brick_mesh = make_mesh(ctx, program, depth_program, brick_v, brick_i)
    draws.append(Draw(brick_mesh, np.eye(4, dtype=np.float32), 3,
                      (1.0, 1.0, 1.0), roughness=0.81))

    # The visible fixture is centered on the same overhead-left source used
    # by u_light_dir and the shadow camera.  It is deliberately clipped by the
    # top edge, making the lighting provenance visible without dominating the
    # evidence panel.
    housing_v, housing_i = build_boxes([
        ((-3.62 * SCENE_SCALE, 3.83 * SCENE_SCALE, 0.72 * SCENE_SCALE),
         (-1.40 * SCENE_SCALE, 3.92 * SCENE_SCALE, 1.30 * SCENE_SCALE),
         (1.0, 1.0, 1.0)),
    ])
    housing_mesh = make_mesh(ctx, program, depth_program, housing_v, housing_i)
    draws.append(Draw(housing_mesh, np.eye(4, dtype=np.float32), 2,
                      (0.17, 0.18, 0.18), 0.82, casts_shadow=False))
    emitter_v, emitter_i = build_boxes([
        ((-3.50 * SCENE_SCALE, 3.815 * SCENE_SCALE, 0.80 * SCENE_SCALE),
         (-1.52 * SCENE_SCALE, 3.835 * SCENE_SCALE, 1.22 * SCENE_SCALE),
         (1.0, 1.0, 1.0)),
    ])
    emitter_mesh = make_mesh(ctx, program, depth_program, emitter_v, emitter_i)
    draws.append(Draw(emitter_mesh, np.eye(4, dtype=np.float32), 6,
                      (1.0, 0.94, 0.80), 0.2, casts_shadow=False,
                      receives_shadow=False))

    # Exact maximum-compression states.  Only the removed cap is flattened;
    # the remaining pressurised shell is spherical.
    spin_ground = 0.54
    ground_model = ball_model(X_GROUND, R - DELTA_G, Z_MOTION, spin_ground)
    draws.append(Draw(ground_ball_mesh, ground_model, 4,
                      (1.0, 1.0, 1.0), roughness=0.60))

    # For the wall-truncated mesh, spin is kept out of the model transform so
    # the finite contact disk remains coplanar with x=X_WALL.  Panel rotation
    # is represented by the fixed local seam orientation.
    wall_model = translation(X_WALL_CONTACT, Y_WALL_CONTACT, Z_MOTION)
    draws.append(Draw(wall_ball_mesh, wall_model, 4,
                      (1.0, 1.0, 1.0), roughness=0.60))

    # Ballistic tubes and branches are derived from the same equations as the
    # sample positions; no hand-drawn control points are used.
    incoming_t = np.linspace(-0.62 * TIME_SCALE, -0.045 * TIME_SCALE, 70)
    incoming_pts = np.array([[X_GROUND + V_X * t, floor_y_in(t), Z_MOTION]
                             for t in incoming_t] +
                            [[X_GROUND, R - DELTA_G, Z_MOTION]])
    outgoing_t = np.linspace(0.035 * TIME_SCALE, T_WALL, 90)
    outgoing_pts = np.array([[X_GROUND, R - DELTA_G, Z_MOTION]] +
                            [[X_GROUND + V_X * t, floor_y_out(t), Z_MOTION]
                             for t in outgoing_t])
    wall_tau = np.linspace(0.0, 0.49 * TIME_SCALE, 65)
    expected_pts = np.array([[X_WALL_CONTACT - E_W * V_X * t,
                              wall_y(t), Z_MOTION] for t in wall_tau])
    wrong_pts = np.array([[X_WALL_CONTACT + V_X * t,
                           wall_y(t), Z_MOTION] for t in wall_tau])

    path_specs = [
        (incoming_pts, (0.23, 0.28, 0.30)),
        (outgoing_pts, (0.055, 0.48, 0.44)),
        (expected_pts, (0.055, 0.48, 0.44)),
        (wrong_pts, (0.98, 0.31, 0.22)),
    ]
    for pts, colour in path_specs:
        tv, ti = build_tube(pts, 0.0085 * SCENE_SCALE, colour, sides=9)
        tm = make_mesh(ctx, program, depth_program, tv, ti)
        # Tube vertices already carry the measured branch colour; a unit
        # uniform avoids accidentally squaring it in the shader.
        draws.append(Draw(tm, np.eye(4, dtype=np.float32), 5, (1.0, 1.0, 1.0),
                          roughness=1.0, alpha=0.92, casts_shadow=False,
                          receives_shadow=False))

    # Equal-time samples.  Their separation follows speed automatically.
    ghost_draws: list[Draw] = []
    floor_ao_v, floor_ao_i = build_ao_disk(
        (X_GROUND, 0.00020, Z_MOTION), 1.85 * FLOOR_PATCH_RADIUS, "floor")
    floor_ao_mesh = make_mesh(ctx, program, depth_program, floor_ao_v, floor_ao_i)
    ghost_draws.append(Draw(floor_ao_mesh, np.eye(4, dtype=np.float32), 7,
                            (1.0, 1.0, 1.0), 1.0, 0.34,
                            casts_shadow=False, receives_shadow=False))
    wall_ao_v, wall_ao_i = build_ao_disk(
        (X_WALL - 0.00020, Y_WALL_CONTACT, Z_MOTION),
        1.85 * WALL_PATCH_RADIUS, "wall")
    wall_ao_mesh = make_mesh(ctx, program, depth_program, wall_ao_v, wall_ao_i)
    ghost_draws.append(Draw(wall_ao_mesh, np.eye(4, dtype=np.float32), 7,
                            (1.0, 1.0, 1.0), 1.0, 0.31,
                            casts_shadow=False, receives_shadow=False))
    for i, t in enumerate(np.linspace(-0.60 * TIME_SCALE,
                                      -0.085 * TIME_SCALE, 7)):
        x, y = X_GROUND + V_X * t, floor_y_in(t)
        spin = spin_ground - (x - X_GROUND) / R
        alpha = 0.075 + 0.013 * i
        ghost_draws.append(Draw(sphere_mesh, ball_model(x, y, Z_MOTION, spin), 4,
                                (1.0, 1.0, 1.0), 0.60, alpha,
                                casts_shadow=False))
    for i, t in enumerate(np.linspace(0.085 * TIME_SCALE,
                                      T_WALL - 0.070 * TIME_SCALE, 8)):
        x, y = X_GROUND + V_X * t, floor_y_out(t)
        spin = spin_ground - (x - X_GROUND) / R
        alpha = 0.075 + 0.012 * i
        ghost_draws.append(Draw(sphere_mesh, ball_model(x, y, Z_MOTION, spin), 4,
                                (1.0, 1.0, 1.0), 0.60, alpha,
                                rim=(0.05, 0.58, 0.53), rim_strength=0.22,
                                casts_shadow=False))
    for i, tau0 in enumerate((0.09, 0.18, 0.27, 0.36, 0.45)):
        tau = tau0 * TIME_SCALE
        x, y = X_WALL_CONTACT - E_W * V_X * tau, wall_y(tau)
        spin = spin_ground + E_W * V_X * tau / R
        ghost_draws.append(Draw(sphere_mesh, ball_model(x, y, Z_MOTION, spin), 4,
                                (1.0, 1.0, 1.0), 0.60, 0.085 + 0.018 * i,
                                rim=(0.03, 0.62, 0.55), rim_strength=0.76,
                                casts_shadow=False))
    for i, tau0 in enumerate((0.10, 0.20, 0.30, 0.40, 0.49)):
        tau = tau0 * TIME_SCALE
        x, y = X_WALL_CONTACT + V_X * tau, wall_y(tau)
        spin = spin_ground - V_X * tau / R
        ghost_draws.append(Draw(sphere_mesh, ball_model(x, y, Z_MOTION, spin), 4,
                                (1.0, 1.0, 1.0), 0.60, 0.085 + 0.018 * i,
                                rim=(1.0, 0.18, 0.10), rim_strength=0.84,
                                casts_shadow=False))

    # Depth-only pass for one coherent shadow field.
    shadow_tex = ctx.depth_texture((SHADOW_SIZE, SHADOW_SIZE))
    shadow_tex.compare_func = ""
    shadow_tex.repeat_x = False
    shadow_tex.repeat_y = False
    shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    shadow_fbo = ctx.framebuffer(depth_attachment=shadow_tex)
    shadow_fbo.use()
    shadow_fbo.clear(depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.cull_face = "back"
    set_uniform(depth_program, "u_light_view_proj", light_view_proj)
    for d in draws:
        if not d.casts_shadow:
            continue
        set_uniform(depth_program, "u_model", d.model)
        d.mesh.depth_vao.render(moderngl.TRIANGLES)

    # Main high-resolution colour pass.
    colour_tex = ctx.texture((RENDER_W, RENDER_H), 4, dtype="f1")
    depth_buffer = ctx.depth_renderbuffer((RENDER_W, RENDER_H))
    fbo = ctx.framebuffer(color_attachments=[colour_tex], depth_attachment=depth_buffer)
    fbo.use()
    fbo.clear(0.44, 0.425, 0.395, 1.0, depth=1.0)
    ctx.viewport = (0, 0, RENDER_W, RENDER_H)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    ctx.disable(moderngl.BLEND)
    shadow_tex.use(location=0)
    brick_tex.use(location=1)
    concrete_tex.use(location=2)
    ball_tex.use(location=3)
    set_uniform(program, "u_shadow", 0)
    set_uniform(program, "u_brick_tex", 1)
    set_uniform(program, "u_concrete_tex", 2)
    set_uniform(program, "u_ball_tex", 3)
    set_uniform(program, "u_view_proj", view_proj)
    set_uniform(program, "u_light_view_proj", light_view_proj)
    set_uniform(program, "u_light_dir", tuple(float(x) for x in light_dir))
    set_uniform(program, "u_view_pos", tuple(float(x) for x in eye))
    set_uniform(program, "u_inv_scene_scale", float(INV_SCENE_SCALE))

    def render_draw(d: Draw) -> None:
        set_uniform(program, "u_model", d.model)
        set_uniform(program, "u_material", d.material)
        set_uniform(program, "u_base", d.base)
        set_uniform(program, "u_roughness", d.roughness)
        set_uniform(program, "u_alpha", d.alpha)
        set_uniform(program, "u_rim", d.rim)
        set_uniform(program, "u_rim_strength", d.rim_strength)
        set_uniform(program, "u_receive_shadow", int(d.receives_shadow))
        d.mesh.vao.render(moderngl.TRIANGLES)

    # Opaque physical geometry first, then measurement tubes.
    for d in draws:
        render_draw(d)

    # Transparent temporal states: sorted far-to-near in camera space, depth
    # tested against the wall and floor but not written into the depth buffer.
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.depth_mask = False
    ghost_draws.sort(key=lambda d: float((view @ d.model)[2, 3]))
    for d in ghost_draws:
        render_draw(d)
    ctx.depth_mask = True

    raw = fbo.read(components=4, alignment=1)
    image = Image.frombytes("RGBA", (RENDER_W, RENDER_H), raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image = image.resize((OUT_W, OUT_H), Image.Resampling.LANCZOS).convert("RGB")
    image.save(OUTPUT, quality=96)

    print(f"wrote {OUTPUT} ({OUT_W}x{OUT_H})")
    print(f"scene scale s={SCENE_SCALE:.9f}; dynamic time scale sqrt(s)={TIME_SCALE:.9f}")
    print(f"adult basketball: R={R:.3f} m, diameter={2*R:.3f} m")
    print(f"delta_g/R={DELTA_G/R:.3f}, a_g={FLOOR_PATCH_RADIUS:.4f} m, e_g={E_G:.2f}")
    print(f"delta_w/R={DELTA_W/R:.3f}, a_w={WALL_PATCH_RADIUS:.4f} m, e_w={E_W:.2f}")
    print(f"launch/rebound velocity: vx={V_X:.4f} m/s, vy+={V_Y_OUT:.4f} m/s, vy-={V_Y_IN:.4f} m/s")
    print(f"spin rates from v/R: |omega_flight|={V_X/R:.4f} rad/s, |omega_rebound|={E_W*V_X/R:.4f} rad/s")
    print(f"wall flight time={T_WALL:.4f} s; pre-wall vy={V_Y_WALL:.4f} m/s")
    print(f"wall dimensions: height={WALL_HEIGHT:.4f} m, depth={WALL_Z_MAX-WALL_Z_MIN:.4f} m, thickness={WALL_THICKNESS:.4f} m")
    print(f"real brick solids: {BRICK_LENGTH:.4f} x {BRICK_HEIGHT:.4f} x {BRICK_DEPTH:.4f} m; count={len(brick_boxes)}")
    print(f"mortar gap={MORTAR_GAP:.4f} m, recess={MORTAR_RECESS:.4f} m; brick bevel={BRICK_BEVEL:.4f} m")
    print(f"area-light emitter: {1.98*SCENE_SCALE:.4f} x {0.42*SCENE_SCALE:.4f} m")
    print(f"ground contact=({X_GROUND:.4f}, {R-DELTA_G:.4f}, {Z_MOTION:.4f}) m")
    print(f"wall plane x={X_WALL:.4f} m; wall contact centre=({X_WALL_CONTACT:.4f}, {Y_WALL_CONTACT:.4f}, {Z_MOTION:.4f}) m")
    print(f"light direction (surface->source)={tuple(round(float(x),4) for x in light_dir)}")
    print("key pixels (origin top-left):")
    print(f"  ground-contact centre={project_pixel((X_GROUND, R-DELTA_G, Z_MOTION), view_proj)}")
    print(f"  floor contact={project_pixel((X_GROUND, 0.0, Z_MOTION), view_proj)}")
    print(f"  wall-contact centre={project_pixel((X_WALL_CONTACT, Y_WALL_CONTACT, Z_MOTION), view_proj)}")
    print(f"  wall-plane contact={project_pixel((X_WALL, Y_WALL_CONTACT, Z_MOTION), view_proj)}")
    print(f"  wall foot={project_pixel((X_WALL, 0.0, Z_MOTION), view_proj)}")
    print(f"  area-light centre={project_pixel(fixture_centre, view_proj)}")


if __name__ == "__main__":
    main()
