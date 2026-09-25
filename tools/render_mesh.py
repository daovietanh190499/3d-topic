import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw


def rot(ax, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def render(path, out, rx, ry, rz, size=700, center=None, scale=None):
    m = trimesh.load(path, force="mesh")
    if len(m.faces) > 400000:
        m = m.simplify_quadric_decimation(face_count=400000)
    v = np.asarray(m.vertices, dtype=np.float64)
    if center is None:
        center = np.median(v, axis=0)
    v = v - center
    R = rot("z", rz) @ rot("y", ry) @ rot("x", rx)
    v = v @ R.T
    if scale is None:
        scale = 0.42 * size / np.percentile(np.abs(v[:, :2]), 99)
    f = np.asarray(m.faces)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    shade = np.abs(n @ light)
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    xy = tri[:, :, :2] * scale
    xy[:, :, 0] += size / 2
    xy[:, :, 1] = size / 2 - xy[:, :, 1]
    for i in order:
        g = int(60 + 185 * shade[i])
        p = [tuple(q) for q in xy[i]]
        dr.polygon(p, fill=(g, g, int(g * 1.02) if g < 250 else g))
    img.save(out)
    return center, scale


if __name__ == "__main__":
    a = sys.argv
    render(a[1], a[2], float(a[3]), float(a[4]), float(a[5]))
