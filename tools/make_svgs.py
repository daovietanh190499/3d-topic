"""Sinh các sơ đồ SVG có tính toán định lượng cho bốn deck."""
import numpy as np
from scipy.spatial import Delaunay, Voronoi

OUT = "/mnt/data/research/slide/nghien-cuu-3d/img/"
P, A = "#2f3e7a", "#b15a2b"
FONT = "font-family='Helvetica,Arial,sans-serif'"


def svg(w, h, body, title):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' {FONT}>"
            f"<title>{title}</title><rect width='{w}' height='{h}' fill='white'/>{body}</svg>")


def save(path, s):
    with open(OUT + path, "w") as f:
        f.write(s)


def poly(xs, ys, color, width=2.5, dash=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) > 500:
        k = int(np.ceil(len(xs) / 500))
        keep = np.r_[np.arange(0, len(xs), k), len(xs) - 1]
        xs, ys = xs[keep], ys[keep]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{width}'{d}/>"


def text(x, y, s, size=15, color="#222", anchor="middle", weight="normal", italic=False):
    st = " font-style='italic'" if italic else ""
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{color}' "
            f"text-anchor='{anchor}' font-weight='{weight}'{st}>{s}</text>")


class Axes:
    def __init__(self, x0, y0, w, h, xr, yr):
        self.x0, self.y0, self.w, self.h, self.xr, self.yr = x0, y0, w, h, xr, yr

    def X(self, x):
        return self.x0 + (np.asarray(x) - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, y):
        return self.y0 + self.h - (np.asarray(y) - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def frame(self, xlabel, ylabel):
        b = (f"<line x1='{self.x0}' y1='{self.y0+self.h}' x2='{self.x0+self.w}' y2='{self.y0+self.h}' stroke='#444' stroke-width='1.5'/>"
             f"<line x1='{self.x0}' y1='{self.y0}' x2='{self.x0}' y2='{self.y0+self.h}' stroke='#444' stroke-width='1.5'/>")
        b += text(self.x0 + self.w / 2, self.y0 + self.h + 34, xlabel, 15)
        b += (f"<text x='{self.x0-38}' y='{self.y0+self.h/2}' font-size='15' text-anchor='middle' "
              f"transform='rotate(-90 {self.x0-38} {self.y0+self.h/2})'>{ylabel}</text>")
        return b


def phi(x, s):
    e = np.exp(-s * x)
    return s * e / (1 + e) ** 2


def Phi(x, s):
    return 1 / (1 + np.exp(-s * x))


# ---------------- NeuS: logistic density ----------------
def neus_logistic():
    body = ""
    x = np.linspace(-1, 1, 400)
    cols = ["#9aa6cf", "#5a6aa8", P]
    ax = Axes(70, 40, 330, 220, (-1, 1), (0, 10.5))
    body += ax.frame("giá trị SDF x", "φₛ(x)")
    for s, c in zip([5, 15, 40], cols):
        body += poly(ax.X(x), ax.Y(np.minimum(phi(x, s), 10.4)), c)
    body += text(ax.X(0.42), ax.Y(9.6), "s = 40", 14, P, "start")
    body += text(ax.X(0.42), ax.Y(4.6), "s = 15", 14, "#5a6aa8", "start")
    body += text(ax.X(0.42), ax.Y(1.9), "s = 5", 14, "#6f7bb0", "start")
    body += text(235, 24, "Mật độ logistic φₛ (S-density)", 16, P, weight="bold")
    ax2 = Axes(520, 40, 330, 220, (-1, 1), (0, 1.05))
    body += ax2.frame("giá trị SDF x", "Φₛ(x)")
    for s, c in zip([5, 15, 40], cols):
        body += poly(ax2.X(x), ax2.Y(Phi(x, s)), c)
    body += text(685, 24, "Hàm phân phối Φₛ = sigmoid(s·x)", 16, P, weight="bold")
    body += text(460, 318, "Độ lệch chuẩn của phân phối là π/(s√3); khi s tăng, trọng số co về mặt mức 0.", 14, "#444")
    save("neus/logistic.svg", svg(900, 330, body, "Mật độ logistic và hàm phân phối của NeuS với s bằng 5, 15, 40"))


# ---------------- NeuS: bias of weight functions ----------------
def weights_along(f, t, s):
    dt = t[1] - t[0]
    sig = phi(f, s)
    T = np.exp(-np.concatenate([[0], np.cumsum(sig[:-1] * dt)]))
    w_naive = T * sig
    Pf = Phi(f, s)
    dPhi = np.gradient(Pf, dt)
    rho = np.maximum(-dPhi / np.maximum(Pf, 1e-12), 0)
    T2 = np.exp(-np.concatenate([[0], np.cumsum(rho[:-1] * dt)]))
    w_neus = T2 * rho
    return w_naive, w_neus


def neus_bias():
    body = ""
    s = 12.0
    t = np.linspace(0, 1, 4000)
    # Panel 1: single plane at t*=0.5
    ts = 0.5
    f = -(t - ts)
    wn, wu = weights_along(f, t, s)
    ax = Axes(70, 50, 360, 220, (0, 1), (-0.1, 1.08))
    body += ax.frame("độ sâu t dọc tia", "trọng số (chuẩn hóa)")
    body += poly(ax.X(t), ax.Y(wn / wn.max()), A, 2.8)
    body += poly(ax.X(t), ax.Y(wu / wu.max()), P, 2.8)
    body += poly(ax.X(t), ax.Y(np.clip(f + 0.5, -0.1, 1.08) * 0.8), "#999", 1.6, "5,4")
    tn = t[np.argmax(wn)]
    body += f"<line x1='{ax.X(ts):.1f}' y1='{ax.Y(-0.1):.1f}' x2='{ax.X(ts):.1f}' y2='{ax.Y(1.08):.1f}' stroke='#2e7d32' stroke-width='1.6' stroke-dasharray='3,3'/>"
    body += text(ax.X(ts) + 4, ax.Y(1.0), "t* (bề mặt)", 13, "#2e7d32", "start")
    body += f"<circle cx='{ax.X(tn):.1f}' cy='{ax.Y(1):.1f}' r='4' fill='{A}'/>"
    body += text(ax.X(tn) - 26, ax.Y(0.5), f"cực đại tại t={tn:.3f}", 13, A, "end")
    body += text(ax.X(0.83), ax.Y(0.62), "f(t) = −(t − t*)", 13, "#777")
    body += text(250, 30, "Một mặt phẳng, s = 12", 16, P, weight="bold")
    # Panel 2: two slabs (visible and occluded surface)
    def sdf_slab(t, a, b):
        return np.where(t < a, a - t, np.where(t > b, t - b, -np.minimum(t - a, b - t)))
    f2 = np.minimum(sdf_slab(t, 0.3, 0.5), sdf_slab(t, 0.65, 0.85))
    wn2, wu2 = weights_along(f2, t, 60.0)
    ax2 = Axes(530, 50, 360, 220, (0, 1), (-0.1, 1.08))
    body += ax2.frame("độ sâu t dọc tia", "trọng số (chuẩn hóa)")
    body += poly(ax2.X(t), ax2.Y(np.clip(f2 * 2 + 0.35, -0.1, 1.08)), "#999", 1.6, "5,4")
    body += poly(ax2.X(t), ax2.Y(wn2 / wn2.max()), A, 2.8)
    body += poly(ax2.X(t), ax2.Y(wu2 / wu2.max()), P, 2.8)
    for tt, lab in [(0.3, "mặt thấy được"), (0.65, "mặt bị che")]:
        body += f"<line x1='{ax2.X(tt):.1f}' y1='{ax2.Y(-0.1):.1f}' x2='{ax2.X(tt):.1f}' y2='{ax2.Y(1.08):.1f}' stroke='#2e7d32' stroke-width='1.4' stroke-dasharray='3,3'/>"
        body += text(ax2.X(tt) + 4, ax2.Y(1.0), lab, 13, "#2e7d32", "start")
    body += text(710, 30, "Hai lớp vật thể, s = 60", 16, P, weight="bold")
    # legend
    body += f"<line x1='250' y1='338' x2='290' y2='338' stroke='{A}' stroke-width='3'/>" + text(298, 343, "ngây thơ: w = T·φₛ(f)", 14, "#222", "start")
    body += f"<line x1='500' y1='338' x2='540' y2='338' stroke='{P}' stroke-width='3'/>" + text(548, 343, "NeuS: w = T·ρ, ρ = max(−Φₛ′/Φₛ, 0)", 14, "#222", "start")
    body += f"<line x1='250' y1='362' x2='290' y2='362' stroke='#999' stroke-width='2' stroke-dasharray='5,4'/>" + text(298, 367, "SDF dọc tia (tỉ lệ minh họa)", 14, "#222", "start")
    save("neus/bias.svg", svg(940, 380, body, "So sánh trọng số ngây thơ và trọng số NeuS dọc một tia"))
    return tn


# ---------------- NeRF: hierarchical sampling ----------------
def nerf_sampling():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 2000)
    sigma = 8 * np.exp(-((t - 0.42) / 0.03) ** 2) + 25 * np.exp(-((t - 0.7) / 0.03) ** 2)
    Nc, Nf = 16, 32
    edges = np.linspace(0, 1, Nc + 1)
    tc = edges[:-1] + rng.uniform(0, 1, Nc) / Nc
    sc = np.interp(tc, t, sigma)
    delta = np.diff(np.append(tc, 1.0))
    alpha = 1 - np.exp(-sc * delta)
    T = np.cumprod(np.append(1, 1 - alpha[:-1]))
    w = T * alpha
    pdf = w / w.sum()
    cdf = np.concatenate([[0], np.cumsum(pdf)])
    bins = np.append(tc, 1.0)
    u = (np.arange(Nf) + 0.5) / Nf
    idx = np.searchsorted(cdf, u, side="right") - 1
    idx = np.clip(idx, 0, Nc - 1)
    frac = (u - cdf[idx]) / np.maximum(cdf[idx + 1] - cdf[idx], 1e-9)
    tf = bins[idx] + frac * (bins[idx + 1] - bins[idx])
    body = ""
    X = lambda v: 60 + v * 800
    # density curve
    body += text(460, 26, "Lấy mẫu phân cấp: mẫu thô → phân phối trọng số → mẫu tinh", 17, P, weight="bold")
    ys = 150 - sigma / sigma.max() * 90
    body += poly(X(t), ys, "#999", 1.8)
    body += text(870, 70, "σ thật (ẩn)", 13, "#777", "end")
    body += f"<line x1='{X(0)}' y1='150' x2='{X(1)}' y2='150' stroke='#444' stroke-width='1.5'/>"
    body += text(24, 154, "thô", 14, P, "start", "bold")
    for e in edges:
        body += f"<line x1='{X(e):.1f}' y1='145' x2='{X(e):.1f}' y2='155' stroke='#bbb'/>"
    for v in tc:
        body += f"<circle cx='{X(v):.1f}' cy='150' r='4.5' fill='{P}'/>"
    # weights bars
    base = 270
    body += f"<line x1='{X(0)}' y1='{base}' x2='{X(1)}' y2='{base}' stroke='#444' stroke-width='1.5'/>"
    for i in range(Nc):
        h = pdf[i] / pdf.max() * 80
        body += f"<rect x='{X(bins[i]):.1f}' y='{base-h:.1f}' width='{X(bins[i+1])-X(bins[i]):.1f}' height='{h:.1f}' fill='#f7efe6' stroke='{A}'/>"
    body += text(24, base - 30, "ŵᵢ", 16, A, "start", "bold")
    body += text(70, base - 62, "PDF hằng từng khúc ŵᵢ = wᵢ / Σⱼ wⱼ", 13, A, "start")
    # fine samples
    body += f"<line x1='{X(0)}' y1='340' x2='{X(1)}' y2='340' stroke='#444' stroke-width='1.5'/>"
    body += text(24, 344, "tinh", 14, A, "start", "bold")
    for v in tc:
        body += f"<circle cx='{X(v):.1f}' cy='340' r='4' fill='{P}' opacity='.45'/>"
    for v in tf:
        body += f"<circle cx='{X(v):.1f}' cy='340' r='4' fill='{A}'/>"
    body += text(460, 378, f"Minh họa với {Nc} mẫu thô và {Nf} mẫu tinh; NeRF dùng Nc = 64, Nf = 128 (mạng tinh nhận 192 điểm).", 13.5, "#444")
    save("nerf/sampling.svg", svg(920, 392, body, "Lấy mẫu phân cấp của NeRF"))


# ---------------- RadFoam geometry ----------------
def bounded_voronoi(pts, box=(0, 0, 1, 1)):
    x0, y0, x1, y1 = box
    mirrored = [pts]
    mirrored.append(np.c_[2 * x0 - pts[:, 0], pts[:, 1]])
    mirrored.append(np.c_[2 * x1 - pts[:, 0], pts[:, 1]])
    mirrored.append(np.c_[pts[:, 0], 2 * y0 - pts[:, 1]])
    mirrored.append(np.c_[pts[:, 0], 2 * y1 - pts[:, 1]])
    allp = np.vstack(mirrored)
    vor = Voronoi(allp)
    cells = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        cells.append(vor.vertices[reg])
    return cells


def order_poly(v):
    c = v.mean(0)
    ang = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])
    return v[np.argsort(ang)]


def radfoam_dual():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.08, 0.92, (16, 2))
    S, ox, oy = 360, 40, 40
    M = lambda p: (ox + p[0] * S, oy + (1 - p[1]) * S)
    body = ""
    cells = bounded_voronoi(pts)
    for c in cells:
        c = order_poly(c)
        body += "<polygon points='" + " ".join("%.1f,%.1f" % M(q) for q in c) + f"' fill='#edf4ff' stroke='{P}' stroke-width='2.2'/>"
    tri = Delaunay(pts)
    for s in tri.simplices:
        q = [M(pts[k]) for k in s]
        body += "<polygon points='" + " ".join("%.1f,%.1f" % v for v in q) + f"' fill='none' stroke='{A}' stroke-width='1.3' stroke-dasharray='5,3'/>"
    # one circumcircle
    s = tri.simplices[len(tri.simplices) // 2]
    a, b, c = pts[s]
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    r = np.linalg.norm(a - [ux, uy])
    cx, cy = M((ux, uy))
    body += f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{r*S:.1f}' fill='none' stroke='#2e7d32' stroke-width='1.8'/>"
    body += f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='#2e7d32'/>"
    for p in pts:
        x, y = M(p)
        body += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.5' fill='#c62828'/>"
    body += f"<rect x='{ox}' y='{oy}' width='{S}' height='{S}' fill='none' stroke='#444'/>"
    lx = 430
    body += f"<circle cx='{lx}' cy='90' r='5' fill='#c62828'/>" + text(lx + 14, 95, "site pᵢ (điểm sinh)", 15, "#222", "start")
    body += f"<line x1='{lx-10}' y1='125' x2='{lx+10}' y2='125' stroke='{P}' stroke-width='2.4'/>" + text(lx + 14, 130, "mặt ô Voronoi", 15, "#222", "start")
    body += f"<line x1='{lx-10}' y1='160' x2='{lx+10}' y2='160' stroke='{A}' stroke-width='1.5' stroke-dasharray='5,3'/>" + text(lx + 14, 165, "cạnh Delaunay (kề nhau)", 15, "#222", "start")
    body += f"<circle cx='{lx}' cy='195' r='9' fill='none' stroke='#2e7d32' stroke-width='1.8'/>" + text(lx + 14, 200, "đường tròn ngoại tiếp rỗng", 15, "#222", "start")
    body += text(lx + 14, 222, "tâm = đỉnh Voronoi", 15, "#222", "start")
    body += text(lx - 10, 280, "Mỗi cạnh Delaunay (i, j) ↔ một mặt", 15, P, "start", "bold")
    body += text(lx - 10, 302, "Voronoi nằm trên mặt trung trực", 15, P, "start", "bold")
    body += text(lx - 10, 324, "của đoạn pᵢpⱼ.", 15, P, "start", "bold")
    save("radfoam/delaunay_voronoi.svg", svg(770, 440, body, "Tam giác hóa Delaunay và sơ đồ Voronoi đối ngẫu"))


def radfoam_raywalk():
    rng = np.random.default_rng(11)
    pts = rng.uniform(0.04, 0.96, (34, 2))
    tri = Delaunay(pts)
    indptr, nbr = tri.vertex_neighbor_vertices
    N = lambda i: nbr[indptr[i]:indptr[i + 1]]
    o = np.array([0.02, 0.18])
    d = np.array([1.0, 0.55]); d /= np.linalg.norm(d)
    i = int(np.argmin(np.linalg.norm(pts - o, axis=1)))
    t0, seq, ts = 0.0, [i], [0.0]
    tmax = (1 - o[0]) / d[0]
    tmax = min(tmax, (1 - o[1]) / d[1])
    while True:
        best, bj = np.inf, None
        for j in N(i):
            n = pts[j] - pts[i]
            den = d @ n
            if den <= 1e-12:
                continue
            m = 0.5 * (pts[i] + pts[j])
            tj = ((m - o) @ n) / den
            if t0 - 1e-9 < tj < best:
                best, bj = tj, j
        if bj is None or best > tmax:
            ts.append(tmax); break
        ts.append(best); t0 = best; i = int(bj); seq.append(i)
    S, ox, oy = 440, 30, 30
    M = lambda p: (ox + p[0] * S, oy + (1 - p[1]) * S)
    sig = rng.uniform(0, 1, len(pts))
    body = ""
    cells = bounded_voronoi(pts)
    for k, c in enumerate(cells):
        c = order_poly(c)
        fill = "#f7efe6" if k in seq else "#fafafa"
        body += "<polygon points='" + " ".join("%.1f,%.1f" % M(q) for q in c) + f"' fill='{fill}' stroke='{P}' stroke-width='1.6'/>"
    for p in pts:
        x, y = M(p)
        body += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='#555'/>"
    cols = [A, P]
    for k in range(len(seq)):
        a, b = o + ts[k] * d, o + ts[k + 1] * d
        (x1, y1), (x2, y2) = M(a), M(b)
        body += f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{cols[k%2]}' stroke-width='4.5'/>"
        if 0 < k < len(seq):
            body += f"<circle cx='{x1:.1f}' cy='{y1:.1f}' r='4' fill='white' stroke='#222' stroke-width='1.4'/>"
        if k in (1, 3):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            body += text(mx - 2, my + 22, f"δ{['₀','₁','₂','₃','₄'][k]}", 16, cols[k % 2], weight="bold")
    x, y = M(o)
    body += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='6' fill='#c62828'/>" + text(x + 4, y + 22, "o", 16, "#c62828", "start", "bold")
    body += f"<rect x='{ox}' y='{oy}' width='{S}' height='{S}' fill='none' stroke='#444'/>"
    lx = 500
    lines = [("Tia đi qua {} ô; mỗi ô là một đoạn".format(len(seq)), P, "bold"),
             ("hằng số (σₙ, cₙ) dài δₙ.", P, "bold"),
             ("", "#222", "normal"),
             ("Tại ô i: xét mọi láng giềng j ∈ N(i),", "#222", "normal"),
             ("giao tia với mặt trung trực, chỉ lấy", "#222", "normal"),
             ("mặt có d·(pⱼ − pᵢ) > 0 và t nhỏ nhất.", "#222", "normal"),
             ("", "#222", "normal"),
             ("Không cần BVH; chi phí mỗi bước", "#222", "normal"),
             ("tỉ lệ với bậc của ô (≈ 15 trong 3D).", "#222", "normal")]
    for k, (s_, c_, w_) in enumerate(lines):
        body += text(lx, 70 + 26 * k, s_, 15, c_, "start", w_)
    save("radfoam/ray_walk.svg", svg(820, 500, body, "Dò tia qua các ô Voronoi theo thuật toán duyệt mặt"))
    return len(seq)


def radfoam_edgeflip():
    body = ""
    hs = [1.35, 1.0, 0.75]
    labels = ["h > 1: A, B là láng giềng", "h = 1: bốn điểm đồng viên", "h < 1: C, D là láng giềng"]
    for k, h in enumerate(hs):
        base = np.array([[-1, 0], [1, 0], [0, h], [0, -h]], float)
        far = np.array([[x, y] for x in (-6, 0, 6) for y in (-6, 0, 6) if not (x == 0 and y == 0)], float)
        allp = np.vstack([base, far])
        vor = Voronoi(allp)
        ox = 20 + k * 300
        S = 70
        M = lambda p, ox=ox: (ox + 140 + p[0] * S, 170 - p[1] * S)
        body += f"<clipPath id='c{k}'><rect x='{ox}' y='40' width='280' height='260'/></clipPath><g clip-path='url(#c{k})'>"
        fills = ["#edf4ff", "#edf4ff", "#f7efe6", "#f7efe6"]
        for i in range(4):
            reg = vor.regions[vor.point_region[i]]
            if -1 in reg:
                continue
            c = vor.vertices[reg]
            body += "<polygon points='" + " ".join("%.1f,%.1f" % M(q) for q in c) + f"' fill='{fills[i]}' stroke='{P}' stroke-width='2.4'/>"
        # Delaunay diagonal
        if h > 1.0001:
            (x1, y1), (x2, y2) = M(base[0]), M(base[1])
        elif h < 0.9999:
            (x1, y1), (x2, y2) = M(base[2]), M(base[3])
        else:
            x1 = None
        if x1 is not None:
            body += f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{A}' stroke-width='1.6' stroke-dasharray='5,3'/>"
        else:
            cx, cy = M((0, 0))
            body += f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{S:.1f}' fill='none' stroke='#2e7d32' stroke-width='1.8' stroke-dasharray='4,3'/>"
        for i, nm in enumerate("ABCD"):
            x, y = M(base[i])
            body += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='#c62828'/>"
            dx = -16 if nm == "A" else (16 if nm == "B" else 14)
            dy = 5 if nm in "AB" else (-8 if nm == "C" else 18)
            body += text(x + dx, y + dy, nm, 15, "#c62828", weight="bold")
        body += "</g>"
        body += f"<rect x='{ox}' y='40' width='280' height='260' fill='none' stroke='#999'/>"
        body += text(ox + 140, 28, labels[k], 15, P, weight="bold")
    body += text(460, 330, "Khi h → 1, mặt Voronoi chung của A và B co về độ dài 0 rồi mặt chung của C và D mới xuất hiện:", 14.5, "#333")
    body += text(460, 352, "kết nối thay đổi rời rạc, nhưng hình dạng các ô và trường (σ, c) trong không gian vẫn biến thiên liên tục.", 14.5, "#333")
    save("radfoam/edge_flip.svg", svg(920, 366, body, "Lật cạnh Delaunay và tính liên tục của ô Voronoi"))


if __name__ == "__main__":
    neus_logistic()
    tn = neus_bias()
    print("naive peak", tn)
    nerf_sampling()
    radfoam_dual()
    print("cells crossed", radfoam_raywalk())
    radfoam_edgeflip()
