"""Sơ đồ SVG cho deck Plenoxels."""
import numpy as np

from make_diagrams import D, P, A, G, esc
from make_svgs import Axes, poly, text, svg, save

GREY = "#bbb"


def sparse_grid():
    d = D(1000, 340, "Lưới chỉ số dày đặc trỏ vào bảng dữ liệu thưa")
    nx, ny, c, x0, y0 = 9, 7, 34, 40, 70
    occ = {(3, 2), (4, 2), (5, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (3, 4), (4, 4), (5, 4), (4, 5)}
    idx = {}
    for j in range(ny):
        for i in range(nx):
            on = (i, j) in occ
            if on:
                idx[(i, j)] = len(idx)
            fill = "#e6eefc" if on else "#f4f4f4"
            d.raw(f"<rect x='{x0+i*c}' y='{y0+j*c}' width='{c}' height='{c}' fill='{fill}' stroke='#999' stroke-width='1'/>")
            lab = str(idx[(i, j)]) if on else "−1"
            col = P if on else "#999"
            d.raw(f"<text x='{x0+i*c+c/2}' y='{y0+j*c+c/2+5}' font-size='14' text-anchor='middle' fill='{col}'>{lab}</text>")
    d.raw(f"<ellipse cx='{x0+4.5*c}' cy='{y0+3.6*c}' rx='{2.7*c}' ry='{1.9*c}' fill='none' stroke='{A}' stroke-width='2.5' stroke-dasharray='6,4'/>")
    d.text(x0 + nx * c / 2, y0 - 22, "links[i, j, k] (int32, dày đặc)", 16, P, weight="bold")
    d.text(x0 + nx * c / 2, y0 + ny * c + 28, "−1: ô rỗng, coi σ = 0 và SH = 0", 15, "#555")
    d.text(x0 + nx * c / 2, y0 + ny * c + 50, "nét đứt: vùng có vật thể sau khi tỉa", 15, A)

    tx, ty, rw, rh = 560, 70, 400, 30
    d.text(tx + rw / 2, ty - 22, "Bảng dữ liệu (chỉ voxel bị chiếm)", 16, P, weight="bold")
    d.raw(f"<rect x='{tx}' y='{ty}' width='{rw}' height='{rh}' fill='#f7efe6' stroke='{A}'/>")
    d.raw(f"<line x1='{tx+60}' y1='{ty}' x2='{tx+60}' y2='{ty+7*rh}' stroke='#ccc'/>")
    d.raw(f"<line x1='{tx+130}' y1='{ty}' x2='{tx+130}' y2='{ty+7*rh}' stroke='#ccc'/>")
    d.raw(f"<text x='{tx+30}' y='{ty+20}' font-size='14' text-anchor='middle' fill='{A}' font-weight='bold'>chỉ số</text>")
    d.raw(f"<text x='{tx+95}' y='{ty+20}' font-size='15' text-anchor='middle' fill='{A}' font-weight='bold'>σ</text>")
    d.raw(f"<text x='{tx+265}' y='{ty+20}' font-size='15' text-anchor='middle' fill='{A}' font-weight='bold'>SH: 3 kênh × 9 hệ số = 27</text>")
    rows = ["0", "1", "2", "3", "…", "11"]
    for r, lab in enumerate(rows):
        y = ty + (r + 1) * rh
        d.raw(f"<rect x='{tx}' y='{y}' width='{rw}' height='{rh}' fill='white' stroke='#bbb'/>")
        d.raw(f"<text x='{tx+30}' y='{y+20}' font-size='14' text-anchor='middle' fill='{P}'>{lab}</text>")
        if lab != "…":
            d.raw(f"<text x='{tx+95}' y='{y+20}' font-size='14' text-anchor='middle'>σ</text>")
            for k in range(9):
                for ch, colr in enumerate(["#c0504d", "#4f9a4f", "#4a6fc0"]):
                    d.raw(f"<rect x='{tx+140+k*28+ch*8}' y='{y+9}' width='7' height='12' fill='{colr}' opacity='0.75'/>")
    for (i, j), k in [((3, 2), 0), ((4, 2), 1), ((5, 2), 2)]:
        d.path(f"M {x0+i*c+c/2} {y0+j*c+4} C {x0+i*c+c/2} {y0-8}, {tx-60} {ty+(k+1)*rh+15}, {tx-4} {ty+(k+1)*rh+15}", "#7a86b8", "4,3")
    d.text(tx + rw / 2, ty + 7 * rh + 30, "Khi huấn luyện: tham số + gradient + trạng thái RMSProp", 15, "#444")
    d.text(tx + rw / 2, ty + 7 * rh + 52, "→ 28 × 3 × 4 byte = 336 byte mỗi voxel bị chiếm", 15, "#444")
    d.save("plenoxels/sparse_grid.svg")


def trilinear():
    b = ""
    ax = Axes(70, 50, 400, 200, (0, 6), (0, 1.1))
    b += ax.frame("vị trí dọc tia (đơn vị voxel)", "giá trị σ")
    f = lambda x: 0.15 + 0.75 * np.exp(-((x - 3.1) / 1.05) ** 2)
    xs = np.linspace(0, 6, 400)
    b += poly(ax.X(xs), ax.Y(f(xs)), "#999", 2, "5,4")
    nodes = np.arange(0, 7)
    vals = f(nodes)
    sx, sy = [], []
    for k in range(len(nodes)):
        lo, hi = max(nodes[k] - 0.5, 0), min(nodes[k] + 0.5, 6)
        sx += [lo, hi]
        sy += [vals[k], vals[k]]
    b += poly(ax.X(sx), ax.Y(sy), A, 2.5)
    b += poly(ax.X(nodes), ax.Y(vals), P, 3)
    for x, v in zip(nodes, vals):
        b += f"<circle cx='{ax.X(x):.1f}' cy='{ax.Y(v):.1f}' r='5' fill='{P}'/>"
    b += text(270, 30, "Lân cận gần nhất so với ba tuyến tính (1D)", 16, P, weight="bold")
    ly = 305
    for col, lab, dash in [("#999", "hàm thật", "5,4"), (A, "gần nhất: bậc thang, đạo hàm theo vị trí bằng 0 trong ô", None), (P, "ba tuyến tính: liên tục, đạo hàm khác 0", None)]:
        ds = f" stroke-dasharray='{dash}'" if dash else ""
        b += f"<line x1='70' y1='{ly}' x2='100' y2='{ly}' stroke='{col}' stroke-width='3'{ds}/>"
        b += text(108, ly + 5, lab, 15, "#222", "start")
        ly += 24

    ox, oy, s = 650, 62, 205
    u, v = 0.45, 0.35
    px, py = ox + u * s, oy + (1 - v) * s
    b += text(ox + s / 2, 24, "Trọng số trong một ô (2D)", 16, P, weight="bold")
    rects = [(ox, oy, px - ox, py - oy, "#fbe7d6", "ω₁₀ = u(1−v)"), (px, oy, ox + s - px, py - oy, "#dbe6fb", "ω₀₀ = (1−u)(1−v)"),
             (ox, py, px - ox, oy + s - py, "#f3e3f3", "ω₁₁ = uv"), (px, py, ox + s - px, oy + s - py, "#e3f1e3", "ω₀₁ = (1−u)v")]
    for (x, y, w, h, colr, lab) in rects:
        b += f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' fill='{colr}' stroke='none'/>"
        l1, l2 = lab.split(" = ")
        b += text(x + w / 2, y + h / 2 - 3, l1, 15, "#333", weight="bold") + text(x + w / 2, y + h / 2 + 16, l2, 14, "#333")
    b += f"<rect x='{ox}' y='{oy}' width='{s}' height='{s}' fill='none' stroke='#444' stroke-width='2'/>"
    corners = [(ox, oy + s, "f₀₀"), (ox + s, oy + s, "f₁₀"), (ox, oy, "f₀₁"), (ox + s, oy, "f₁₁")]
    for (x, y, lab) in corners:
        b += f"<circle cx='{x}' cy='{y}' r='7' fill='{P}'/>"
        b += text(x + (-22 if x == ox else 22), y + (22 if y > oy else -10), lab, 16, P, weight="bold")
    b += f"<circle cx='{px:.1f}' cy='{py:.1f}' r='7' fill='{A}'/>" + text(px + 16, py - 10, "x", 17, A, "start", "bold", True)
    b += text(ox + s / 2, oy + s + 50, "Trọng số của góc = diện tích phần đối diện", 15, "#444")
    b += text(ox + s / 2, oy + s + 72, "3D: 8 góc, trọng số = thể tích hộp đối diện", 15, "#444")
    b += text(ox + s / 2, oy + s + 94, "(u, v): tọa độ của x trong ô, gốc tại f₀₀", 15, "#444")
    save("plenoxels/trilinear.svg", svg(1000, 380, b, "Nội suy lân cận gần nhất và ba tuyến tính; trọng số theo diện tích đối diện"))


def coarse_to_fine():
    d = D(1000, 330, "Lịch thô đến mịn cho cảnh 360 độ")
    xs = [30, 250, 470, 690]
    labels = [("128³", "bước 0"), ("256³", "bước 25 600"), ("512³", "bước 51 200"), ("640³", "bước 76 800")]
    for k, (x, (r, st)) in enumerate(zip(xs, labels)):
        d.box(x, 60, 170, 80, [f"Lưới {r}", st], "data", 18)
        if k < 3:
            d.arrow(x + 170, 100, x + 220, 100)
    d.text(935, 92, "kết thúc", 15, "#444")
    d.text(935, 112, "102 400", 15, "#444")
    d.arrow(860, 100, 895, 100)
    d.text(500, 32, "Mỗi 25 600 bước: tỉa → giãn nở → chia đôi mỗi chiều → tối ưu tiếp", 16, P, weight="bold")

    y0, c = 190, 16
    def mask(ox, cells, color, title):
        for j in range(6):
            for i in range(6):
                on = (i, j) in cells
                d.raw(f"<rect x='{ox+i*c}' y='{y0+j*c}' width='{c}' height='{c}' fill='{color if on else '#f4f4f4'}' stroke='#aaa' stroke-width='0.8'/>")
        d.text(ox + 3 * c, y0 + 6 * c + 24, title, 15, "#333")
    core = {(2, 2), (3, 2), (2, 3), (3, 3), (3, 4)}
    dil = set()
    for (i, j) in core:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if 0 <= i + di < 6 and 0 <= j + dj < 6:
                    dil.add((i + di, j + dj))
    mask(60, core, "#7a8fd0", "max trọng số ≥ ngưỡng")
    d.arrow(170, y0 + 48, 230, y0 + 48)
    mask(250, dil, "#b9c6ee", "giãn nở (giữ lân cận)")
    d.arrow(360, y0 + 48, 420, y0 + 48)
    ox = 440
    for j in range(12):
        for i in range(12):
            on = (i // 2, j // 2) in dil
            d.raw(f"<rect x='{ox+i*8}' y='{y0+j*8}' width='8' height='8' fill='{'#dbe4fa' if on else '#f7f7f7'}' stroke='#bbb' stroke-width='0.5'/>")
    d.text(ox + 48, y0 + 96 + 24, "chia đôi, nội suy giá trị", 15, "#333")
    d.box(620, 180, 360, 110, ["Ngưỡng tỉa", "w_v = max_r Tᵢ(1 − e^(−σᵢδᵢ)) trên mọi tia", "360°: 1,28 / D_z · tổng hợp: 0,256 / D_z", "hướng thẳng: σ < 5 (vùng rìa ít góc nhìn)"], "op", 16)
    d.save("plenoxels/coarse_to_fine.svg")


def msi():
    b = ""
    cx, cy, R = 250, 180, 55
    K = 8
    for k in range(K):
        r = 1 / (1 - k / K)
        if r > 5.2:
            break
        rr = R * r
        col = "#2f3e7a" if k == 0 else "#8a96c8"
        b += f"<circle cx='{cx}' cy='{cy}' r='{rr:.1f}' fill='none' stroke='{col}' stroke-width='{2.5 if k == 0 else 1.5}'/>"
    b += f"<rect x='{cx-R*0.7:.1f}' y='{cy-R*0.7:.1f}' width='{R*1.4:.1f}' height='{R*1.4:.1f}' fill='#e6eefc' stroke='{P}' stroke-width='2'/>"
    for t in (1, 2):
        b += f"<line x1='{cx-R*0.7+t*R*1.4/3:.1f}' y1='{cy-R*0.7:.1f}' x2='{cx-R*0.7+t*R*1.4/3:.1f}' y2='{cy+R*0.7:.1f}' stroke='{P}' stroke-width='0.8'/>"
        b += f"<line x1='{cx-R*0.7:.1f}' y1='{cy-R*0.7+t*R*1.4/3:.1f}' x2='{cx+R*0.7:.1f}' y2='{cy-R*0.7+t*R*1.4/3:.1f}' stroke='{P}' stroke-width='0.8'/>"
    b += f"<rect x='0' y='0' width='1000' height='14' fill='white'/><rect x='0' y='346' width='1000' height='14' fill='white'/>"
    b += f"<rect x='505' y='0' width='495' height='360' fill='white'/>"
    b += f"<line x1='{cx-20}' y1='{cy+10}' x2='{cx+240}' y2='{cy-150}' stroke='{A}' stroke-width='2.5'/>"
    b += f"<circle cx='{cx-20}' cy='{cy+10}' r='6' fill='{A}'/>" + text(cx - 20, cy + R + 50, "camera (trong cảnh)", 14, A)
    b += f"<line x1='{cx-20}' y1='{cy+16}' x2='{cx-20}' y2='{cy+R+36}' stroke='{A}' stroke-width='1' stroke-dasharray='3,3'/>"
    b += text(cx + 28, cy - 28, "lưới", 14, P, weight="bold")
    b += text(cx + 20, cy + R + 18, "r = 1", 14, P)
    b += text(cx - 5, 334, "càng xa, các mặt cầu càng thưa (đều theo 1/r)", 15, "#444")
    body = [("Tiền cảnh", "lưới voxel thưa trong mặt cầu đơn vị"),
            ("Hậu cảnh", "K = 64 mặt cầu, 1/r_k = 1 − k/K"),
            ("Mỗi mặt cầu", "ảnh equirect 2048 × 1024, (σ, RGB)"),
            ("Nội suy", "ba tuyến tính theo (θ, φ, 1/r)"),
            ("Tỉa", "chỉ theo σ; TV quấn quanh mép ảnh")]
    y = 60
    b += text(520, 34, "Mô hình nền đa mặt cầu (MSI)", 17, P, "start", "bold")
    for h, t in body:
        b += text(520, y + 22, h, 15, A, "start", "bold") + text(640, y + 22, t, 15, "#222", "start")
        y += 50
    b += text(520, y + 30, "Cảnh hướng thẳng: dùng NDC thay cho MSI", 15, G, "start", "bold")
    save("plenoxels/msi.svg", svg(1000, 360, b, "Tiền cảnh là lưới voxel trong mặt cầu đơn vị; hậu cảnh là 64 mặt cầu đồng tâm đều theo nghịch đảo bán kính"))


def pipeline():
    d = D(1000, 300, "Vòng tối ưu Plenoxels")
    d.box(10, 90, 150, 90, ["Ảnh + tư thế", "5000 tia mỗi batch", "chọn ngẫu nhiên"], "data")
    d.arrow(160, 135, 195, 135)
    d.box(195, 90, 165, 90, ["Ray marching", "bước ½ voxel", "bỏ qua ô rỗng"], "op")
    d.arrow(360, 135, 395, 135)
    d.box(395, 90, 165, 90, ["Nội suy 8 góc", "σ, 27 hệ số SH", "→ (σᵢ, cᵢ)"], "data")
    d.arrow(560, 135, 595, 135)
    d.box(595, 90, 170, 90, ["Kết xuất thể tích", "Ĉ = Σ Tᵢ(1−e^(−σᵢδᵢ))cᵢ", "cùng kernel CUDA"], "op")
    d.arrow(765, 135, 800, 135)
    d.box(800, 90, 185, 90, ["MSE + TV (1% voxel)", "+ Cauchy, beta", "(cảnh thật)"], "loss")
    d.path("M 892 180 L 892 250 L 600 250", "#a33a1f", "6,4")
    d.box(390, 222, 210, 58, ["RMSProp thưa", "chỉ voxel có gradient"], "net", 15)
    d.path("M 390 250 L 110 250 L 110 185", G)
    d.text(250, 272, "mỗi 25 600 bước: tỉa + chia đôi lưới", 14, G)
    d.text(500, 40, "Không có mạng nơ-ron: gradient đi thẳng vào σ và hệ số SH của 8 voxel quanh mỗi mẫu", 16, P, weight="bold")
    d.save("plenoxels/pipeline.svg")


if __name__ == "__main__":
    sparse_grid(); trilinear(); coarse_to_fine(); msi(); pipeline()
    print("ok")
