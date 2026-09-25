"""Sơ đồ khối (pipeline, kiến trúc) cho bốn deck."""
import numpy as np

OUT = "/mnt/data/research/slide/nghien-cuu-3d/img/"
P, A, G = "#2f3e7a", "#b15a2b", "#2e7d32"
KIND = {"data": ("#edf4ff", P), "op": ("#f7efe6", A), "loss": ("#fdecea", "#a33a1f"),
        "net": ("#eef6ee", G), "plain": ("#fafafa", "#888")}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class D:
    def __init__(self, w, h, title):
        self.w, self.h, self.title, self.b = w, h, title, []

    def box(self, x, y, w, h, lines, kind="op", size=15, bold_first=True, r=10):
        fill, stroke = KIND[kind]
        self.b.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' fill='{fill}' stroke='{stroke}' stroke-width='2'/>")
        n = len(lines)
        lh = size * 1.3
        y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
        for k, s in enumerate(lines):
            wt = "bold" if (k == 0 and bold_first) else "normal"
            col = stroke if (k == 0 and bold_first) else "#222"
            sz = size if k == 0 else size * 0.88
            self.b.append(f"<text x='{x+w/2}' y='{y0+k*lh:.1f}' font-size='{sz:.1f}' text-anchor='middle' font-weight='{wt}' fill='{col}'>{esc(s)}</text>")
        return (x, y, w, h)

    def arrow(self, x1, y1, x2, y2, label=None, color="#555", dy=-8, size=13, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.b.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='2' marker-end='url(#ar)'{d}/>")
        if label:
            self.b.append(f"<text x='{(x1+x2)/2:.1f}' y='{(y1+y2)/2+dy:.1f}' font-size='{size}' text-anchor='middle' fill='{color}'>{esc(label)}</text>")

    def path(self, d, color="#555", dash=None):
        ds = f" stroke-dasharray='{dash}'" if dash else ""
        self.b.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' marker-end='url(#ar)'{ds}/>")

    def text(self, x, y, s, size=14, color="#222", anchor="middle", weight="normal", italic=False):
        st = " font-style='italic'" if italic else ""
        self.b.append(f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' fill='{color}'{st}>{esc(s)}</text>")

    def raw(self, s):
        self.b.append(s)

    def save(self, path):
        s = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {self.w} {self.h}' font-family='Helvetica,Arial,sans-serif'>"
             f"<title>{esc(self.title)}</title><defs><marker id='ar' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
             f"<path d='M0,0 L10,5 L0,10 z' fill='#555'/></marker></defs><rect width='{self.w}' height='{self.h}' fill='white'/>"
             + "".join(self.b) + "</svg>")
        open(OUT + path, "w").write(s)


def nerf_pipeline():
    d = D(1000, 300, "Pipeline huấn luyện NeRF")
    d.box(10, 90, 150, 90, ["Ảnh + tư thế", "(Iₖ, Kₖ, Rₖ, tₖ)", "từ COLMAP"], "data")
    d.arrow(160, 135, 200, 135)
    d.box(200, 90, 150, 90, ["Tia r(t)=o+td", "qua từng pixel", "4096 tia/batch"], "op")
    d.arrow(350, 135, 390, 135)
    d.box(390, 90, 150, 90, ["Điểm mẫu", "(xᵢ, d), i=1..N", "phân tầng + phân cấp"], "op")
    d.arrow(540, 135, 580, 135)
    d.box(580, 80, 150, 110, ["MLP F_Θ", "γ(x), γ(d)", "→ (cᵢ, σᵢ)"], "net")
    d.arrow(730, 135, 770, 135)
    d.box(770, 90, 210, 90, ["Kết xuất thể tích", "Ĉ(r)=Σ Tᵢ(1−e^(−σᵢδᵢ))cᵢ"], "op")
    d.arrow(875, 180, 875, 225)
    d.box(720, 225, 260, 60, ["Mất mát ‖Ĉ(r) − C(r)‖²", "so với màu pixel thật"], "loss", 14)
    d.path("M 720 255 L 655 255 L 655 195", "#a33a1f", "6,4")
    d.text(560, 272, "lan truyền ngược qua phép tổng khả vi", 13, "#a33a1f")
    d.text(500, 40, "Toàn bộ chuỗi khả vi: chỉ cần ảnh 2D và tư thế camera để học Θ", 16, P, weight="bold")
    d.save("nerf/pipeline.svg")


def nerf_mlp():
    d = D(1000, 330, "Kiến trúc MLP của NeRF")
    d.box(10, 120, 110, 70, ["γ(x)", "60 chiều"], "data")
    xs = [150 + k * 62 for k in range(8)]
    for k, x in enumerate(xs):
        d.box(x, 110, 48, 90, [str(256)], "net", 13, bold_first=False, r=6)
        if k < 7:
            d.arrow(x + 48, 155, x + 62, 155)
    d.arrow(120, 155, 150, 155)
    d.text(xs[0] + 24, 100, "lớp 1", 12, "#555")
    d.text(xs[7] + 24, 100, "lớp 8", 12, "#555")
    d.path(f"M 65 120 L 65 60 L {xs[4]+24} 60 L {xs[4]+24} 108", A, "5,4")
    d.text((65 + xs[4]) / 2 + 40, 52, "nối tắt: γ(x) ghép vào lớp 5", 13, A)
    x8 = xs[7] + 48
    d.arrow(x8, 140, 690, 90)
    d.box(690, 60, 110, 50, ["σ (ReLU)", "chỉ phụ thuộc x"], "op", 13)
    d.arrow(x8, 170, 640, 200)
    d.box(640, 175, 100, 50, ["đặc trưng", "256 chiều"], "net", 13)
    d.box(640, 250, 100, 50, ["γ(d)", "24 chiều"], "data", 13)
    d.arrow(740, 200, 780, 215)
    d.arrow(740, 275, 780, 235)
    d.box(780, 195, 70, 60, ["128"], "net", 13, bold_first=False, r=6)
    d.arrow(850, 225, 880, 225)
    d.box(880, 195, 110, 60, ["RGB c", "(sigmoid)"], "op", 13)
    d.text(500, 26, "8 lớp ReLU × 256 kênh cho hình học, 1 lớp × 128 kênh cho màu phụ thuộc hướng", 15, P, weight="bold")
    d.text(330, 250, "≈ 0,6 triệu tham số mỗi mạng; mạng thô + mạng tinh ≈ 5 MB", 14, "#444")
    d.text(330, 275, "γ(x): L = 10 tần số × (sin, cos) × 3 trục; γ(d): L = 4", 14, "#444")
    d.save("nerf/mlp.svg")


def neus_arch():
    d = D(1000, 330, "Kiến trúc NeuS")
    d.box(10, 120, 120, 70, ["x ∈ ℝ³", "γ(x), L = 6"], "data")
    d.arrow(130, 155, 170, 155)
    d.box(170, 100, 200, 110, ["MLP SDF f_θ", "8 lớp × 256, nối tắt lớp 4", "khởi tạo hình học (cầu)"], "net")
    d.arrow(370, 130, 430, 85)
    d.box(430, 55, 150, 60, ["f(x): SDF", "∇f → pháp tuyến n"], "op", 14)
    d.arrow(370, 180, 430, 205)
    d.box(430, 180, 150, 50, ["đặc trưng z", "256 chiều"], "net", 14)
    d.box(430, 255, 150, 50, ["hướng v", "γ(v), L = 4"], "data", 14)
    d.arrow(580, 85, 640, 180)
    d.arrow(580, 205, 640, 205)
    d.arrow(580, 280, 640, 230)
    d.box(640, 165, 170, 80, ["MLP màu c", "4 lớp × 256", "vào: x, v, n, z"], "net", 14)
    d.arrow(580, 70, 830, 70)
    d.box(830, 40, 160, 70, ["α từ Φₛ(f)", "biến thiên s (1 tham số)"], "op", 14)
    d.arrow(810, 205, 880, 205)
    d.box(880, 175, 110, 60, ["màu cᵢ"], "op", 14)
    d.arrow(910, 110, 910, 175)
    d.box(600, 265, 390, 55, ["Ĉ = Σ Tᵢ αᵢ cᵢ", "nền ngoài cầu đơn vị: NeRF++ (khi không có mask)"], "loss", 14)
    d.arrow(935, 235, 935, 265)
    d.text(500, 26, "Hình học (SDF) và màu tách thành hai mạng; hệ số s học chung với mạng", 15, P, weight="bold")
    d.save("neus/architecture.svg")


def gs_pipeline():
    d = D(1000, 330, "Pipeline huấn luyện 3D Gaussian Splatting")
    d.box(10, 100, 140, 90, ["SfM (COLMAP)", "đám mây điểm", "thưa + camera"], "data")
    d.arrow(150, 145, 190, 145)
    d.box(190, 90, 170, 110, ["Gaussian 3D", "μ, q, s, o, SH", "59 số thực / Gaussian"], "data")
    d.arrow(360, 145, 400, 145)
    d.box(400, 90, 150, 110, ["Chiếu EWA", "Σ′ = J W Σ Wᵀ Jᵀ", "(2D, mỗi camera)"], "op")
    d.arrow(550, 145, 590, 145)
    d.box(590, 90, 180, 110, ["Rasterizer theo ô", "ô 16×16, sắp xếp", "radix theo độ sâu"], "op")
    d.arrow(770, 145, 810, 145)
    d.box(810, 100, 170, 90, ["Ảnh kết xuất", "α-blending", "trước → sau"], "op")
    d.arrow(895, 190, 895, 230)
    d.box(760, 230, 230, 70, ["L = (1−λ)L₁ + λ L_D-SSIM", "λ = 0,2"], "loss", 14)
    d.path("M 760 265 L 320 265 L 320 205", "#a33a1f", "6,4")
    d.text(540, 288, "gradient theo μ, q, s, o, SH (lan truyền ngược qua rasterizer)", 13, "#a33a1f")
    d.box(10, 230, 165, 75, ["Điều khiển mật độ", "clone · split · prune", "mỗi 100 vòng"], "net", 14)
    d.path("M 175 262 L 235 262 L 235 205", G)
    d.text(500, 40, "Biểu diễn tường minh + rasterization khả vi thay cho MLP + ray marching", 16, P, weight="bold")
    d.save("3dgs/pipeline.svg")


def gs_gaussian():
    d = D(1000, 330, "Gaussian 3D dị hướng và phép chiếu xuống ảnh")
    cx, cy = 200, 180
    d.raw(f"<g transform='rotate(-30 {cx} {cy})'><ellipse cx='{cx}' cy='{cy}' rx='130' ry='48' fill='#b15a2b' fill-opacity='.18' stroke='{A}' stroke-width='2'/>"
          f"<ellipse cx='{cx}' cy='{cy}' rx='80' ry='30' fill='#b15a2b' fill-opacity='.22'/>"
          f"<line x1='{cx}' y1='{cy}' x2='{cx+130}' y2='{cy}' stroke='{P}' stroke-width='2.5' marker-end='url(#ar)'/>"
          f"<line x1='{cx}' y1='{cy}' x2='{cx}' y2='{cy-48}' stroke='{P}' stroke-width='2.5' marker-end='url(#ar)'/></g>")
    d.raw(f"<circle cx='{cx}' cy='{cy}' r='4' fill='#222'/>")
    d.text(cx - 10, cy + 22, "μ", 16, "#222", weight="bold")
    d.text(320, 105, "s₁ r₁", 15, P, weight="bold")
    d.text(150, 120, "s₂ r₂", 15, P, weight="bold")
    d.text(200, 300, "Σ = R S Sᵀ Rᵀ,  R = R(q),  S = diag(s)", 16, P, weight="bold")
    d.text(200, 40, "Không gian thế giới", 16, "#222", weight="bold")
    d.arrow(390, 180, 520, 180, "W (camera), J (Jacobi)", "#555", -12, 14)
    d.raw("<rect x='540' y='60' width='280' height='220' fill='#fafafa' stroke='#888'/>")
    for k in range(1, 7):
        d.raw(f"<line x1='{540+k*40}' y1='60' x2='{540+k*40}' y2='280' stroke='#ddd'/>")
    for k in range(1, 6):
        d.raw(f"<line x1='540' y1='{60+k*40}' x2='820' y2='{60+k*40}' stroke='#ddd'/>")
    d.raw(f"<g transform='rotate(-18 680 170)'><ellipse cx='680' cy='170' rx='95' ry='40' fill='#2f3e7a' fill-opacity='.18' stroke='{P}' stroke-width='2'/>"
          f"<ellipse cx='680' cy='170' rx='57' ry='24' fill='#2f3e7a' fill-opacity='.2'/></g>")
    d.raw("<circle cx='680' cy='170' r='4' fill='#222'/>")
    d.text(690, 195, "μ′", 16, "#222", "start", "bold")
    d.text(680, 40, "Mặt phẳng ảnh (lưới pixel)", 16, "#222", weight="bold")
    d.text(680, 305, "Σ′ = (J W Σ Wᵀ Jᵀ) lấy khối 2×2", 16, P, weight="bold")
    d.text(910, 140, "α(x′) =", 15, "#222")
    d.text(910, 165, "o · exp(−½ Δᵀ Σ′⁻¹ Δ)", 14, "#222")
    d.text(910, 190, "Δ = x′ − μ′", 14, "#555")
    d.save("3dgs/gaussian.svg")


def gs_densify():
    d = D(1000, 330, "Clone và split trong điều khiển mật độ thích nghi")
    def curve(x0, y0):
        d.raw(f"<path d='M {x0} {y0+80} C {x0+60} {y0-20}, {x0+140} {y0-20}, {x0+200} {y0+80}' fill='none' stroke='#222' stroke-width='2.2'/>")
    d.text(250, 30, "Thiếu tái tạo (under-reconstruction) → CLONE", 15, A, weight="bold")
    d.text(750, 30, "Tái tạo thừa (over-reconstruction) → SPLIT", 15, P, weight="bold")
    for k, x0 in enumerate([40, 280]):
        curve(x0, 80)
        d.raw(f"<ellipse cx='{x0+60}' cy='{112}' rx='26' ry='12' transform='rotate(-45 {x0+60} 112)' fill='{A}' fill-opacity='.45' stroke='{A}'/>")
        if k == 1:
            d.raw(f"<ellipse cx='{x0+100}' cy='{88}' rx='26' ry='12' fill='{A}' fill-opacity='.45' stroke='{A}'/>")
    d.arrow(245, 130, 275, 130, color=A)
    d.text(250, 215, "Gaussian nhỏ ở vùng thiếu hình học:", 14)
    d.text(250, 237, "sao chép cùng kích thước, dời theo gradient vị trí", 14)
    d.text(250, 259, "(tăng cả tổng thể tích lẫn số Gaussian)", 14, "#555")
    for k, x0 in enumerate([540, 780]):
        curve(x0, 80)
        if k == 0:
            d.raw(f"<ellipse cx='{x0+100}' cy='{95}' rx='95' ry='26' fill='{P}' fill-opacity='.3' stroke='{P}'/>")
        else:
            d.raw(f"<ellipse cx='{x0+55}' cy='{104}' rx='40' ry='13' transform='rotate(-40 {x0+55} 104)' fill='{P}' fill-opacity='.4' stroke='{P}'/>")
            d.raw(f"<ellipse cx='{x0+145}' cy='{104}' rx='40' ry='13' transform='rotate(40 {x0+145} 104)' fill='{P}' fill-opacity='.4' stroke='{P}'/>")
    d.arrow(745, 130, 775, 130, color=P)
    d.text(750, 215, "Gaussian lớn phủ vùng chi tiết cao:", 14)
    d.text(750, 237, "thay bằng 2 Gaussian, tỉ lệ chia φ = 1,6,", 14)
    d.text(750, 259, "vị trí lấy mẫu từ chính Gaussian gốc (giữ thể tích)", 14, "#555")
    d.text(500, 305, "Điều kiện chung: trung bình ‖∇_μ′ L‖ trong không gian ảnh > τ_pos = 0,0002; kiểm tra mỗi 100 vòng", 15, "#a33a1f", weight="bold")
    d.save("3dgs/densify.svg")


def gs_tiles():
    d = D(1000, 330, "Rasterizer theo ô của 3DGS")
    x0, y0, S = 30, 60, 50
    for i in range(6):
        for j in range(4):
            d.raw(f"<rect x='{x0+i*S}' y='{y0+j*S}' width='{S}' height='{S}' fill='white' stroke='#bbb'/>")
    ell = [(100, 110, 55, 22, -20, A), (190, 150, 40, 30, 30, P), (240, 200, 70, 20, 10, G)]
    for (cx, cy, rx, ry, rot, c) in ell:
        d.raw(f"<ellipse cx='{cx}' cy='{cy}' rx='{rx}' ry='{ry}' transform='rotate({rot} {cx} {cy})' fill='{c}' fill-opacity='.3' stroke='{c}'/>")
    d.text(180, 45, "Màn hình chia ô 16×16 pixel", 15, "#222", weight="bold")
    d.text(180, 285, "Mỗi Gaussian được nhân bản cho mọi ô", 13, "#444")
    d.text(180, 303, "mà elip 99% tin cậy của nó phủ lên", 13, "#444")
    d.arrow(340, 160, 380, 160)
    d.box(380, 80, 250, 160, ["Khóa 64 bit", "[ ID ô | độ sâu z ]", "một lần radix sort", "trên GPU cho cả ảnh", "→ danh sách mỗi ô"], "op", 14)
    d.arrow(630, 160, 670, 160)
    d.box(670, 70, 310, 180, ["Mỗi ô = 1 thread block", "nạp Gaussian vào bộ nhớ chia sẻ", "mỗi pixel: C += T·αᵢ·cᵢ, T ← T(1−αᵢ)", "dừng khi độ phủ α tiệm cận 1", "(không vượt quá 0,9999)"], "net", 14)
    d.text(680, 285, "Lan truyền ngược: duyệt lại danh sách từ sau ra trước,", 13, "#444")
    d.text(680, 305, "chỉ lưu độ phủ cuối mỗi pixel để khôi phục Tᵢ", 13, "#444")
    d.save("3dgs/tiles.svg")


def radfoam_pipeline():
    d = D(1000, 330, "Vòng huấn luyện Radiant Foam")
    d.box(10, 110, 140, 90, ["COLMAP", "điểm thưa → site", "init 131 072"], "data")
    d.arrow(150, 155, 190, 155)
    d.box(190, 100, 170, 110, ["Voronoi foam", "pᵢ, σᵢ, SH bậc 3", "kề Delaunay (CSR)"], "data")
    d.arrow(360, 155, 400, 155)
    d.box(400, 100, 170, 110, ["Dò tia", "duyệt mặt ô", "tích phân đúng"], "op")
    d.arrow(570, 155, 610, 155)
    d.box(610, 100, 170, 110, ["L = L_rgb", "+ λ L_quantile", "λ = 10⁻⁴ (cấu hình)"], "loss")
    d.arrow(780, 155, 820, 155)
    d.box(820, 100, 170, 110, ["Adam", "cập nhật p, σ, SH", "(p cố định sau 18k)"], "net")
    d.path("M 905 210 L 905 275 L 640 275", G)
    d.box(430, 245, 210, 60, ["Tam giác hóa lại", "mỗi k bước, k: 1 → 100"], "op", 14)
    d.path("M 430 275 L 275 275 L 275 215", G)
    d.box(60, 245, 170, 60, ["Densify / prune", "2k → 11k, ×1,15"], "net", 14)
    d.path("M 145 245 L 145 210 L 190 180", G, "5,4")
    d.text(500, 40, "Cùng khung tối ưu như 3DGS, nhưng kết xuất bằng ray tracing trên ô không chồng lấn", 16, P, weight="bold")
    d.save("radfoam/pipeline.svg")


def overview():
    d = D(1000, 250, "Năm biểu diễn cảnh 3D")
    items = [("NeRF (2020)", ["MLP liên tục", "(c, σ) = F_Θ(x, d)", "ray marching"], "net"),
             ("NeuS (2021)", ["SDF nơ-ron", "f(x) = 0 là bề mặt", "trọng số không lệch"], "net"),
             ("Plenoxels (2022)", ["lưới voxel thưa", "(σ, SH) tại đỉnh", "nội suy ba tuyến tính"], "loss"),
             ("3DGS (2023)", ["Gaussian tường minh", "μ, Σ, o, SH", "rasterization theo ô"], "op"),
             ("RadFoam (2025)", ["bọt Voronoi", "(σ, SH) hằng mỗi ô", "dò tia duyệt mặt"], "data")]
    for k, (t, lines, kind) in enumerate(items):
        x = 8 + k * 198
        d.box(x, 50, 180, 150, [t] + lines, kind, 16)
        if k < 4:
            d.arrow(x + 180, 125, x + 198, 125)
    d.text(500, 30, "Cùng mô hình kết xuất thể tích C = Σ Tᵢ αᵢ cᵢ, khác nhau ở biểu diễn và cách duyệt tia", 15, P, weight="bold")
    d.text(500, 235, "Xu hướng: ẩn và chậm → tường minh và nhanh; NeuS bổ sung bề mặt SDF", 14, "#444")
    d.save("overview.svg")


if __name__ == "__main__":
    nerf_pipeline(); nerf_mlp(); neus_arch(); gs_pipeline(); gs_gaussian(); gs_densify(); gs_tiles(); radfoam_pipeline(); overview()
    print("ok")
