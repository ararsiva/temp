import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import numpy as np

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Thermo RC1", layout="wide")

# --- БАЗА ДАННЫХ ---
materials_db = {"HDPE": 0.00015, "Поликарбонат": 0.00007, "Акрил": 0.000075, "Стеклопластик": 0.000025, "Свой вариант": 0.00015}

# --- САЙДБАР (ПАНЕЛЬ УПРАВЛЕНИЯ) ---
st.sidebar.header("Параметры Thermo RC1")
material = st.sidebar.selectbox("Материал:", list(materials_db.keys()))

if material == "Свой вариант":
    alpha = st.sidebar.number_input("КЛТР (1/°C):", value=0.00015, format="%.7f")
else:
    alpha = materials_db[material]

W0 = st.sidebar.number_input("Ширина (мм):", value=2000)
H0 = st.sidebar.number_input("Высота (мм):", value=1000)
st.sidebar.markdown("---")
st.sidebar.caption("Thermo RC1 Build 2025")

# --- ЛОГИКА ---
def f_num(val, show_sign=False):
    rounded = round(float(val), 2)
    res = "{:g}".format(rounded)
    if show_sign and rounded > 0: return "+" + res
    return res

def calculate_coords(total_len, edge_offset, min_dist=165, max_dist=300):
    center = total_len / 2
    span = center - edge_offset
    n_spaces = max(1, int(span / min_dist))
    step = span / n_spaces
    if step > max_dist: n_spaces += 1; step = span / n_spaces
    side = [edge_offset + i * step for i in range(n_spaces + 1)]
    return sorted(list(set(side + [total_len - s for s in side])))

def draw_correct_slot(ax, x, y, w, h, angle_deg, shift_x, shift_y):
    slot = patches.FancyBboxPatch((-w/2, -h/2), w, h, boxstyle=f"round,pad=0,rounding_size={min(w,h)/2}",
                                  edgecolor='black', facecolor='none', linewidth=1.5, zorder=2)
    t = mtransforms.Affine2D().rotate_deg(angle_deg).translate(x + shift_x, y + shift_y) + ax.transData
    slot.set_transform(t)
    ax.add_patch(slot)

# --- ВИЗУАЛИЗАЦИЯ ---
hole_d = 6.0
temps, colors = [-40.0, 20.0, 40.0], {-40.0: '#1f77b4', 20.0: 'black', 40.0: '#d62728'}
x_base, y_base = calculate_coords(W0, 100), calculate_coords(H0, 100)
key_pts = [(min(x_base), max(y_base), "Отверстие №1"), (W0/2, max(y_base), "Отверстие №2"),
           (W0/2, H0/2, "Отверстие №3"), (min(x_base), H0/2, "Отверстие №4")]

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 4, height_ratios=[1, 0.18, 0.45], hspace=0.35, wspace=0.35)
ax_main = fig.add_subplot(gs[0, :]); ax_table = fig.add_subplot(gs[1, :]); ax_table.axis('off')

# Заголовок с форматированием
ax_main.text(0.5, 1.15, "Проект Thermo: ", transform=ax_main.transAxes, ha='right', fontsize=14)
ax_main.text(0.5, 1.15, f" {material} ", transform=ax_main.transAxes, ha='left', fontsize=14, color='red', weight='bold')
ax_main.text(0.72, 1.15, f"(КЛТР: {alpha:.7f})", transform=ax_main.transAxes, ha='left', fontsize=12, color='black')

rows = []
for T in temps:
    dt = T - 20.0
    f = 1 + alpha * dt
    Wt, Ht = W0*f, H0*f
    ax_main.add_patch(patches.Rectangle(((W0-Wt)/2, (H0-Ht)/2), Wt, Ht, edgecolor=colors[T], facecolor='none', alpha=0.3, lw=1.2))
    xs = [(W0/2) + (bx-W0/2)*f for bx in x_base for by in y_base]
    ys = [(H0/2) + (by-H0/2)*f for bx in x_base for by in y_base]
    ax_main.scatter(xs, ys, s=6, color=colors[T], alpha=0.5)
    if T == 20:
        for i, (kp_x, kp_y, _) in enumerate(key_pts): ax_main.text(kp_x, kp_y+30, str(i+1), weight='bold', ha='center', fontsize=10)
    rows.append([f"{T:+.0f}°C", f_num(Wt), f_num(Wt-W0, True), f_num(Ht), f_num(Ht-H0, True)])

ax_main.set_aspect('equal')
tab = ax_table.table(cellText=rows, colLabels=("T, °C", "Ширина", "ΔW", "Высота", "ΔH"), loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
tab.auto_set_font_size(False); tab.set_fontsize(11)
for cell in tab.get_celld().values(): cell.set_height(0.4) 

for i, (kp_x, kp_y, name) in enumerate(key_pts):
    ax_s = fig.add_subplot(gs[2, i]); dx_v, dy_v = kp_x - W0/2, kp_y - H0/2
    s_min_x, s_min_y = dx_v * alpha * (-60), dy_v * alpha * (-60)
    s_max_x, s_max_y = dx_v * alpha * (+20), dy_v * alpha * (+20)
    mid_x, mid_y = (s_min_x + s_max_x)/2, (s_min_y + s_max_y)/2
    total_travel = np.sqrt((s_max_x - s_min_x)**2 + (s_max_y - s_min_y)**2)
    ang = np.degrees(np.arctan2(dy_v, dx_v))
    
    if i == 2: ax_s.add_patch(plt.Circle((0, 0), hole_d/2, color='black', fill=False, lw=1.5))
    elif i == 1: draw_correct_slot(ax_s, 0, 0, hole_d, hole_d + total_travel, 0, 0, mid_y)
    elif i == 3: draw_correct_slot(ax_s, 0, 0, hole_d + total_travel, hole_d, 0, mid_x, 0)
    else: draw_correct_slot(ax_s, 0, 0, hole_d + total_travel, hole_d, ang, mid_x, mid_y)

    for T in temps:
        dt_val = T - 20.0
        f_t = 1 + alpha * dt_val
        sx, sy = ((W0/2) + (kp_x-W0/2)*f_t) - kp_x, ((H0/2) + (kp_y-H0/2)*f_t) - kp_y
        ld = np.sqrt(sx**2 + sy**2) * (1 if dt_val >= 0 else -1)
        ax_s.add_patch(plt.Circle((sx, sy), hole_d/2, facecolor=colors[T], edgecolor='black', lw=0.7, alpha=0.9, zorder=10))

        if T != 20:
            if i == 2: continue
            if i == 0: tx, ty = 0, (8.5 if sy > 0 else -8.5)
            elif i == 1: tx, ty = (8.5 if dt_val > 0 else -8.5), 0
            elif i == 3: tx, ty = 0, (8.5 if dt_val > 0 else -8.5)
            else: tx, ty = 8, 8
            ax_s.text(sx + tx, sy + ty, f"{f_num(ld, True)}мм", color=colors[T], fontsize=8, weight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1))
        elif i == 2:
            ax_s.text(0, 7.5, "0 мм", color='black', fontsize=8, weight='bold', ha='center')

    ax_s.set_xlim(-15, 15); ax_s.set_ylim(-15, 15); ax_s.set_aspect('equal')
    ax_s.grid(True, linestyle=':', alpha=0.4); ax_s.set_title(name, fontsize=9.5)

fig.text(0.5, 0.05, "Примечание: Смещение замеряется между центрами отверстий (раскрой +20°C). Паз рассчитан по внешним границам кругов.", ha='center', fontsize=9, style='italic', color='#555')

st.pyplot(fig)
