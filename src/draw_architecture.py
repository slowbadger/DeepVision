"""
Draw publication-quality architecture diagram matching the DeepVision Dual-Branch model.

Layout:
Input Image (I) (3x256x256)
  ├─> RGB Stream -> ResNet-34 -> [F_rgb] ────────────────┐
  │                                                      ▼
  └─> ELA Stream -> ResNet-34 -> [F_ela] ───> Cross-Stream Gated Attention Fusion ───> Multi-Task Heads:
                                                - Head 1: Binary Authenticity
                                                - Head 2: Type Attribution
                                                - Head 3: Heatmap Mask Decoder
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram(output_path="results/paper_figures/fig1_architecture.png"):
    fig = plt.figure(figsize=(16, 7.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Color Palette - Professional IEEE / Springer style
    c_input    = "#E3F2FD"  # soft light blue
    c_input_b  = "#1565C0"
    c_rgb      = "#E8F5E9"  # soft light green
    c_rgb_b    = "#2E7D32"
    c_ela      = "#FFF3E0"  # soft orange
    c_ela_b    = "#E65100"
    c_resnet   = "#EDE7F6"  # soft purple
    c_resnet_b = "#512DA8"
    c_fusion   = "#E0F2F1"  # soft teal
    c_fusion_b = "#00695C"
    c_heads    = "#FCE4EC"  # soft pink/rose
    c_heads_b  = "#AD1457"
    c_line     = "#37474F"

    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle=None, badge=None, fontsize=10.5):
        # Drop shadow
        shadow = FancyBboxPatch(
            (x + 0.05, y - 0.05), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#CFD8DC", edgecolor="none", alpha=0.45, zorder=1
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=2.0, zorder=2
        )
        ax.add_patch(box)

        # Title
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title,
                    ha="center", va="center", fontsize=fontsize, fontweight="bold",
                    color="#212121", zorder=3)
            ax.text(x + w / 2, y + h * 0.32, subtitle,
                    ha="center", va="center", fontsize=fontsize - 2,
                    color="#455A64", style="italic", zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center", fontsize=fontsize, fontweight="bold",
                    color="#212121", zorder=3)

        # Optional corner badge (tensor shape)
        if badge:
            badge_box = FancyBboxPatch(
                (x + w - 1.15, y - 0.22), 1.15, 0.32,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#FFFFFF", edgecolor=border_color, linewidth=1.2, zorder=4
            )
            ax.add_patch(badge_box)
            ax.text(x + w - 0.575, y - 0.06, badge,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color=border_color, zorder=5)

    def draw_arrow(p1, p2, label=None, label_pos=0.5, label_offset=(0, 0.18), rad=0.0):
        connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0.0"
        arrow = patches.FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>,head_length=6.0,head_width=4.0",
            connectionstyle=connectionstyle,
            color=c_line, linewidth=2.0, zorder=3
        )
        ax.add_patch(arrow)
        if label:
            lx = p1[0] + (p2[0] - p1[0]) * label_pos + label_offset[0]
            ly = p1[1] + (p2[1] - p1[1]) * label_pos + label_offset[1]
            ax.text(lx, ly, label, ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1A237E",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#B0BEC5", alpha=0.9),
                    zorder=5)

    # ─────────────────────────────────────────────────────────────
    # 1. INPUT IMAGE (Left)
    # ─────────────────────────────────────────────────────────────
    x_input, y_input, w_input, h_input = 0.5, 2.7, 2.4, 2.0
    draw_box(x_input, y_input, w_input, h_input, c_input, c_input_b,
             "Input Image (I)", "RGB Tensor\n(3 × 256 × 256)", badge="3×256²", fontsize=11)

    # ─────────────────────────────────────────────────────────────
    # 2. STREAM BRANCHES (Upper = RGB, Lower = ELA)
    # ─────────────────────────────────────────────────────────────
    w_stream, h_stream = 2.2, 1.4
    x_stream = 3.6
    y_rgb = 4.8
    y_ela = 1.2

    draw_box(x_stream, y_rgb, w_stream, h_stream, c_rgb, c_rgb_b,
             "RGB Stream", "Normalized [0, 1]\nImageNet Stats", badge="3×256²", fontsize=10.5)

    draw_box(x_stream, y_ela, w_stream, h_stream, c_ela, c_ela_b,
             "ELA Stream", "Error Level Residuals\nQ=90, S=15", badge="3×256²", fontsize=10.5)

    # Arrows from Input -> Streams (Orthogonal branch)
    # Split point
    x_split = 3.1
    ax.plot([x_input + w_input, x_split], [y_input + h_input/2, y_input + h_input/2], color=c_line, linewidth=2.0, zorder=3)
    ax.plot([x_split, x_split], [y_rgb + h_stream/2, y_ela + h_stream/2], color=c_line, linewidth=2.0, zorder=3)
    draw_arrow((x_split, y_rgb + h_stream/2), (x_stream, y_rgb + h_stream/2))
    draw_arrow((x_split, y_ela + h_stream/2), (x_stream, y_ela + h_stream/2))

    # ─────────────────────────────────────────────────────────────
    # 3. FEATURE EXTRACTORS (ResNet-34)
    # ─────────────────────────────────────────────────────────────
    w_res, h_res = 2.2, 1.4
    x_res = 6.4

    draw_box(x_res, y_rgb, w_res, h_res, c_resnet, c_resnet_b,
             "ResNet-34", "Feature Extractor\n(Branch 1)", badge="512×8²", fontsize=10.5)

    draw_box(x_res, y_ela, w_res, h_res, c_resnet, c_resnet_b,
             "ResNet-34", "Feature Extractor\n(Branch 2)", badge="512×8²", fontsize=10.5)

    draw_arrow((x_stream + w_stream, y_rgb + h_stream/2), (x_res, y_rgb + h_stream/2))
    draw_arrow((x_stream + w_stream, y_ela + h_stream/2), (x_res, y_ela + h_stream/2))

    # ─────────────────────────────────────────────────────────────
    # 4. CROSS-STREAM GATED ATTENTION FUSION (Center)
    # ─────────────────────────────────────────────────────────────
    w_fus, h_fus = 2.7, 3.2
    x_fus = 9.4
    y_fus = 2.1

    draw_box(x_fus, y_fus, w_fus, h_fus, c_fusion, c_fusion_b,
             "Cross-Stream Gated\nAttention Fusion",
             "Channel Concat [1024×8²]\n↓\nConv2d + BN + ReLU\n↓\nSpatial Saliency Gate σ(W*F)\n↓\nAdaptive Pooling",
             badge="512-dim", fontsize=11)

    # Arrow from RGB ResNet -> Fusion (downwards diagonal)
    draw_arrow((x_res + w_res, y_rgb + h_stream/2), (x_fus, y_fus + h_fus * 0.75),
               label="[F_rgb]", label_pos=0.45, label_offset=(0, 0.22))

    # Arrow from ELA ResNet -> Fusion (upwards diagonal)
    draw_arrow((x_res + w_res, y_ela + h_stream/2), (x_fus, y_fus + h_fus * 0.25),
               label="[F_ela]", label_pos=0.45, label_offset=(0, -0.22))

    # ─────────────────────────────────────────────────────────────
    # 5. MULTI-TASK OUTPUT HEADS (Right)
    # ─────────────────────────────────────────────────────────────
    w_head, h_head = 2.9, 1.25
    x_head = 12.8
    y_head1 = 5.0
    y_head2 = 3.1
    y_head3 = 1.2

    draw_box(x_head, y_head1, w_head, h_head, c_heads, c_heads_b,
             "Head 1: Binary Authenticity", "FC (512→1) + Sigmoid\n→ Pristine vs Tampered",
             badge="P(Tampered)", fontsize=9.5)

    draw_box(x_head, y_head2, w_head, h_head, c_heads, c_heads_b,
             "Head 2: Type Attribution", "FC (512→5) + Softmax\n→ 5-Class Manipulation",
             badge="5 Classes", fontsize=9.5)

    draw_box(x_head, y_head3, w_head, h_head, c_heads, c_heads_b,
             "Head 3: Heatmap Mask Decoder", "ConvTranspose2d Upsampling\n→ Spatial Localization",
             badge="1×256²", fontsize=9.5)

    # Multi-head split from Fusion
    x_hsplit = 12.4
    y_mid_fus = y_fus + h_fus / 2

    ax.plot([x_fus + w_fus, x_hsplit], [y_mid_fus, y_mid_fus], color=c_line, linewidth=2.0, zorder=3)
    ax.plot([x_hsplit, x_hsplit], [y_head1 + h_head/2, y_head3 + h_head/2], color=c_line, linewidth=2.0, zorder=3)

    draw_arrow((x_hsplit, y_head1 + h_head/2), (x_head, y_head1 + h_head/2))
    draw_arrow((x_hsplit, y_head2 + h_head/2), (x_head, y_head2 + h_head/2))
    draw_arrow((x_hsplit, y_head3 + h_head/2), (x_head, y_head3 + h_head/2))

    # ─────────────────────────────────────────────────────────────
    # Header & Legend Banner
    # ─────────────────────────────────────────────────────────────
    ax.text(8.0, 7.15, "DeepVision Dual-Branch Multi-Task Tampering Detection Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#263238")
    ax.text(8.0, 6.82, "Synergistic RGB context and Error Level Analysis (ELA) with Gated Attention & Multi-Task Heads",
            ha="center", va="center", fontsize=10, style="italic", color="#546E7A")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram("results/paper_figures/fig1_architecture.png")
    # Also save a secondary named copy for convenience
    create_architecture_diagram("results/paper_figures/architecture_diagram.png")
