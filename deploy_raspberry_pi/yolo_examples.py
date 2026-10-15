#!/usr/bin/env python3
"""
YOLO11 识别效果示例图：(a) 废墟识别；(b) 人员识别

对给定图片分别运行 best_collapsed.pt 与 best_person.pt，画出检测框并拼成一张示例图，
同时打印每张图的检测数量、置信度与推理耗时。建议选用训练时划出的验证集图片（未参与训练）。
参数也可以是目录：对目录中的全部图片推理，按检出置信度之和选出 --pick 张作为示例。

示例:
    python3 yolo_examples.py --collapsed valid/ruins_01.jpg --person valid/person_03.jpg \
        --out yolo_examples.png
    python3 yolo_examples.py --collapsed a.jpg b.jpg --person c.jpg d.jpg --conf 0.25 --width 15
    python3 yolo_examples.py --collapsed Data/collapsed.v1i.yolov11/valid/images \
        --person Data/person.v1i.yolov11/valid/images --pick 2
"""

import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

GROUPS = [
    ("a", "collapsed", HERE / "best_collapsed.pt", "#E8590C"),
    ("b", "person", HERE / "best_person.pt", "#12B886"),
]


IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def expand(paths):
    """展开目录参数，返回 (图片列表, 是否来自目录)"""
    out, from_dir = [], False
    for p in map(Path, paths):
        if p.is_dir():
            from_dir = True
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        else:
            out.append(p)
    return out, from_dir


def detect(model_path, images, conf, imgsz):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    out = []
    for img in images:
        t0 = time.perf_counter()
        r = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes = [(*map(float, b.xyxy[0].tolist()), float(b.conf[0]), r.names[int(b.cls[0])]) for b in r.boxes]
        out.append((img, r.orig_img[:, :, ::-1], boxes, r.speed.get("inference", ms)))
    return out


def plot(results, out_path, width_cm, font):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    panels = [(tag, color, item) for tag, color, items in results for item in items]
    ratios = [im.shape[1] / im.shape[0] for _, _, (_, im, _, _) in panels]
    w = width_cm / 2.54
    h = w / sum(ratios) + 0.28
    fig, axes = plt.subplots(1, len(panels), figsize=(w, h), gridspec_kw={"width_ratios": ratios})
    axes = [axes] if len(panels) == 1 else list(axes)
    counts = {}
    for tag, *_ in panels:
        counts[tag] = counts.get(tag, 0) + 1
    seen = {}
    for ax, (tag, color, (_, im, boxes, _)) in zip(axes, panels):
        seen[tag] = seen.get(tag, 0) + 1
        ax.imshow(im)
        lw = max(0.8, im.shape[1] / 400)
        if not boxes:
            ax.text(0.02, 0.97, "no detection", transform=ax.transAxes, fontsize=font, color="white",
                    va="top", ha="left", bbox=dict(boxstyle="square,pad=0.15", fc="0.3", ec="none"))
        for x1, y1, x2, y2, c, name in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw))
            top = y1 < im.shape[0] * 0.06                  # 框贴近上边缘时把标签放到框内
            ax.text(x1, y1, f"{name} {c:.2f}", fontsize=font, color="white", va="top" if top else "bottom",
                    ha="left", bbox=dict(boxstyle="square,pad=0.12", fc=color, ec="none"), clip_on=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        label = f"({tag})" if counts[tag] == 1 else f"({tag}{seen[tag]})"
        ax.set_xlabel(label, fontsize=font + 1.5, labelpad=2)
    fig.tight_layout(pad=0.2, w_pad=0.4)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collapsed", nargs="+", required=True, help="废墟场景图片")
    ap.add_argument("--person", nargs="+", required=True, help="人员场景图片")
    ap.add_argument("--collapsed-model", default=str(GROUPS[0][2]))
    ap.add_argument("--person-model", default=str(GROUPS[1][2]))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="yolo_examples.png")
    ap.add_argument("--width", type=float, default=15.0, help="成图宽度（厘米），按报告版心设置")
    ap.add_argument("--font", type=float, default=6.0, help="框标签字号（磅）")
    ap.add_argument("--pick", type=int, default=1, help="参数为目录时，每组选出的示例图片数")
    args = ap.parse_args()

    jobs = [("a", args.collapsed_model, args.collapsed, GROUPS[0][3]),
            ("b", args.person_model, args.person, GROUPS[1][3])]
    results = []
    for tag, model, paths, color in jobs:
        images, from_dir = expand(paths)
        items = detect(model, images, args.conf, args.imgsz)
        hit = sum(1 for *_, boxes, _ in items if boxes)
        print(f"({tag}) {len(items)} 张图片，{hit} 张有检出")
        if from_dir:
            items = sorted(items, key=lambda it: -sum(b[4] for b in it[2]))[:args.pick]
        for img, _, boxes, ms in items:
            confs = ", ".join(f"{c:.2f}" for *_, c, _ in boxes) or "-"
            print(f"({tag}) {Path(img).name}: {len(boxes)} 个目标, 置信度 [{confs}], 推理 {ms:.1f} ms")
        results.append((tag, color, items))
    plot(results, args.out, args.width, args.font)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
