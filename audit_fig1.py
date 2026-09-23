#!/usr/bin/env python3
"""Mechanical overlap audit for Fig1 in make_figures.py.

Parses the fig1() source (no rendering) and checks that no panel title,
sub-note, box, inset, or side panel overlap in data coordinates
(xlim/ylim = 0..100 over a 6.9 x 7.8 inch figure).

Text half-heights are estimated conservatively from the font size:
  half_y(units) = fs * 1.4 / 72 * (100/7.8) / 2   (line height ~1.4em)
  half_x(units) = len * 0.5 * fs / 72 * (100/6.9) / 2  (avg glyph ~0.5em)
"""
import re

SRC = "."

with open(SRC) as f:
    txt = f.read()

# extract fig1 block
m = re.search(r"def fig1\(\):.*?(?=\ndef fig2\(\):|def fig[0-9])", txt, re.S)
block = m.group(0)

FS_PANEL = 10.5
FS_NOTE = 8.5
BOX_FS = 8.4


def half_y(fs):
    return fs * 1.4 / 72.0 * (100.0 / 7.8) / 2.0


def half_x(fs, nchars):
    return nchars * 0.5 * fs / 72.0 * (100.0 / 6.9) / 2.0


# --- panel titles ---------------------------------------------------------
titles = {}
for mm in re.finditer(r'ax\.text\(([\d.]+),\s*([\d.]+),\s*"([A-D])[^"]*"', block):
    x, y, letter = float(mm.group(1)), float(mm.group(2)), mm.group(3)
    # capture the full quoted string for length
    s = re.search(r'ax\.text\(' + re.escape(mm.group(1)) + r',\s*' + re.escape(mm.group(2)) +
                  r',\s*"([^"]*)"', block)
    nchars = len(s.group(1))
    titles[letter] = (x, y, nchars)

# --- boxes ---------------------------------------------------------------
boxes = []  # (x, y, w, h)
for mm in re.finditer(r'box\(\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),', block):
    boxes.append((float(mm.group(1)), float(mm.group(2)),
                  float(mm.group(3)), float(mm.group(4))))

# --- inset ---------------------------------------------------------------
ins = re.search(r'inset = fig\.add_axes\(\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]\)', block)
inset = (float(ins.group(1)), float(ins.group(2)), float(ins.group(3)), float(ins.group(4)))
# convert figure-fraction -> data units (axes spans full 0..100)
inset_rect = (inset[0]*100, inset[1]*100, inset[2]*100, inset[3]*100)

# --- side note panel -----------------------------------------------------
pn = re.search(r'FancyBboxPatch\(\(([\d.]+),\s*([\d.]+)\),\s*([\d.]+),\s*([\d.]+),', block)
panel = (float(pn.group(1)), float(pn.group(2)), float(pn.group(3)), float(pn.group(4)))

# --- sub-notes (ax.text with FS_NOTE, not a panel title) -----------------
subnotes = []
for mm in re.finditer(r'ax\.text\(([\d.]+),\s*([\d.]+),\s*"([^"A-D][^"]*)"[^)]*fontsize=FS_NOTE', block):
    x, y = float(mm.group(1)), float(mm.group(2))
    nchars = len(mm.group(3))
    subnotes.append((x, y, nchars))


def overlap(a, b):
    """a,b = (x,y,w,h); return overlap area > 0."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ox = max(0.0, min(ax0+aw, bx0+bw) - max(ax0, bx0))
    oy = max(0.0, min(ay0+ah, by0+bh) - max(ay0, by0))
    return ox * oy


problems = []


def check_title(title_letter):
    x, y, n = titles[title_letter]
    # panel titles are left-aligned (default ha="left")
    tr = (x, y - half_y(FS_PANEL), 2*half_x(FS_PANEL, n), 2*half_y(FS_PANEL))
    for (bx, by, bw, bh) in boxes:
        # pad box by 0.6 for the fancy bbox rounding
        br = (bx - 0.6, by - 0.6, bw + 1.2, bh + 1.2)
        ov = overlap(tr, br)
        if ov > 0:
            problems.append(f"[FAIL] Panel {title_letter} title overlaps a box "
                            f"(title y={y}, box top={by+bh:.1f})")


for L in ("A", "B", "C", "D"):
    check_title(L)

# sub-notes vs boxes
for (x, y, n) in subnotes:
    tr = (x - half_x(FS_NOTE, n), y - half_y(FS_NOTE),
          2*half_x(FS_NOTE, n), 2*half_y(FS_NOTE))
    for (bx, by, bw, bh) in boxes:
        br = (bx - 0.6, by - 0.6, bw + 1.2, bh + 1.2)
        if overlap(tr, br) > 0:
            problems.append(f"[FAIL] sub-note at y={y} overlaps a box")

# inset vs C boxes and vs D title
inset_padded = (inset_rect[0]-0.5, inset_rect[1]-0.5, inset_rect[2]+1.0, inset_rect[3]+1.0)
for (bx, by, bw, bh) in boxes:
    br = (bx - 0.6, by - 0.6, bw + 1.2, bh + 1.2)
    if overlap(inset_padded, br) > 0:
        problems.append(f"[FAIL] inset overlaps a box at y={by:.1f}")
# D title vs inset
dx, dy, dn = titles["D"]
dtr = (dx - half_x(FS_PANEL, dn), dy - half_y(FS_PANEL),
       2*half_x(FS_PANEL, dn), 2*half_y(FS_PANEL))
if overlap(dtr, inset_padded) > 0:
    problems.append("[FAIL] D title overlaps inset")

# panel vs inset / bounds
panel_padded = (panel[0]-0.5, panel[1]-0.5, panel[2]+1.0, panel[3]+1.0)
if overlap(panel_padded, inset_padded) > 0:
    problems.append("[FAIL] side note panel overlaps inset")
if panel[0] < 0 or panel[1] < 0 or panel[0]+panel[2] > 100 or panel[1]+panel[3] > 100:
    problems.append("[FAIL] side note panel out of bounds")

# B loop arrow (the curved dashed return arrow) vs its sub-note ---------------
loop = re.search(r'arrow\(\(58\.5,\s*([\d.]+)\),\s*\(34\.0,\s*[\d.]+\),\s*'
                 r'color="#8fa8bf",\s*rad=([-0-9.]+)', block)
bsub = re.search(r'ax\.text\(46\.0,\s*([\d.]+),\s*"repeat the backward sweep', block)
if loop and bsub:
    ly = float(loop.group(1))
    rad = float(loop.group(2))
    chord = 58.5 - 34.0
    bulge = abs(rad) * chord / 2.0          # quadratic-bezier max deviation
    loop_lowest = ly - bulge
    by = float(bsub.group(1))
    sub_top = by + 0.76                      # single-line note half-height
    gap = loop_lowest - sub_top
    print(f"[B] loop arrow y={ly} rad={rad} -> dips to y={loop_lowest:.2f}; "
          f"sub-note top y={sub_top:.2f}; gap={gap:.2f}")
    if gap < 0.5:
        problems.append(f"[FAIL] B loop arrow dips to {loop_lowest:.1f} and "
                        f"overlaps the 'repeat...' sub-note (gap {gap:.1f})")
    else:
        print("     -> loop clears the sub-note (no press)")
else:
    print("[B] loop-arrow / sub-note pattern not found (skipped dip check)")

# report
print("=== Fig1 geometry audit ===")
print(f"inset (data units): x={inset_rect[0]:.1f} y={inset_rect[1]:.1f} "
      f"w={inset_rect[2]:.1f} h={inset_rect[3]:.1f} "
      f"(was 6.5x9.1 area; now {inset_rect[2]*inset_rect[3]:.0f} units^2)")
print(f"side note panel: x={panel[0]:.1f} y={panel[1]:.1f} "
      f"w={panel[2]:.1f} h={panel[3]:.1f}")
print(f"boxes: {len(boxes)}  subnotes: {len(subnotes)}")
print()
if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print("  " + p)
else:
    print("PASS: no title/box/inset/panel overlaps detected.")
