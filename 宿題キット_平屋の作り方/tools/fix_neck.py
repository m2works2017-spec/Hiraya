# -*- coding: utf-8 -*-
"""Soul Cinema の三面図で、体のパネルに入った頭を外す（0cr）。
中と右のパネルの中身を shift px 上げて首から下にし、空いた下は床の色の流れを続けて埋める。
使い方: python3 fix_neck.py 入力.png 出力.png shift mid_x0 mid_x1 right_x0 right_x1 [patch=x0,y0,x1,y1]
"""
import sys
import numpy as np
from PIL import Image, ImageFilter
src,dst,shift,mx0,mx1,rx0,rx1=sys.argv[1],sys.argv[2],*map(int,sys.argv[3:8])
patch=sys.argv[8] if len(sys.argv)>8 else None
im=np.asarray(Image.open(src).convert("RGB")).astype(np.float32)
H=im.shape[0]
out=im.copy()
rng=np.random.default_rng(1)
if patch:  # ラベルなどを、周りの布の色からなめらかに埋める（上下左右の縁から補間＋布目の細かい揺れ）
    x0,y0,x1,y1=map(int,patch.split(",")[:4])
    top=out[y0-1,x0:x1]; bot=out[y1,x0:x1]; lef=out[y0:y1,x0-1]; rig=out[y0:y1,x1]
    h=y1-y0; w=x1-x0
    v=np.linspace(0,1,h)[:,None,None]; u=np.linspace(0,1,w)[None,:,None]
    vert=top[None]*(1-v)+bot[None]*v
    horz=lef[:,None]*(1-u)+rig[:,None]*u
    fill=(vert+horz)/2
    # 布目：すぐ下の無地の所の細かい揺れだけを足す
    src=out[y1+4:y1+4+h,x0:x1]
    blur=np.asarray(Image.fromarray(np.clip(src,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))).astype(np.float32)
    fill=fill+(src-blur)*0.9
    out[y0:y1,x0:x1]=fill
    reg=Image.fromarray(np.clip(out[y0-2:y1+2,x0-2:x1+2],0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    out[y0-2:y1+2,x0-2:x1+2]=np.asarray(reg).astype(np.float32)
    im=out.copy()
def hblur(row,r):
    k=np.ones(2*r+1)/(2*r+1)
    pad=np.pad(row,((r,r),(0,0)),mode="edge")
    return np.stack([np.convolve(pad[:,c],k,mode="valid") for c in range(3)],axis=1)
for x0,x1 in [(mx0,mx1),(rx0,rx1)]:
    p=im[:,x0:x1+1].copy()
    W=p.shape[1]
    q=np.empty_like(p)
    q[:H-shift]=p[shift:]
    base=hblur(p[H-6:H].mean(axis=0),90)          # 元の一番下の数行を横に強くならす
    # 縦の明るさの流れ（左右の端の列で、下から200行の傾き）
    edge=np.concatenate([p[H-200:H,:30],p[H-200:H,-30:]],axis=1).mean(axis=(1,2))
    slope=np.polyfit(np.arange(200),edge,1)[0]
    slope=float(np.clip(slope,-0.05,0.05))
    for i,y in enumerate(range(H-shift,H)):
        q[y]=base+slope*i+rng.normal(0,1.4,base.shape)
    # つなぎ目（12行）をなじませる
    j=H-shift
    for k in range(12):
        y=j-6+k
        if 0<=y<j:
            w=(k+1)/13
            q[y]=q[y]*(1-w)+base*w
    out[:,x0:x1+1]=q
img=Image.fromarray(np.clip(out,0,255).astype(np.uint8))
img.save(dst); print("saved",dst)
