# -*- coding: utf-8 -*-
"""生成 DCCA-GS 全流程配图（SVG）并检查文字溢出。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_a, gen_b, gen_c, gen_d

OUT = os.path.dirname(os.path.abspath(__file__))

mods = [gen_a, gen_b, gen_c, gen_d]
for m in mods:
    for name in dir(m):
        if name.startswith("fig") and callable(getattr(m, name)):
            getattr(m, name)(OUT)
            print("ok:", name)
print("done ->", OUT)
