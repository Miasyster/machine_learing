#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体测试脚本 - 验证中文显示效果
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 检查可用字体
print("检查系统中可用的中文字体:")
font_list = [f.name for f in fm.fontManager.ttflist if 'SimHei' in f.name or 'Microsoft YaHei' in f.name or 'SimSun' in f.name]
print(f"找到的中文字体: {font_list}")

# 创建测试图表
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 生成测试数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 绘制图表
ax.plot(x, y, label='正弦波')
ax.set_title('中文字体测试 - 标题显示测试')
ax.set_xlabel('时间 (秒)')
ax.set_ylabel('幅度')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加文本注释
ax.text(5, 0.5, '这是中文注释文本\n负数测试: -123.45', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        fontsize=12, ha='center')

plt.tight_layout()
plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
print("字体测试图表已保存为 'font_test.png'")
plt.show()