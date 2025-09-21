#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表查看脚本
用于验证生成的图表是否正确显示中文和负值
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def view_chart():
    """查看生成的图表"""
    chart_file = 'time_series_training_results.png'
    
    if not os.path.exists(chart_file):
        print(f"❌ 图表文件 '{chart_file}' 不存在")
        return
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 读取并显示图片
    img = mpimg.imread(chart_file)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('时间序列训练结果图表', fontsize=16, pad=20)
    
    # 保存预览图
    plt.savefig('chart_preview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 图表已显示")
    print(f"📊 原图表: {chart_file}")
    print(f"🔍 预览图: chart_preview.png")
    
    # 显示图片信息
    print(f"\n📋 图片信息:")
    print(f"   - 尺寸: {img.shape}")
    print(f"   - 文件大小: {os.path.getsize(chart_file)} bytes")

if __name__ == "__main__":
    view_chart()