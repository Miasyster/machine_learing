#!/usr/bin/env python3
"""
测试图表中文显示和负值显示
"""

import matplotlib.pyplot as plt
import numpy as np

def test_chart_display():
    """测试图表显示效果"""
    
    # 设置中文字体和负号显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试数据
    models = ['线性回归', '岭回归', '套索回归', '随机森林', '梯度提升', 'SVR']
    scores = [0.95, 0.85, -2.5, -1.2, -0.8, -3.1]  # 包含负值
    errors = [0.05, 0.15, 1.2, 0.8, 0.5, 1.5]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：柱状图
    bars = axes[0].bar(models, scores, yerr=errors, capsize=5, alpha=0.7)
    axes[0].set_title('模型性能比较（包含负值）')
    axes[0].set_ylabel('R² 分数')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 设置Y轴范围以正确显示负值
    min_score = min(scores) - max(errors)
    max_score = max(scores) + max(errors)
    y_margin = (max_score - min_score) * 0.1
    axes[0].set_ylim(min_score - y_margin, max_score + y_margin)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='零线')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 为负值柱子设置不同颜色
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if score < 0:
            bar.set_color('red')
        else:
            bar.set_color('blue')
    
    # 右图：折线图
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) - 0.5  # 包含负值
    y2 = np.cos(x) * 0.8
    
    axes[1].plot(x, y1, label='正弦波（偏移）', linewidth=2)
    axes[1].plot(x, y2, label='余弦波', linewidth=2)
    axes[1].set_title('时间序列数据（包含负值）')
    axes[1].set_xlabel('时间')
    axes[1].set_ylabel('数值')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('chart_display_test.png', dpi=300, bbox_inches='tight')
    print("✅ 测试图表已保存为 'chart_display_test.png'")
    print("📊 图表特点:")
    print("   - 中文标题和标签")
    print("   - 负值正确显示")
    print("   - 零线参考线")
    print("   - 负值柱子用红色标识")
    
    # 显示图表（如果在交互环境中）
    try:
        plt.show()
    except:
        print("💡 在非交互环境中运行，图表已保存到文件")
    
    plt.close()

if __name__ == "__main__":
    test_chart_display()