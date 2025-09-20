"""
测试运行脚本
用于运行项目中的所有测试
"""

import sys
import os
import subprocess
from pathlib import Path

def run_data_pipeline_tests():
    """运行数据管道测试"""
    print("🧪 运行数据管道测试...")
    
    try:
        # 使用subprocess运行测试文件
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "src/tests/test_data_pipeline.py", 
            "-v"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ 数据管道测试通过")
            return True
        else:
            print(f"❌ 数据管道测试失败")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 数据管道测试异常: {e}")
        return False

def run_pytest_tests():
    """运行pytest风格的测试"""
    print("🧪 运行pytest测试...")
    
    try:
        # 使用pytest运行所有测试
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "src/tests/", 
            "-v"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ pytest测试通过")
            return True
        else:
            print(f"❌ pytest测试失败")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ pytest测试异常: {e}")
        return False

def run_all_tests(verbose=True):
    """运行所有测试"""
    print("🚀 开始运行所有测试...")
    print("=" * 60)
    
    tests = [
        ("pytest测试套件", run_pytest_tests),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 运行 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("🏁 测试总结")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试套件通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
    
    return passed == len(results)

def run_quick_tests():
    """快速运行测试（不显示详细输出）"""
    print("⚡ 快速测试模式...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "src/tests/", 
            "-q"  # 安静模式
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ 所有测试通过")
            return True
        else:
            print("❌ 测试失败")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行项目测试")
    parser.add_argument("--quick", "-q", action="store_true", 
                       help="快速测试模式（不显示详细输出）")
    parser.add_argument("--data-pipeline", "-d", action="store_true",
                       help="只运行数据管道测试")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_tests()
    elif args.data_pipeline:
        run_data_pipeline_tests()
    else:
        run_all_tests()