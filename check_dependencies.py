#!/usr/bin/env python3
"""
Batch Image Difference to Prompt Files - 依赖检查工具
用于验证所需的Python库是否正确安装

作者信息:
作者: Asir
QQ交流群: 960598442
Discord: asir_50811
公众号: AsirAI
公众号: https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA

使用方法:
python check_dependencies.py
"""

import sys
import subprocess
import importlib.util

def check_library(library_name, import_name=None, version_attr=None):
    """
    检查指定库是否已安装
    
    Args:
        library_name: pip包名称
        import_name: 导入时使用的名称（如果与包名不同）
        version_attr: 版本属性名称
    
    Returns:
        tuple: (是否安装成功, 版本信息, 错误信息)
    """
    if import_name is None:
        import_name = library_name
        
    try:
        # 尝试导入库
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            return False, None, f"模块 '{import_name}' 未找到"
            
        module = importlib.import_module(import_name)
        
        # 尝试获取版本信息
        version = "未知版本"
        if version_attr:
            if hasattr(module, version_attr):
                version = getattr(module, version_attr)
        elif hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
            
        return True, version, None
        
    except ImportError as e:
        return False, None, f"导入错误: {str(e)}"
    except Exception as e:
        return False, None, f"未知错误: {str(e)}"

def install_library(library_name):
    """
    尝试自动安装缺失的库
    
    Args:
        library_name: 要安装的库名称
        
    Returns:
        bool: 安装是否成功
    """
    try:
        print(f"  🔄 正在安装 {library_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", library_name
        ], capture_output=True, text=True)
        print(f"  ✅ {library_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ {library_name} 安装失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {library_name} 安装过程出错: {e}")
        return False

def main():
    """主检查函数"""
    print("🔍 Batch Image Difference to Prompt Files (Kontext Style) V2.0 - 依赖检查")
    print("=" * 70)
    
    # 定义所需的依赖库
    required_libraries = [
        {
            'pip_name': 'torch',
            'import_name': 'torch', 
            'description': 'PyTorch - 深度学习框架，ComfyUI核心依赖',
            'critical': True
        },
        {
            'pip_name': 'numpy',
            'import_name': 'numpy',
            'description': 'NumPy - 数值计算基础库',
            'critical': True
        },
        {
            'pip_name': 'pillow',
            'import_name': 'PIL',
            'description': 'Pillow - 图像处理库',
            'critical': True
        },
        {
            'pip_name': 'opencv-python',
            'import_name': 'cv2',
            'description': 'OpenCV - 计算机视觉库，用于轮廓检测',
            'critical': True
        },
        {
            'pip_name': 'scikit-image',
            'import_name': 'skimage',
            'description': 'Scikit-Image - 图像处理库，用于SSIM计算',
            'critical': True
        },
        {
            'pip_name': 'scipy',
            'import_name': 'scipy',
            'description': 'SciPy - 科学计算库，用于高级图像处理',
            'critical': True
        }
    ]
    
    # 检查结果统计
    total_libs = len(required_libraries)
    installed_libs = 0
    failed_libs = []
    
    print("\n📦 检查依赖库安装状态:")
    print("-" * 60)
    
    for lib in required_libraries:
        pip_name = lib['pip_name']
        import_name = lib['import_name'] 
        description = lib['description']
        critical = lib.get('critical', False)
        
        print(f"\n🔗 {pip_name}")
        print(f"   📝 {description}")
        
        # 检查库是否已安装
        is_installed, version, error = check_library(pip_name, import_name)
        
        if is_installed:
            print(f"   ✅ 已安装 - 版本: {version}")
            installed_libs += 1
        else:
            print(f"   ❌ 未安装 - {error}")
            failed_libs.append((pip_name, critical))
            
            # 询问是否自动安装
            if critical:
                print(f"   ⚠️  这是核心依赖，节点无法正常工作")
                try:
                    response = input(f"   📥 是否立即安装 {pip_name}? (y/n): ").lower().strip()
                    if response in ['y', 'yes', '是']:
                        if install_library(pip_name):
                            # 重新检查安装结果
                            is_installed, version, _ = check_library(pip_name, import_name)
                            if is_installed:
                                print(f"   ✅ 安装成功 - 版本: {version}")
                                installed_libs += 1
                                failed_libs.remove((pip_name, critical))
                except KeyboardInterrupt:
                    print("\n\n⏹️  用户中断操作")
                    sys.exit(1)
                    
    # 生成检查报告
    print("\n" + "=" * 60)
    print("📊 依赖检查报告:")
    print(f"✅ 已安装: {installed_libs}/{total_libs}")
    print(f"❌ 缺失: {len(failed_libs)}")
    
    if failed_libs:
        print("\n⚠️  缺失的依赖库:")
        critical_missing = []
        optional_missing = []
        
        for lib_name, is_critical in failed_libs:
            if is_critical:
                critical_missing.append(lib_name)
            else:
                optional_missing.append(lib_name)
                
        if critical_missing:
            print("🚨 核心依赖 (必须安装):")
            for lib in critical_missing:
                print(f"   - {lib}")
                
        if optional_missing:
            print("🔶 可选依赖:")
            for lib in optional_missing:
                print(f"   - {lib}")
                
        print("\n🛠️  安装命令:")
        print("pip install " + " ".join([lib for lib, _ in failed_libs]))
        
        if critical_missing:
            print("\n❌ 由于缺少核心依赖，节点可能无法正常工作")
            return False
    else:
        print("\n🎉 所有依赖库已正确安装！")
        
    # 进行功能测试
    print("\n🧪 功能测试:")
    print("-" * 30)
    
    # 测试SSIM计算
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # 创建测试数据
        test_img1 = np.random.rand(100, 100).astype(np.float64)
        test_img2 = test_img1 + np.random.rand(100, 100) * 0.1
        
        # 测试SSIM计算（指定data_range参数）
        score = ssim(test_img1, test_img2, data_range=1.0)
        print(f"✅ SSIM计算测试通过 - 分数: {score:.3f}")
        
    except Exception as e:
        print(f"❌ SSIM计算测试失败: {e}")
        
    # 测试OpenCV轮廓检测
    try:
        import cv2
        import numpy as np
        
        # 创建测试图像
        test_img = np.zeros((100, 100), dtype=np.uint8)
        test_img[25:75, 25:75] = 255
        
        # 测试轮廓检测
        contours, _ = cv2.findContours(test_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✅ OpenCV轮廓检测测试通过 - 检测到 {len(contours)} 个轮廓")
        
    except Exception as e:
        print(f"❌ OpenCV轮廓检测测试失败: {e}")
        
    # 测试PyTorch张量操作
    try:
        import torch
        
        # 创建测试张量
        test_tensor = torch.rand(2, 64, 64, 3)
        batch_tensor = torch.stack([test_tensor[0], test_tensor[1]], dim=0)
        print(f"✅ PyTorch张量操作测试通过 - 张量形状: {batch_tensor.shape}")
        
    except Exception as e:
        print(f"❌ PyTorch张量操作测试失败: {e}")
        
    print("\n🎯 环境检查完成！")
    
    if installed_libs == total_libs:
        print("✅ 您的环境已准备就绪，可以正常使用 Batch Diff Prompt to Files 节点。")
        return True
    else:
        print("⚠️  请先安装缺失的依赖库，然后重新运行此检查脚本。")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  检查已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 检查脚本执行失败: {e}")
        sys.exit(1) 