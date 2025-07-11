#!/usr/bin/env python3
"""
ComfyUI Manager 上传验证脚本
检查所有必要文件是否完整，依赖是否正确

作者: Asir
QQ交流群: 960598442
Discord: asir_50811
公众号: AsirAI
"""

import os
import json
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (缺失)")
        return False

def check_json_validity(file_path):
    """检查JSON文件格式是否正确"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"✅ JSON格式检查: {file_path}")
        return True
    except Exception as e:
        print(f"❌ JSON格式错误: {file_path} - {e}")
        return False

def check_python_imports():
    """检查核心Python导入"""
    try:
        # 检查节点文件是否存在并可读取
        if not os.path.exists('batch_image_difference_to_prompt_files.py'):
            print("❌ 主节点文件不存在")
            return False
            
        # 简单检查文件内容
        with open('batch_image_difference_to_prompt_files.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'NODE_CLASS_MAPPINGS' in content and 'BatchImageDifferenceToPromptFiles' in content:
                print("✅ 节点文件格式检查通过")
                print("   - BatchImageDifferenceToPromptFiles")
                return True
            else:
                print("❌ 节点文件格式不正确")
                return False
                
    except Exception as e:
        print(f"❌ 节点文件检查失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🔍 ComfyUI Manager 上传前验证")
    print("=" * 50)
    
    # 必需文件检查
    required_files = [
        ("__init__.py", "节点注册文件"),
        ("batch_image_difference_to_prompt_files.py", "主节点实现"),
        ("README.md", "中文说明文档"),
        ("README_EN.md", "英文说明文档"),
        ("LICENSE", "许可证文件"),
        ("requirements.txt", "Python依赖列表"),
        ("pyproject.toml", "项目配置文件"),
        (".gitignore", "Git忽略文件"),
        ("node_list.json", "ComfyUI Manager配置")
    ]
    
    print("\n📁 文件完整性检查:")
    missing_files = 0
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            missing_files += 1
    
    # JSON文件格式检查
    print("\n📋 配置文件格式检查:")
    json_files = ["node_list.json", "batch_diff_prompt_example_workflow.json"]
    json_errors = 0
    for json_file in json_files:
        if os.path.exists(json_file):
            if not check_json_validity(json_file):
                json_errors += 1
    
    # Python导入检查
    print("\n🐍 Python模块检查:")
    import_success = check_python_imports()
    
    # 项目结构检查
    print("\n📦 项目结构检查:")
    current_dir = Path(".")
    print(f"✅ 项目根目录: {current_dir.absolute()}")
    
    # 版本一致性检查
    print("\n🔢 版本一致性检查:")
    versions = {}
    
    # 检查__init__.py版本
    try:
        with open('__init__.py', 'r', encoding='utf-8') as f:
            content = f.read()
            for line in content.split('\n'):
                if '__version__' in line and '=' in line:
                    versions['__init__.py'] = line.split('=')[1].strip().strip('"').strip("'")
                    break
            else:
                versions['__init__.py'] = 'Unknown'
    except:
        versions['__init__.py'] = 'Unknown'
    
    # 检查pyproject.toml版本
    if os.path.exists('pyproject.toml'):
        try:
            with open('pyproject.toml', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.strip().startswith('version ='):
                        versions['pyproject.toml'] = line.split('=')[1].strip().strip('"')
                        break
        except:
            versions['pyproject.toml'] = 'Unknown'
    
    # 检查node_list.json版本
    if os.path.exists('node_list.json'):
        try:
            with open('node_list.json', 'r') as f:
                data = json.load(f)
                versions['node_list.json'] = data['custom_nodes'][0].get('version', 'Unknown')
        except:
            versions['node_list.json'] = 'Unknown'
    
    # 过滤掉未知版本
    known_versions = {k: v for k, v in versions.items() if v != 'Unknown'}
    version_consistent = len(set(known_versions.values())) <= 1 if known_versions else True
    for file, version in versions.items():
        print(f"   {file}: {version}")
    
    if version_consistent:
        print("✅ 版本号一致")
    else:
        print("❌ 版本号不一致")
    
    # 生成报告
    print("\n" + "=" * 50)
    print("📊 验证报告:")
    print(f"缺失文件: {missing_files}")
    print(f"JSON错误: {json_errors}")
    print(f"导入测试: {'通过' if import_success else '失败'}")
    print(f"版本一致: {'是' if version_consistent else '否'}")
    
    if missing_files == 0 and json_errors == 0 and import_success and version_consistent:
        print("\n🎉 所有检查通过！项目已准备好上传到ComfyUI Manager")
        print("\n📋 下一步操作:")
        print("1. 确保代码已推送到GitHub仓库")
        print("2. 创建Release标签（如 v2.1.0）")
        print("3. 在ComfyUI Manager仓库提交PR")
        print("4. 提供node_list.json文件内容")
        return True
    else:
        print("\n⚠️ 存在问题，请修复后重新验证")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 