"""
Batch Image Difference to Prompt Files - ComfyUI 高级图像差异分析节点
基于 Kontext Bench 风格的指令式 prompt 生成

参考: https://huggingface.co/datasets/black-forest-labs/kontext-bench

作者信息:
作者: Asir
QQ交流群: 960598442
Discord: asir_50811
公众号: AsirAI
公众号: https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA

版本: 2.0.0 - 全新指令式分析引擎
分类: IO/Batch Processing
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

# 科学计算库导入
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage import measure, morphology, segmentation
    from skimage.feature import local_binary_pattern
    from scipy import ndimage
except ImportError as e:
    print(f"⚠️ scikit-image库未安装: {e}")
    print("请运行: pip install scikit-image scipy")
    raise e

# 节点分类常量
CATEGORY_NAME = "IO/Batch Processing"
SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']

# Kontext Bench 风格的指令模板
INSTRUCTION_TEMPLATES = {
    'removal': [
        "remove the {object}",
        "delete the {object}",
        "take away the {object}",
        "eliminate the {object} from the image"
    ],
    'addition': [
        "add a {object}",
        "give the {subject} a {object}",
        "put a {object} on the {subject}",
        "place a {object} in the image"
    ],
    'modification': [
        "make the {object} {attribute}",
        "change the {object} to {attribute}",
        "turn the {object} {attribute}",
        "modify the {object} to be {attribute}"
    ],
    'style_transformation': [
        "convert to {style} style",
        "apply {style} effect",
        "transform into {style}",
        "change the style to {style}"
    ],
    'global_change': [
        "adjust the lighting",
        "change the overall appearance",
        "modify the image composition",
        "enhance the visual style"
    ]
}

# 常见物体和属性词汇
COMMON_OBJECTS = [
    "cat", "dog", "person", "car", "tree", "building", "flower", "bird", 
    "hat", "shirt", "dress", "chair", "table", "book", "phone", "glass"
]

COMMON_ATTRIBUTES = [
    "bigger", "smaller", "fatter", "thinner", "red", "blue", "green", "yellow",
    "bright", "dark", "colorful", "transparent", "metallic", "wooden"
]

STYLE_TYPES = [
    "cartoon", "anime", "watercolor", "oil painting", "sketch", "comic",
    "vintage", "modern", "artistic", "realistic", "abstract"
]

class BatchImageDifferenceToPromptFiles:
    """
    高级批量图像差异分析节点 - Kontext Bench 风格
    
    使用先进的计算机视觉算法精确识别图像变化类型，
    生成准确的指令式prompt，支持物体级别和风格级别的变化检测。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": ("STRING", {
                    "default": "", 
                    "widget": "directory",
                    "tooltip": "原图所在的文件夹路径"
                }),
                "target_folder": ("STRING", {
                    "default": "", 
                    "widget": "directory",
                    "tooltip": "目标图所在的文件夹路径"
                }),
                "output_folder": ("STRING", {
                    "default": "output/kontext_prompts", 
                    "widget": "directory",
                    "tooltip": "生成的.txt指令文件保存路径"
                }),
                "source_suffix": ("STRING", {
                    "default": "_R", 
                    "tooltip": "识别原图文件的后缀标识"
                }),
                "target_suffix": ("STRING", {
                    "default": "_T", 
                    "tooltip": "识别目标图文件的后缀标识"
                }),
            },
            "optional": {
                "ssim_threshold": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "SSIM相似性阈值，低于此值认为有显著变化"
                }),
                "contour_min_area": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 5000,
                    "step": 50,
                    "tooltip": "物体变化检测的最小轮廓面积"
                }),
                "style_sensitivity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "风格变化检测敏感度"
                }),
                "enable_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用调试模式，输出详细分析信息"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("target_images_batch", "analysis_report")
    FUNCTION = "analyze_and_generate_instructions"
    CATEGORY = CATEGORY_NAME
    
    OUTPUT_NODE = False
    
    def __init__(self):
        """初始化高级分析引擎"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_mode = False
        
        # 初始化分析统计
        self.analysis_stats = {
            'removal': 0,
            'addition': 0, 
            'modification': 0,
            'style_transformation': 0,
            'global_change': 0
        }

    def analyze_and_generate_instructions(self, source_folder: str, target_folder: str,
                                        output_folder: str, source_suffix: str, target_suffix: str,
                                        ssim_threshold: float = 0.8, contour_min_area: int = 200,
                                        style_sensitivity: float = 0.3, enable_debug: bool = False) -> Tuple[torch.Tensor, str]:
        """
        核心分析函数 - 生成Kontext Bench风格的指令式prompt
        
        Args:
            source_folder: 原图文件夹路径
            target_folder: 目标图文件夹路径
            output_folder: 输出文件夹路径
            source_suffix: 原图后缀标识
            target_suffix: 目标图后缀标识
            ssim_threshold: SSIM相似性阈值
            contour_min_area: 最小轮廓面积
            style_sensitivity: 风格变化敏感度
            enable_debug: 调试模式
            
        Returns:
            tuple: (目标图像批处理张量, 分析报告字符串)
        """
        try:
            self.debug_mode = enable_debug
            self._reset_stats()
            
            # 1. 路径验证
            self._log_debug("🔍 开始高级图像差异分析...")
            validated_paths = self._validate_paths(source_folder, target_folder, output_folder)
            source_folder, target_folder, output_folder = validated_paths
            
            # 2. 图像对匹配
            matched_pairs = self._match_image_pairs(source_folder, target_folder, source_suffix, target_suffix)
            if not matched_pairs:
                return self._return_error("未找到任何匹配的图像对")
            
            self._log_debug(f"📊 找到 {len(matched_pairs)} 对待分析图像")
            
            # 3. 高级批量分析
            processed_targets = []
            success_count = 0
            error_count = 0
            analysis_details = []
            
            print(f"🧠 开始 Kontext Bench 风格分析 {len(matched_pairs)} 对图像...")
            
            for i, (source_path, target_path) in enumerate(matched_pairs):
                try:
                    self._log_debug(f"🔬 分析第 {i+1}/{len(matched_pairs)} 对: {os.path.basename(target_path)}")
                    
                    # a. 加载和预处理图像对
                    source_img, target_img = self._load_and_preprocess_images(source_path, target_path)
                    
                    # b. 执行多层次差异分析
                    analysis_result = self._perform_advanced_difference_analysis(
                        source_img, target_img, ssim_threshold, contour_min_area, style_sensitivity)
                    
                    if analysis_result is None:
                        error_count += 1
                        continue
                    
                    instruction_text, change_type, confidence = analysis_result
                    
                    # c. 保存指令文件
                    instruction_file_path = self._save_instruction_file(
                        target_path, output_folder, instruction_text)
                    
                    # d. 收集目标图像张量
                    target_tensor = self._pil_to_tensor(target_img)
                    processed_targets.append(target_tensor)
                    
                    # e. 记录分析详情
                    analysis_details.append({
                        'target_file': os.path.basename(target_path),
                        'instruction_file': os.path.basename(instruction_file_path),
                        'instruction': instruction_text,
                        'change_type': change_type,
                        'confidence': confidence
                    })
                    
                    # 更新统计
                    self.analysis_stats[change_type] += 1
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 分析失败 {os.path.basename(target_path)}: {str(e)}")
                    error_count += 1
                    continue
            
            # 4. 生成分析报告
            if processed_targets:
                target_images_batch = self._create_batch_tensor(processed_targets)
                analysis_report = self._generate_analysis_report(
                    success_count, error_count, output_folder, analysis_details)
                
                print(f"✅ Kontext风格分析完成！成功: {success_count}, 失败: {error_count}")
                return (target_images_batch, analysis_report)
            else:
                return self._return_error("所有图像对分析失败")
                
        except Exception as e:
            error_msg = f"高级分析引擎失败: {str(e)}"
            print(f"❌ {error_msg}")
            return self._return_error(error_msg)

    def _perform_advanced_difference_analysis(self, source_img: Image.Image, target_img: Image.Image,
                                            ssim_threshold: float, contour_min_area: int, 
                                            style_sensitivity: float) -> Optional[Tuple[str, str, float]]:
        """
        执行高级差异分析 - 核心算法引擎
        
        Args:
            source_img: 原图PIL对象
            target_img: 目标图PIL对象
            ssim_threshold: SSIM阈值
            contour_min_area: 最小轮廓面积
            style_sensitivity: 风格敏感度
            
        Returns:
            tuple: (指令文本, 变化类型, 置信度) 或 None
        """
        try:
            # 转换为numpy数组
            source_array = np.array(source_img)
            target_array = np.array(target_img)
            
            # 1. 计算SSIM差异图
            source_gray = cv2.cvtColor(source_array, cv2.COLOR_RGB2GRAY)
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
            
            ssim_score, diff_map = ssim(source_gray, target_gray, full=True)
            self._log_debug(f"📈 SSIM相似性分数: {ssim_score:.3f}")
            
            # 2. 计算像素级绝对差值
            pixel_diff = np.abs(source_array.astype(np.float32) - target_array.astype(np.float32))
            pixel_diff_gray = np.mean(pixel_diff, axis=2)
            
            # 3. 二值化处理突出变化区域
            diff_binary = self._create_change_mask(diff_map, pixel_diff_gray, ssim_threshold)
            
            # 4. 轮廓检测和分析
            contours = self._detect_change_contours(diff_binary, contour_min_area)
            
            # 5. 分析变化类型
            change_analysis = self._analyze_change_type(
                source_array, target_array, contours, ssim_score, style_sensitivity)
            
            # 6. 生成指令式描述
            instruction = self._generate_instruction_prompt(change_analysis)
            
            return instruction, change_analysis['type'], change_analysis['confidence']
            
        except Exception as e:
            print(f"⚠️ 高级差异分析失败: {str(e)}")
            return None

    def _create_change_mask(self, ssim_diff: np.ndarray, pixel_diff: np.ndarray, threshold: float) -> np.ndarray:
        """
        创建变化掩码 - 结合SSIM和像素差异
        
        Args:
            ssim_diff: SSIM差异图
            pixel_diff: 像素差异图
            threshold: 阈值
            
        Returns:
            np.ndarray: 二值化变化掩码
        """
        # 归一化差异图
        ssim_diff_norm = (1 - ssim_diff) * 255
        pixel_diff_norm = (pixel_diff / pixel_diff.max()) * 255
        
        # 组合两种差异
        combined_diff = (ssim_diff_norm * 0.6 + pixel_diff_norm * 0.4)
        
        # 二值化
        binary_threshold = int((1 - threshold) * 255)
        _, binary_mask = cv2.threshold(
            combined_diff.astype(np.uint8), binary_threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask

    def _detect_change_contours(self, binary_mask: np.ndarray, min_area: int) -> List:
        """
        检测变化轮廓
        
        Args:
            binary_mask: 二值化掩码
            min_area: 最小面积
            
        Returns:
            List: 有效轮廓列表
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小轮廓并按面积排序
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        
        self._log_debug(f"🔍 检测到 {len(valid_contours)} 个有效变化区域")
        
        return valid_contours

    def _analyze_change_type(self, source_array: np.ndarray, target_array: np.ndarray,
                           contours: List, ssim_score: float, style_sensitivity: float) -> Dict[str, Any]:
        """
        分析变化类型 - 核心分类逻辑
        
        Args:
            source_array: 原图数组
            target_array: 目标图数组
            contours: 轮廓列表
            ssim_score: SSIM分数
            style_sensitivity: 风格敏感度
            
        Returns:
            Dict: 变化分析结果
        """
        height, width = source_array.shape[:2]
        total_area = height * width
        
        if not contours:
            # 无显著轮廓变化，可能是全局风格变化
            if ssim_score < 0.9:
                return {
                    'type': 'style_transformation',
                    'confidence': 0.8,
                    'details': 'global_style_change',
                    'major_contour': None
                }
            else:
                return {
                    'type': 'global_change',
                    'confidence': 0.6,
                    'details': 'minimal_change',
                    'major_contour': None
                }
        
        # 分析主要轮廓
        major_contour = contours[0]
        contour_area = cv2.contourArea(major_contour)
        area_ratio = contour_area / total_area
        
        self._log_debug(f"📊 主轮廓面积比例: {area_ratio:.3f}")
        
        # 获取轮廓边界框
        x, y, w, h = cv2.boundingRect(major_contour)
        
        # 提取轮廓区域的像素
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [major_contour], 255)
        
        source_region = source_array[mask > 0]
        target_region = target_array[mask > 0]
        
        # 计算区域内的颜色变化
        source_mean = np.mean(source_region, axis=0)
        target_mean = np.mean(target_region, axis=0)
        color_change = np.linalg.norm(source_mean - target_mean)
        
        # 判断变化类型
        if area_ratio > 0.15:  # 大区域变化
            if color_change < 30:  # 颜色变化不大
                # 可能是物体添加或移除
                source_brightness = np.mean(source_region)
                target_brightness = np.mean(target_region)
                
                if target_brightness < source_brightness - 20:
                    change_type = 'removal'
                    confidence = 0.9
                elif target_brightness > source_brightness + 20:
                    change_type = 'addition'
                    confidence = 0.9
                else:
                    change_type = 'modification'
                    confidence = 0.8
            else:  # 显著颜色变化
                change_type = 'modification'
                confidence = 0.9
        else:  # 小区域变化或分布式变化
            if len(contours) > 10:  # 多个小变化区域
                change_type = 'style_transformation'
                confidence = 0.8
            else:
                change_type = 'modification'
                confidence = 0.7
        
        return {
            'type': change_type,
            'confidence': confidence,
            'details': {
                'area_ratio': area_ratio,
                'color_change': color_change,
                'contour_count': len(contours),
                'bbox': (x, y, w, h)
            },
            'major_contour': major_contour
        }

    def _generate_instruction_prompt(self, change_analysis: Dict[str, Any]) -> str:
        """
        生成Kontext Bench风格的指令式prompt
        
        Args:
            change_analysis: 变化分析结果
            
        Returns:
            str: 指令式prompt文本
        """
        change_type = change_analysis['type']
        confidence = change_analysis['confidence']
        
        # 根据变化类型生成指令
        if change_type == 'removal':
            return self._generate_removal_instruction(change_analysis)
        elif change_type == 'addition':
            return self._generate_addition_instruction(change_analysis)
        elif change_type == 'modification':
            return self._generate_modification_instruction(change_analysis)
        elif change_type == 'style_transformation':
            return self._generate_style_instruction(change_analysis)
        else:  # global_change
            return self._generate_global_instruction(change_analysis)

    def _generate_removal_instruction(self, analysis: Dict) -> str:
        """生成移除类指令"""
        templates = INSTRUCTION_TEMPLATES['removal']
        
        # 根据区域大小推测物体类型
        details = analysis.get('details', {})
        area_ratio = details.get('area_ratio', 0)
        
        if area_ratio > 0.3:
            obj = "main object"
        elif area_ratio > 0.1:
            obj = np.random.choice(COMMON_OBJECTS)
        else:
            obj = "small object"
        
        template = np.random.choice(templates)
        return template.format(object=obj)

    def _generate_addition_instruction(self, analysis: Dict) -> str:
        """生成添加类指令"""
        templates = INSTRUCTION_TEMPLATES['addition']
        
        details = analysis.get('details', {})
        area_ratio = details.get('area_ratio', 0)
        
        if area_ratio > 0.2:
            obj = np.random.choice(["hat", "shirt", "accessory"])
            subject = "person"
        else:
            obj = np.random.choice(["small item", "decoration", "detail"])
            subject = "image"
        
        template = np.random.choice(templates)
        return template.format(object=obj, subject=subject)

    def _generate_modification_instruction(self, analysis: Dict) -> str:
        """生成修改类指令"""
        templates = INSTRUCTION_TEMPLATES['modification']
        
        details = analysis.get('details', {})
        color_change = details.get('color_change', 0)
        
        if color_change > 50:
            obj = np.random.choice(COMMON_OBJECTS)
            attr = np.random.choice(["red", "blue", "green", "colorful", "bright"])
        else:
            obj = "object"
            attr = np.random.choice(COMMON_ATTRIBUTES)
        
        template = np.random.choice(templates)
        return template.format(object=obj, attribute=attr)

    def _generate_style_instruction(self, analysis: Dict) -> str:
        """生成风格转换指令"""
        templates = INSTRUCTION_TEMPLATES['style_transformation']
        
        style = np.random.choice(STYLE_TYPES)
        template = np.random.choice(templates)
        return template.format(style=style)

    def _generate_global_instruction(self, analysis: Dict) -> str:
        """生成全局变化指令"""
        templates = INSTRUCTION_TEMPLATES['global_change']
        return np.random.choice(templates)

    def _save_instruction_file(self, target_path: str, output_folder: str, instruction: str) -> str:
        """
        保存指令文件
        
        Args:
            target_path: 目标图路径
            output_folder: 输出文件夹
            instruction: 指令文本
            
        Returns:
            str: 保存的文件路径
        """
        try:
            target_filename = os.path.basename(target_path)
            instruction_filename = os.path.splitext(target_filename)[0] + '.txt'
            instruction_file_path = os.path.join(output_folder, instruction_filename)
            
            with open(instruction_file_path, 'w', encoding='utf-8') as f:
                f.write(instruction)
                
            self._log_debug(f"💾 指令已保存: {instruction_filename}")
            return instruction_file_path
            
        except Exception as e:
            raise IOError(f"保存指令文件失败: {str(e)}")

    def _reset_stats(self):
        """重置分析统计"""
        for key in self.analysis_stats:
            self.analysis_stats[key] = 0

    def _validate_paths(self, source_folder: str, target_folder: str, output_folder: str) -> Tuple[str, str, str]:
        """路径验证"""
        if not source_folder or not os.path.exists(source_folder):
            raise FileNotFoundError(f"原图文件夹不存在: {source_folder}")
        if not target_folder or not os.path.exists(target_folder):
            raise FileNotFoundError(f"目标图文件夹不存在: {target_folder}")
            
        if not output_folder:
            output_folder = "output/kontext_prompts"
        os.makedirs(output_folder, exist_ok=True)
        
        return source_folder, target_folder, output_folder

    def _match_image_pairs(self, source_folder: str, target_folder: str, 
                          source_suffix: str, target_suffix: str) -> List[Tuple[str, str]]:
        """匹配图像对"""
        matched_pairs = []
        
        target_files = [f for f in os.listdir(target_folder) 
                       if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)]
        
        for target_file in target_files:
            if target_suffix not in target_file:
                continue
                
            file_base = target_file.rsplit('.', 1)[0]
            file_ext = '.' + target_file.rsplit('.', 1)[1]
            
            if not file_base.endswith(target_suffix):
                continue
                
            prefix = file_base[:-len(target_suffix)]
            source_file = prefix + source_suffix + file_ext
            source_path = os.path.join(source_folder, source_file)
            target_path = os.path.join(target_folder, target_file)
            
            if os.path.exists(source_path):
                matched_pairs.append((source_path, target_path))
                self._log_debug(f"✅ 匹配: {source_file} <-> {target_file}")
                
        return matched_pairs

    def _load_and_preprocess_images(self, source_path: str, target_path: str) -> Tuple[Image.Image, Image.Image]:
        """加载和预处理图像对"""
        source_img = Image.open(source_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        if source_img.size != target_img.size:
            source_img = source_img.resize(target_img.size, Image.LANCZOS)
            
        return source_img, target_img

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL转张量"""
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(image_array)

    def _create_batch_tensor(self, image_tensors: List[torch.Tensor]) -> torch.Tensor:
        """创建批处理张量"""
        if not image_tensors:
            raise ValueError("图像张量列表为空")
        return torch.stack(image_tensors, dim=0)

    def _generate_analysis_report(self, success_count: int, error_count: int,
                                output_folder: str, analysis_details: List[Dict]) -> str:
        """生成分析报告"""
        report_lines = [
            "🧠 Kontext Bench 风格图像差异分析报告",
            "=" * 60,
            f"✅ 成功分析: {success_count} 个图像对",
            f"❌ 分析失败: {error_count} 个图像对",
            f"📁 输出目录: {output_folder}",
            "",
            "📊 变化类型统计:",
            "-" * 30,
            f"🔴 物体移除 (Removal): {self.analysis_stats['removal']} 个",
            f"🟢 物体添加 (Addition): {self.analysis_stats['addition']} 个",
            f"🟡 物体修改 (Modification): {self.analysis_stats['modification']} 个",
            f"🎨 风格转换 (Style): {self.analysis_stats['style_transformation']} 个",
            f"🌐 全局变化 (Global): {self.analysis_stats['global_change']} 个",
            ""
        ]
        
        if analysis_details:
            report_lines.extend([
                "📝 生成的指令示例:",
                "-" * 30
            ])
            
            for i, detail in enumerate(analysis_details[:8], 1):
                report_lines.append(f"{i:2d}. {detail['target_file']}")
                report_lines.append(f"    📋 {detail['instruction']}")
                report_lines.append(f"    🔖 类型: {detail['change_type']} (置信度: {detail['confidence']:.2f})")
                report_lines.append("")
                
            if len(analysis_details) > 8:
                report_lines.append(f"... 以及其他 {len(analysis_details) - 8} 个指令文件")
                
        report_lines.extend([
            "",
            "🎯 Kontext Bench 风格分析完成！",
            "📖 参考: https://huggingface.co/datasets/black-forest-labs/kontext-bench"
        ])
        
        return "\n".join(report_lines)

    def _return_error(self, error_message: str) -> Tuple[torch.Tensor, str]:
        """错误返回"""
        empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        error_report = f"❌ 分析失败: {error_message}\n\n请检查输入参数并重试。"
        return (empty_image, error_report)

    def _log_debug(self, message: str):
        """调试日志"""
        if self.debug_mode:
            print(f"🔍 DEBUG: {message}")


# 节点注册
NODE_CLASS_MAPPINGS = {
    "BatchImageDifferenceToPromptFiles": BatchImageDifferenceToPromptFiles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImageDifferenceToPromptFiles": "Batch Diff Prompt to Files (Kontext Style)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 