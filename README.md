# 🖼️ 批量图片处理节点

简单易用的ComfyUI批量图片处理扩展，专为初学者设计！

## ✨ 功能特色

- 🚀 **简单易用** - 专为编程初学者设计，操作简单直观
- 📁 **智能加载** - 自动识别文件夹中的图片，支持多种格式
- 💾 **批量保存** - 一键保存处理后的图片，支持多种输出格式
- 📊 **信息展示** - 实时显示图片信息和处理状态
- 🔧 **灵活配置** - 支持文件名自定义、格式选择、质量调节

## 📝 TXT批量编辑器

除了ComfyUI节点，本项目还包含一个独立的**TXT批量编辑器**（`txt_batch_editor.html`），专为Kontext Prompt Files设计的高效文本编辑工具。

### ✨ 核心功能

#### 🌐 中英文双语界面
- **一键切换**：页面右上角优雅的语言切换按钮
- **完整翻译**：所有界面文本支持中英文切换
- **自动记忆**：保存用户语言偏好设置
- **实时响应**：无需刷新页面即可切换语言

#### 📁 智能文件管理
- **文件夹选择**：使用现代浏览器File System Access API
- **智能扫描**：自动识别.txt文件并加载内容
- **实时预览**：左侧文件列表，点击即可编辑
- **状态提示**：清楚显示已修改和未保存的文件

#### ✏️ 高效文本编辑
- **Monaco编辑器体验**：Courier New等宽字体，代码编辑器风格
- **实时保存**：支持单文件保存和批量保存
- **修改追踪**：自动检测文件变化，标记修改状态
- **快捷键支持**：Ctrl+S保存当前，Ctrl+Shift+S全部保存

#### 🔄 批量操作功能
- **批量替换**：在所有文件中查找并替换指定文本
- **统计信息**：实时显示总文件数、已修改、未保存数量
- **修改记录**：导出JSON格式的修改报告
- **文件夹刷新**：一键刷新文件列表

### 🎨 设计特色

#### 苹果风格界面
- **渐变背景**：优雅的紫色渐变背景
- **毛玻璃效果**：语言切换按钮的现代毛玻璃设计
- **卡片布局**：清晰的分区和圆角卡片设计
- **悬停动画**：平滑的交互动画和过渡效果

#### 响应式设计
- **多设备适配**：桌面、平板、手机完美显示
- **弹性布局**：自适应不同屏幕尺寸
- **移动优化**：移动设备上的特殊布局调整

### 🛠️ 使用方法

#### 1. 打开编辑器
```bash
# 直接在浏览器中打开
打开 txt_batch_editor.html

# 或通过本地服务器
python -m http.server 8000
# 然后访问 http://localhost:8000/txt_batch_editor.html
```

#### 2. 选择文件夹
- 点击"📁 选择文件夹"按钮
- 选择包含.txt文件的文件夹
- 系统自动扫描并加载所有.txt文件

#### 3. 编辑文件
- 在左侧文件列表中点击文件名
- 在中央编辑区域修改文本内容
- 系统自动检测并标记修改状态（文件名后显示*）

#### 4. 批量操作
```
批量替换：
├── 在右侧"查找内容"框输入要替换的文本
├── 在"替换为"框输入新文本
└── 点击"🔄 执行替换"

保存操作：
├── 💾 保存当前文件 - 保存正在编辑的文件
├── 📥 全部保存 - 保存所有修改的文件
└── 📋 导出修改记录 - 生成修改报告JSON文件
```

### 🔧 技术特性

#### 浏览器兼容性
- **推荐浏览器**：Chrome 86+、Edge 86+、Safari 15+
- **核心技术**：File System Access API、ES6+、CSS Grid
- **降级支持**：不支持的浏览器会显示友好提示

#### 安全特性
- **本地处理**：所有文件处理在本地完成，无网络传输
- **权限控制**：只能访问用户明确选择的文件夹
- **数据保护**：编辑内容仅存储在浏览器本地

#### 性能优化
- **智能加载**：按需加载文件内容
- **内存管理**：高效的文件内容缓存机制
- **批量处理**：优化的批量保存性能

### 📊 功能对比

| 功能特性 | TXT编辑器 | 传统编辑器 | 优势 |
|---------|---------|-----------|------|
| 批量文件处理 | ✅ | ❌ | 一次性处理多个文件 |
| 中英文界面 | ✅ | ❌ | 国际化支持 |
| 文件状态追踪 | ✅ | 部分 | 清楚显示修改状态 |
| 批量查找替换 | ✅ | 部分 | 跨文件批量操作 |
| 浏览器内运行 | ✅ | ❌ | 无需安装软件 |
| 修改记录导出 | ✅ | ❌ | 详细的操作记录 |

### 💡 使用技巧

#### 1. 工作流程建议
```
推荐流程：
1. 备份原始文件 → 避免意外丢失
2. 选择工作文件夹 → 包含需要编辑的.txt文件
3. 批量替换常见内容 → 提高效率
4. 逐个检查修改 → 确保准确性
5. 全部保存 → 应用所有修改
6. 导出修改记录 → 保留操作历史
```

#### 2. 快捷键使用
- `Ctrl + S` / `Cmd + S`：保存当前文件
- `Ctrl + Shift + S` / `Cmd + Shift + S`：保存所有文件
- `Ctrl + F` / `Cmd + F`：聚焦到查找框

#### 3. 文件管理
- 使用有意义的文件名便于识别
- 定期备份重要的prompt文件
- 利用修改记录功能追踪变更历史

## 📦 包含节点

### 基础批量处理

### 🖼️ 批量图片加载器 (BatchImageLoader)
从文件夹批量加载图片，自动处理不同格式和尺寸

**功能特点:**
- 支持常见图片格式：JPG、PNG、BMP、TIFF、WebP
- 自动统一图片尺寸（用于批处理）
- 灵活的排序方式：按文件名、大小、日期
- 可选择性调整图片尺寸
- 详细的加载信息显示

**参数说明:**
- `folder_path` - 图片文件夹路径（支持相对和绝对路径）
- `file_filter` - 支持的文件格式（用逗号分隔）
- `sort_by` - 排序方式（文件名/大小/日期）
- `limit` - 限制加载数量（0=不限制）
- `resize_to` - 统一调整尺寸（0=保持原始）

**输出:**
- `images` - 批量图片数据
- `file_info` - 加载信息文本
- `count` - 成功加载的图片数量

### 💾 批量图片保存器 (BatchImageSaver)
将处理后的图片批量保存到指定文件夹

**功能特点:**
- 支持多种输出格式：PNG、JPG、WebP、BMP
- 智能文件命名（前缀+编号）
- 可调节的图片质量
- 自动创建输出文件夹
- 详细的保存状态反馈

**参数说明:**
- `images` - 要保存的批量图片
- `output_folder` - 输出文件夹路径
- `filename_prefix` - 文件名前缀
- `format` - 输出格式
- `quality` - 图片质量（1-100）
- `start_number` - 起始编号
- `pad_zeros` - 编号补零位数

**输出:**
- `save_info` - 保存信息文本
- `saved_count` - 成功保存的图片数量

### 📊 批量图片信息 (BatchImageInfo)
显示当前批次图片的详细信息

**功能特点:**
- 显示图片数量、尺寸、通道数
- 内存占用估算
- 像素值统计信息
- 设备和数据类型信息

### 🔍 批量图像差异分析到Prompt文件 (Kontext Style) V2.0 **🆕 UPGRADED!**
基于[HuggingFace Kontext Bench](https://huggingface.co/datasets/black-forest-labs/kontext-bench)风格的专业图像差异分析节点，生成指令式编辑prompt文件

**V2.0 新功能特点:**
- 🎯 **精确变化类型识别**: 自动分类物体移除/添加/修改/风格转换/全局变化
- 📊 **Kontext Bench风格prompt**: 生成符合HuggingFace标准的指令式描述
- 🧠 **多层次分析引擎**: 结合SSIM、像素差异、轮廓检测的高级算法
- 🎨 **智能风格检测**: 精确识别全局风格变化vs局部物体修改
- 📈 **置信度评估**: 为每个分析结果提供可信度评分
- 🔬 **科学计算支持**: 集成SciPy高级图像处理功能

**参数说明:**
- `source_folder` - 原图所在的文件夹路径
- `target_folder` - 目标图所在的文件夹路径  
- `output_folder` - 生成的.txt指令文件保存路径（默认：output/kontext_prompts）
- `source_suffix` - 识别原图文件的后缀标识（默认：_R）
- `target_suffix` - 识别目标图文件的后缀标识（默认：_T）
- `ssim_threshold` - SSIM相似性阈值（0.0-1.0，默认：0.8）
- `contour_min_area` - 物体检测最小轮廓面积（50-5000，默认：200）
- `style_sensitivity` - 风格变化检测敏感度（0.1-1.0，默认：0.3）
- `enable_debug` - 启用调试模式，输出详细分析信息

**输出:**
- `target_images_batch` - 所有成功处理的目标图像批处理张量
- `analysis_report` - 详细的Kontext风格分析报告

**文件命名规范:**
```
source_folder/
├── image001_R.png    # 原图
├── image002_R.jpg    # 原图
└── image003_R.png    # 原图

target_folder/
├── image001_T.png    # 目标图（对应image001_R.png）
├── image002_T.jpg    # 目标图（对应image002_R.jpg）
└── image003_T.png    # 目标图（对应image003_R.png）

output_folder/
├── image001_T.txt    # 生成的指令文件
├── image002_T.txt    # 内容示例："make the cat fatter" (Kontext Bench风格)
└── image003_T.txt    # 5类变化自动识别：移除/添加/修改/风格转换/全局变化
```

### Flux-Kontext 专用批量处理

### 🖼️ 批量Flux-Kontext图片缩放 (BatchFluxKontextImageScale)
专门用于Flux-Kontext模型的批量图片缩放预处理

**功能特点:**
- 专门优化Flux-Kontext模型的图片尺寸要求
- 智能像素对齐（16px倍数）
- 多种缩放模式：适应、裁剪、拉伸
- 保持原始比例或强制正方形
- 详细的缩放信息记录

**参数说明:**
- `images` - 批量图片输入
- `target_size` - 目标尺寸（推荐1024px）
- `scale_method` - 缩放方式（fit/crop/stretch）
- `align_to_multiple` - 像素对齐倍数（推荐16）

**输出:**
- `scaled_images` - 缩放后的批量图片
- `scale_info` - 详细缩放信息

### ⚡ 批量Flux-Kontext采样器 (BatchFluxKontextSampler)
使用Flux-Kontext模型进行批量AI图片生成

**功能特点:**
- 专门适配Flux-Kontext模型参数
- 智能分批处理，避免显存不足
- 支持多种采样器和调度器
- 独立种子管理，确保结果可重现
- 实时处理进度显示

**参数说明:**
- `model` - Flux-Kontext模型
- `positive/negative` - 正面/负面条件
- `latent_images` - 潜在空间图片
- `steps` - 采样步数（推荐20）
- `guidance` - 引导强度（推荐2.5）
- `batch_size` - 分批大小（推荐1-4）

**输出:**
- `samples` - 采样后的潜在空间
- `sampling_info` - 详细采样信息

### 🔄 Flux-Kontext批量工作流 (BatchFluxKontextWorkflow)
一键完成从图片加载到Flux-Kontext处理的完整工作流

**功能特点:**
- 集成完整的Flux-Kontext处理流程
- 自动化图片缩放→编码→采样→解码
- 智能参数优化和错误处理
- 详细的处理状态反馈
- 一个节点完成所有处理

**参数说明:**
- `images` - 输入批量图片
- `model/vae` - Flux-Kontext模型和VAE
- `positive/negative` - 条件文本
- `steps/guidance/denoise` - 生成参数
- `target_size/batch_size` - 处理参数

**输出:**
- `processed_images` - 处理完成的图片
- `workflow_info` - 完整工作流信息

## 🚀 使用教程

### 基础批量处理工作流

1. **加载图片**
   ```
   BatchImageLoader节点
   ├── folder_path: "input/my_photos"  
   ├── file_filter: "jpg,png"
   └── limit: 10
   ```

2. **查看信息**（可选）
   ```
   BatchImageInfo节点
   └── 连接到加载器的images输出
   ```

3. **图片处理**
   ```
   在这里插入你的图片处理节点
   例如：调色、滤镜、放大等
   ```

4. **批量保存**
   ```
   BatchImageSaver节点
   ├── images: 来自处理节点
   ├── output_folder: "output/processed"
   ├── filename_prefix: "enhanced_"
   └── format: "png"
   ```

### Kontext Bench风格图像差异分析工作流 **🆕 V2.0 UPGRADED!**

1. **准备图像对**
   ```
   准备文件结构:
   input/
   ├── sources/          # 原图文件夹
   │   ├── photo_001_R.png
   │   ├── photo_002_R.jpg
   │   └── photo_003_R.png
   └── targets/          # 目标图文件夹
       ├── photo_001_T.png  # 编辑后的图片
       ├── photo_002_T.jpg  # 如：给猫添加帽子
       └── photo_003_T.png  # 如：改变汽车颜色
   ```

2. **执行Kontext风格分析**
   ```
   BatchImageDifferenceToPromptFiles (Kontext Style)节点
   ├── source_folder: "input/sources"
   ├── target_folder: "input/targets"  
   ├── output_folder: "output/kontext_prompts"
   ├── source_suffix: "_R"
   ├── target_suffix: "_T"
   ├── ssim_threshold: 0.8 (SSIM相似性阈值)
   ├── contour_min_area: 200 (物体检测最小面积)
   ├── style_sensitivity: 0.3 (风格变化敏感度)
   └── enable_debug: false
   ```

3. **查看Kontext Bench风格结果**
   ```
   输出文件:
   output/kontext_prompts/
   ├── photo_001_T.txt  # "add a hat" (物体添加)
   ├── photo_002_T.txt  # "change the car to red" (物体修改)
   └── photo_003_T.txt  # "convert to cartoon style" (风格转换)
   
   节点输出:
   ├── target_images_batch → 连接到PreviewImage查看处理的目标图像
   └── analysis_report → 连接到ShowText显示Kontext风格分析报告
   
   分析报告包含:
   ├── 变化类型统计 (移除/添加/修改/风格转换/全局变化)
   ├── 置信度评分
   └── 参考HuggingFace Kontext Bench数据集
   ```

4. **依赖安装**（V2.0新增SciPy）
   ```bash
   pip install scikit-image opencv-python scipy pillow torch numpy
   
   # 或运行检查脚本
   python custom_nodes/batch_image_processor/check_dependencies.py
   ```

### Flux-Kontext批量增强工作流

1. **模型准备**
   ```
   NunchakuFluxDiTLoader节点
   ├── model_path: "flux1-dev-kontext_fp8_scaled.safetensors"
   ├── cache_threshold: 0.12
   └── attention: "nunchaku-fp16"
   
   DualCLIPLoader节点
   ├── clip_name1: "clip_l.safetensors"
   └── clip_name2: "t5xxl_fp8_e4m3fn.safetensors"
   ```

2. **批量加载和缩放**
   ```
   BatchImageLoader节点 → BatchFluxKontextImageScale节点
   ├── target_size: 1024
   ├── scale_method: "fit"
   └── align_to_multiple: 16
   ```

3. **设置提示词**
   ```
   CLIPTextEncode节点（正面）
   └── text: "enhance quality, detailed, professional"
   
   CLIPTextEncode节点（负面）  
   └── text: "blurry, low quality, artifacts"
   ```

4. **批量处理**
   ```
   BatchFluxKontextSampler节点
   ├── steps: 20
   ├── guidance: 2.5
   ├── denoise: 0.8
   └── batch_size: 2
   ```

5. **批量保存**
   ```
   BatchImageSaver节点
   ├── output_folder: "output/flux_enhanced"
   └── format: "png"
   ```

### 路径设置示例

**相对路径（推荐）:**
- `input/my_images` - ComfyUI目录下的input/my_images文件夹
- `output/results` - ComfyUI目录下的output/results文件夹

**绝对路径:**
- `C:/Users/用户名/Pictures/photos` - Windows系统
- `/home/用户名/Pictures/photos` - Linux系统

## 💡 使用技巧

### 1. 路径设置
- 使用相对路径更方便移动项目
- 确保文件夹存在或节点会自动创建
- 中文路径需要确保系统支持

### 2. 批量处理优化
- 使用resize_to参数统一图片尺寸
- 限制加载数量避免内存不足
- 选择合适的输出格式节省空间

### 3. 文件管理
- 使用有意义的文件名前缀
- 设置合适的编号位数便于排序
- 定期清理输出文件夹

### 4. 性能建议
- 大批量图片建议分批处理
- PNG格式质量最好但文件较大
- JPG格式适合照片，quality设置80-95

### 5. Kontext Bench风格分析技巧
- **SSIM阈值调优**: ssim_threshold=0.8适合大多数场景，降低数值检测更细微差异
- **物体检测优化**: contour_min_area=200可过滤小噪声，提高值只检测显著物体变化
- **风格敏感度**: style_sensitivity=0.3平衡物体vs风格检测，降低值更偏向物体变化
- **文件命名**: 保持前缀一致，如 `photo_001_R.png` 对应 `photo_001_T.png`
- **调试模式**: 启用debug可查看SSIM分数、轮廓统计和变化类型置信度
- **格式统一**: 建议原图和目标图使用相同的文件格式以提高匹配准确性
- **参考标准**: 生成的指令遵循HuggingFace Kontext Bench数据集风格

## 🔧 故障排除

### 常见问题

**Q: 提示"文件夹不存在"**
- 检查路径是否正确
- 确保使用正确的路径分隔符（/）
- 尝试使用绝对路径

**Q: 图片加载失败**
- 检查文件格式是否在支持列表中
- 确认图片文件没有损坏
- 检查文件权限

**Q: 保存失败**
- 检查输出文件夹是否有写入权限
- 确保磁盘空间充足
- 检查文件名是否包含非法字符

**Q: 内存不足**
- 减少一次加载的图片数量
- 使用resize_to参数降低图片尺寸
- 关闭其他占用内存的程序

## 📈 版本信息

- **当前版本**: 2.1.0 🆕
- **兼容性**: ComfyUI最新版本
- **依赖库**: PIL (Pillow), torch, numpy, scikit-image, opencv-python, scipy
- **最新功能**: 
  - Kontext Bench风格指令式prompt生成 ✨
  - 中英文双语TXT批量编辑器 🌐

### 版本历史
- **v2.1.0** (最新) - 🌟 **新增独立工具**: 中英文双语TXT批量编辑器
  - 专业的文本批量编辑界面
  - 完整中英文双语支持
  - 现代浏览器文件系统API
  - 苹果风格响应式设计
  - 智能批量查找替换功能
  - 实时文件状态追踪
- **v2.0.0** - 🔥 **重大升级**: Kontext Bench风格指令式分析引擎
  - 精确物体级别变化检测（添加/移除/修改）
  - 智能风格转换识别
  - 参考HuggingFace Kontext Bench数据集标准
  - 置信度评估系统
  - 5类变化自动分类
  - 新增SciPy科学计算支持
- **v1.2.0** - 新增批量图像差异分析节点，支持SSIM算法和启发式prompt生成
- **v1.1.0** - 添加Flux-Kontext模型批量处理支持
- **v1.0.0** - 基础批量图片加载和保存功能

## 🤝 支持与反馈

这是一个开源项目，欢迎提出建议和问题！

### 计划功能

#### ComfyUI节点功能
- [x] 支持Flux-Kontext模型批量处理 ✅
- [x] 图像差异分析和prompt生成 ✅
- [x] Kontext Bench风格指令式prompt生成 ✅
- [ ] 集成更多Kontext Bench类别（Character Reference、Text Editing）
- [ ] AI增强的物体识别和命名
- [ ] 支持更多AI模型（SDXL、SD3等）
- [ ] 添加批量ControlNet处理
- [ ] 支持视频文件批量处理
- [ ] 添加图片重命名工具
- [ ] 支持更多图片格式
- [ ] 集成图片元数据读取
- [ ] 添加批量图片质量评估

#### TXT批量编辑器功能
- [x] 中英文双语界面 ✅
- [x] 批量文件编辑和保存 ✅
- [x] 智能查找替换功能 ✅
- [x] 苹果风格响应式设计 ✅
- [ ] 支持更多语言（日语、韩语、德语等）
- [ ] 语法高亮和代码编辑器功能
- [ ] 文件内容搜索和过滤
- [ ] 正则表达式支持
- [ ] 文件比较和差异显示
- [ ] 自动备份和版本控制
- [ ] 插件系统和自定义功能
- [ ] 集成AI prompt优化建议
- [ ] 导入导出多种文件格式
- [ ] 协作编辑和分享功能

## 👨‍💻 作者信息

**作者**: Asir
- 🐧 **QQ交流群**: 960598442  
- 💬 **Discord**: asir_50811
- 📱 **公众号**: AsirAI
- 📖 **公众号**: [AsirAI](https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA)

欢迎加入交流群讨论ComfyUI使用技巧和工作流分享！

---

**祝您使用愉快！** 🎉 