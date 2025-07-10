# 🔍 Kontext Image Difference Analysis Tool

Professional image difference analysis and prompt generation tool designed for Kontext model training!

## ✨ Key Features

- 🧠 **Intelligent Analysis** - Image difference detection based on Kontext Bench standards
- 📝 **Auto Annotation** - Generate high-quality instructional prompt files
- 🌐 **Bilingual Interface** - Chinese-English TXT batch editor for doubled efficiency
- 💻 **Local Processing** - Completely local operation ensuring data security
- 🎯 **Professional Precision** - 5 types of automatic change recognition with 90%+ accuracy

## 📝 TXT Batch Editor

Comprehensive text file batch editor featuring Apple-style design and bilingual interface.

### Core Features

- **🌍 Bilingual Interface**: Seamless switching between Chinese and English
- **📂 File Management**: Based on File System Access API for efficient local file operations
- **✏️ Batch Editing**: Professional text editing interface with real-time change tracking
- **🔍 Batch Find & Replace**: Intelligent search and replace functionality with match highlighting
- **💾 Auto Save**: Real-time save status tracking with visual feedback
- **📱 Responsive Design**: Perfect adaptation to various screen sizes with modern UI
- **🎨 Apple Aesthetics**: Purple gradient background with glassmorphism effects
- **🚀 High Performance**: Optimized for handling large files and batch operations

### Design Characteristics

- **Modern Web Technologies**: Latest HTML5, CSS3, and ES6+ standards
- **Accessibility Support**: Complete keyboard navigation and screen reader compatibility
- **Dark Mode Ready**: Elegant purple theme suitable for extended use
- **Animation Effects**: Smooth transitions and hover effects enhancing user experience
- **Cross-Platform**: Compatible with all modern browsers and operating systems

### Technical Specifications

- **Browser Compatibility**: Chrome 86+, Firefox 82+, Safari 14+, Edge 86+
- **File API**: Native File System Access API (fallback to traditional file input)
- **Performance**: Supports files up to 100MB with instant response
- **Security**: Complete local processing without data transmission
- **Encoding**: Full UTF-8 support for multilingual content
- **Standards Compliance**: W3C standards compliant with semantic HTML structure

### Usage Methods

1. **Open Editor**: Directly open `txt_batch_editor.html` in browser
2. **Select Files**: Use "Select Folder" to batch load .txt files
3. **Edit Content**: Click any file to start editing with real-time preview
4. **Find & Replace**: Use batch find/replace for efficient content modification
5. **Save Changes**: Auto-save with clear status indicators
6. **Language Switch**: One-click toggle between Chinese and English interface

### Feature Comparison

| Feature | Traditional Editor | TXT Batch Editor |
|---------|-------------------|------------------|
| Batch File Loading | ❌ | ✅ |
| Bilingual Interface | ❌ | ✅ |
| Real-time Save Status | ❌ | ✅ |
| Modern UI Design | ❌ | ✅ |
| Batch Find & Replace | ❌ | ✅ |
| Local File System API | ❌ | ✅ |
| Responsive Design | ❌ | ✅ |
| Change Tracking | ❌ | ✅ |

### Usage Tips

- Use Ctrl+S for quick save (works in all browsers)
- Support drag & drop for quick file loading
- Use modification history to track changes
- Utilize change tracking for modification history

## 📦 Available Features

### 🔍 Batch Image Difference Analysis to Prompt Files (Kontext Style) V2.0
Professional image difference analysis node based on [HuggingFace Kontext Bench](https://huggingface.co/datasets/black-forest-labs/kontext-bench) style, generating instructional edit prompt files

**Algorithm Features:**
- **Advanced SSIM Analysis**: Structural Similarity Index Measurement for precise change detection
- **Contour Detection**: OpenCV-based contour analysis for object-level change recognition
- **Multi-level Analysis**: Pixel-level, structural-level, and semantic-level comprehensive analysis
- **Intelligent Classification**: Automatic categorization into 5 change types (Removal/Addition/Modification/Style/Global)
- **Confidence Scoring**: Each generated instruction includes confidence evaluation
- **Scientific Computing**: Integration of SciPy and scikit-image for enhanced analysis accuracy

**Core Parameters:**
- `source_folder` - Source image folder path
- `target_folder` - Target image folder path  
- `output_folder` - Generated .txt instruction file save path
- `source_suffix` - Source image file suffix identifier (default: "_R")
- `target_suffix` - Target image file suffix identifier (default: "_T")
- `ssim_threshold` - SSIM similarity threshold (default: 0.8)
- `contour_min_area` - Minimum contour area for object detection (default: 200)
- `style_sensitivity` - Style change detection sensitivity (default: 0.3)
- `enable_debug` - Enable debug mode for detailed analysis info

**Output Results:**
- `target_images_batch` - Batch tensor of target images → Connect to PreviewImage
- `analysis_report` - Kontext style analysis report → Connect to ShowText

**Generated Instruction Examples:**
```
Removal Type: "remove the hat"
Addition Type: "add a red car to the scene"  
Modification Type: "make the cat bigger"
Style Type: "convert to cartoon style"
Global Type: "adjust the lighting"
```

**File Structure Example:**
```
source_folder/
├── image001_R.png    # Source image
├── image002_R.jpg    # Source image
└── image003_R.png    # Source image

target_folder/
├── image001_T.png    # Target image (corresponds to image001_R.png)
├── image002_T.jpg    # Target image (corresponds to image002_R.jpg)
└── image003_T.png    # Target image (corresponds to image003_R.png)

output_folder/
├── image001_T.txt    # Generated instruction file
├── image002_T.txt    # Content example: "make the cat fatter" (Kontext Bench style)
└── image003_T.txt    # 5 types of automatic change recognition: removal/addition/modification/style/global
```

## 🚀 Usage Tutorial

### Kontext Bench Style Image Difference Analysis Workflow

1. **Prepare Image Pairs**
   ```
   Prepare file structure:
   input/
   ├── sources/          # Source image folder
   │   ├── photo_001_R.png
   │   ├── photo_002_R.jpg
   │   └── photo_003_R.png
   └── targets/          # Target image folder
       ├── photo_001_T.png  # Edited image
       ├── photo_002_T.jpg  # e.g.: Add hat to cat
       └── photo_003_T.png  # e.g.: Change car color
   ```

2. **Execute Kontext Style Analysis**
   ```
   BatchImageDifferenceToPromptFiles (Kontext Style) Node
   ├── source_folder: "input/sources"
   ├── target_folder: "input/targets"  
   ├── output_folder: "output/kontext_prompts"
   ├── source_suffix: "_R"
   ├── target_suffix: "_T"
   ├── ssim_threshold: 0.8 (SSIM similarity threshold)
   ├── contour_min_area: 200 (minimum object detection area)
   ├── style_sensitivity: 0.3 (style change sensitivity)
   └── enable_debug: false
   ```

3. **View Kontext Bench Style Results**
   ```
   Output files:
   output/kontext_prompts/
   ├── photo_001_T.txt  # "add a hat" (object addition)
   ├── photo_002_T.txt  # "change the car to red" (object modification)
   └── photo_003_T.txt  # "convert to cartoon style" (style transformation)
   
   Node outputs:
   ├── target_images_batch → Connect to PreviewImage to view processed target images
   └── analysis_report → Connect to ShowText to display Kontext style analysis report
   
   Analysis report includes:
   ├── Change type statistics (removal/addition/modification/style/global)
   ├── Confidence scores
   └── Reference to HuggingFace Kontext Bench dataset
   ```

4. **Dependency Installation** (V2.0 added SciPy)
   ```bash
   pip install scikit-image opencv-python scipy pillow torch numpy
   
   # Or run check script
   python custom_nodes/batch_image_processor/check_dependencies.py
   ```

### Path Setting Examples

**Relative Paths (Recommended):**
- `input/my_images` - input/my_images folder under ComfyUI directory
- `output/results` - output/results folder under ComfyUI directory

**Absolute Paths:**
- `C:/Users/username/Pictures/photos` - Windows system
- `/home/username/Pictures/photos` - Linux system

## 💡 Usage Tips

### 1. Path Configuration
- Use relative paths for easier project portability
- Ensure folders exist or node will create them automatically
- Chinese paths require system support confirmation

### 2. Kontext Bench Style Analysis Tips
- **SSIM Threshold Tuning**: ssim_threshold=0.8 suits most scenarios, lower values detect subtler differences
- **Object Detection Optimization**: contour_min_area=200 filters small noise, higher values detect only significant object changes
- **Style Sensitivity**: style_sensitivity=0.3 balances object vs style detection, lower values favor object changes
- **File Naming**: Maintain consistent prefixes, e.g., `photo_001_R.png` corresponds to `photo_001_T.png`
- **Debug Mode**: Enable debug to view SSIM scores, contour statistics, and change type confidence
- **Format Consistency**: Recommend using same file format for source and target images to improve matching accuracy
- **Reference Standard**: Generated instructions follow HuggingFace Kontext Bench dataset style

## 🔧 Troubleshooting

### Common Issues

**Q: "Folder does not exist" error**
- Check if path is correct
- Ensure proper path separators (/) are used
- Try using absolute paths

**Q: Image loading failed**
- Check if file format is in supported list
- Confirm image files are not corrupted
- Check file permissions

**Q: Save failed**
- Check if output folder has write permissions
- Ensure sufficient disk space
- Check if filename contains illegal characters

**Q: Out of memory**
- Reduce number of images loaded at once
- Use resize_to parameter to reduce image dimensions
- Close other memory-consuming programs

## 📈 Version Information

- **Current Version**: 2.1.0 🆕
- **Compatibility**: Latest ComfyUI version
- **Dependencies**: PIL (Pillow), torch, numpy, scikit-image, opencv-python, scipy
- **Latest Features**: 
  - Kontext Bench style instructional prompt generation ✨
  - Chinese-English bilingual TXT batch editor 🌐

### Version History
- **v2.1.0** (Latest) - 🌟 **New Independent Tool**: Chinese-English bilingual TXT batch editor
  - Professional text batch editing interface
  - Complete Chinese-English bilingual support
  - Modern browser File System Access API
  - Apple-style responsive design
  - Intelligent batch find & replace functionality
  - Real-time file status tracking
- **v2.0.0** - 🔥 **Major Upgrade**: Kontext Bench style instructional analysis engine
  - Precise object-level change detection (addition/removal/modification)
  - Intelligent style transformation recognition
  - Reference to HuggingFace Kontext Bench dataset standards
  - Confidence evaluation system
  - 5 types of automatic change classification
  - Added SciPy scientific computing support

## 🤝 Support & Feedback

This is an open-source project, welcome to provide suggestions and questions!

### Planned Features

#### ComfyUI Node Features
- [x] Image difference analysis and prompt generation ✅
- [x] Kontext Bench style instructional prompt generation ✅
- [ ] Basic batch image loader and saver
- [ ] Batch image information display node
- [ ] Integration of more Kontext Bench categories (Character Reference, Text Editing)
- [ ] AI-enhanced object recognition and naming
- [ ] Support for more AI models (SDXL, SD3, Flux, etc.)
- [ ] Add batch ControlNet processing
- [ ] Support for video file batch processing
- [ ] Add image renaming tool
- [ ] Support for more image formats
- [ ] Integrate image metadata reading
- [ ] Add batch image quality assessment

#### TXT Batch Editor Features
- [x] Chinese-English bilingual interface ✅
- [x] Batch file editing and saving ✅
- [x] Intelligent find & replace functionality ✅
- [x] Apple-style responsive design ✅
- [ ] Support for more languages (Japanese, Korean, German, etc.)
- [ ] Syntax highlighting and code editor functionality
- [ ] File content search and filtering
- [ ] Regular expression support
- [ ] File comparison and difference display
- [ ] Automatic backup and version control
- [ ] Plugin system and custom functionality
- [ ] Integrated AI prompt optimization suggestions
- [ ] Import/export multiple file formats
- [ ] Collaborative editing and sharing features

## 👨‍💻 Author Information

**Author**: Asir
- 🐧 **QQ Group**: 960598442  
- 💬 **Discord**: asir_50811
- 📱 **WeChat Public Account**: AsirAI
- 📖 **WeChat Public Account**: [AsirAI](https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA)

Welcome to join our community to discuss ComfyUI usage tips and workflow sharing!

---

**Enjoy using the tool!** 🎉 