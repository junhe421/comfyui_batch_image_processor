# 批量图片处理节点包
# 为ComfyUI提供简单易用的批量图片加载和保存功能
#
# 作者信息:
# 作者: Asir
# QQ交流群: 960598442
# Discord: asir_50811
# 公众号: AsirAI
# 公众号: https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA

from .batch_image_difference_to_prompt_files import NODE_CLASS_MAPPINGS as diff_mappings, NODE_DISPLAY_NAME_MAPPINGS as diff_display_mappings

# 使用存在的节点映射
NODE_CLASS_MAPPINGS = diff_mappings
NODE_DISPLAY_NAME_MAPPINGS = diff_display_mappings

# 节点包信息
__version__ = "2.1.0"
__author__ = "Asir"
__contact__ = {
    "qq": "960598442",
    "discord": "asir_50811", 
    "wechat_public": "AsirAI",
    "wechat_url": "https://mp.weixin.qq.com/s/iRb8y5LKW46pXbYxTN0iRA"
}
__description__ = "Kontext Bench风格批量图片处理工具，支持指令式prompt生成"

# 导出节点映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 