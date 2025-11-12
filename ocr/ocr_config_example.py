# 示例 OCR 配置（保存为 ocr_config.py 或任意路径）
# 支持字段：input (必须), out (可选), lang (可选), gpu 或 use_gpu (可选)

config = {
    'input': 'essay1.png',  # 示例，可以是图片文件或目录
    'out': 'ocr_output',
    'lang': 'en',
    'gpu': False,
}

# 也可以把配置写成顶级变量而非 dict，例如：
# input = 'path/to/image_or_dir'
# out = 'ocr_output'
# lang = 'en'
# gpu = False
