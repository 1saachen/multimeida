## multimeida

一个用于**英文作文自动评分**的实验性项目，当前仓库主要包含三部分能力：

- `ocr/`：作文图片 OCR 识别（PaddleOCR）
- `grade/`：作文自动评分（FusionGraphBERT + 图结构 + 特征工程）
- `comment/`：基于通义千问生成中英文评语

> 当前仓库内容以脚本与模型代码为主，未包含完整前端工程代码。

---

## 目录结构

```text
multimeida/
├── README.md
├── requirements.txt
├── ocr/
│   ├── main.py
│   └── ocr_config_example.py
├── grade/
│   ├── grade_run.py
│   └── training/train.py
└── comment/
    ├── comment_generator.py
    ├── config.py
    ├── start.py
    └── test.py
```

---

## 环境准备

### 1) Python 版本

建议 Python `3.10+`。

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含 OCR 与评语模块的基础依赖。  
若使用 `grade/` 模块，请额外安装对应深度学习依赖（如 `torch`、`torch-geometric`、`transformers`、`spacy`、`nltk`、`scikit-learn`、`textstat` 等）。

### 3) 安装 PaddlePaddle（用于 OCR）

CPU（示例）：

```bash
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

GPU（示例，CUDA 12.9）：

```bash
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

---

## 快速开始

以下命令均在仓库根目录执行。

### OCR：识别作文图片

```bash
python ocr/main.py ocr/essay2.jpg --out ocr/ocr_output --lang en
```

可选参数：

- `--gpu`：启用 GPU（环境支持时）
- `--debug-structure`：打印 PaddleOCR 返回结构

输出会包含：

- 提取文本 `ocr_result.json`
- 标注图片 `*_annotated.jpg`

### 评分：单篇作文打分

```bash
python grade/grade_run.py
```

该脚本默认从当前目录读取模型相关文件（例如 `fusion_graphbert_best.pth`、`feature_scaler.pkl`、`target_scaler.pkl`、`model_config.json`、`tokenizer/`）。

### 评语生成：基于分数与分析结果生成双语反馈

先设置 API Key：

```bash
export QIANWEN_API_KEY=your_real_api_key
```

检查环境变量：

```bash
python comment/test.py
```

运行示例：

```bash
python comment/start.py
```

---

## 模块说明

### `ocr/`

- `main.py`：OCR 主程序，支持单图/目录批量处理
- `ocr_config_example.py`：配置示例

### `grade/`

- `training/train.py`：训练脚本
- `grade_run.py`：推理/评分脚本（含示例）

### `comment/`

- `comment_generator.py`：评语生成核心逻辑（Prompt 构建、解析、质量检查、降级策略）
- `start.py`：评语生成测试入口
- `config.py`：通义千问参数配置

---

## 注意事项

- `grade/` 模块依赖较重，首次运行可能需要下载模型与 NLP 资源（如 tokenizer、spaCy 模型、nltk 数据）。
- 请勿将真实 API Key 写入代码文件，建议仅通过环境变量注入。
