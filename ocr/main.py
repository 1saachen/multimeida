from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


class EnglishOCR:
    """精简后的 OCR 工具类：仅保留文本识别、调试与结果保存功能。"""

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        # 使用新版PaddleOCR初始化参数（保留英文模型）
        # expose lang 和 use_gpu 以便更灵活的配置
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            lang=lang,
            
        )

    def debug_ocr_structure(self, image_path: str):
        """打印 OCR 返回结果的结构，便于调试不同 PaddleOCR 版本的返回值格式。"""
        try:
            result = self.ocr.predict(image_path)
            print("OCR结果结构调试:")
            print(f"结果类型: {type(result)}")

            # 仅打印前几个条目的结构以避免输出过长
            for i, res in enumerate(result[:5]):
                print(f"\n--- 第{i + 1}个结果 ---")
                print(f"结果类型: {type(res)}")
                try:
                    print("repr:", repr(res))
                except Exception:
                    print(str(res))

                if isinstance(res, dict):
                    print("字典键:", list(res.keys()))
                else:
                    # 如果是对象，打印属性名
                    print("属性样例:", [a for a in dir(res) if not a.startswith("__")][:10])

            return result
        except Exception as e:
            print(f"调试失败: {e}")
            return None

    def extract_text(self, image_path: str) -> str:
        """尽量稳健地从 PaddleOCR 的返回值中提取文本字符串。"""
        try:
            result = self.ocr.predict(image_path)
            texts = []

            # 处理常见的返回格式
            for entry in result:
                # case A: entry 带有 rec_texts 属性（新版可能）
                if hasattr(entry, 'rec_texts') and entry.rec_texts:
                    texts.extend([t.strip() for t in entry.rec_texts if t and t.strip()])
                    continue

                # case B: entry 是 dict 并包含 rec_texts 或 texts
                if isinstance(entry, dict):
                    for key in ('rec_texts', 'texts', 'text'):
                        if key in entry and entry[key]:
                            # 可能是列表或单字符串
                            if isinstance(entry[key], (list, tuple)):
                                texts.extend([t.strip() for t in entry[key] if t and t.strip()])
                            else:
                                t = str(entry[key]).strip()
                                if t:
                                    texts.append(t)
                            break

                # case C: 很多旧版 PaddleOCR 返回的是 list of [box, (text, score)]
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    # 若第二项是 tuple 或 list 包含文本
                    maybe_text = entry[1]
                    if isinstance(maybe_text, (list, tuple)) and len(maybe_text) >= 1:
                        txt = str(maybe_text[0]).strip()
                        if txt:
                            texts.append(txt)
                    else:
                        t = str(maybe_text).strip()
                        if t:
                            texts.append(t)

            return ' '.join(texts) if texts else ""

        except Exception as e:
            print(f"OCR识别错误: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def serialize_result(self, result: Any) -> List[Dict[str, Any]]:
        """把 PaddleOCR 的返回结果规范化为列表的字典：{'bbox': [[x,y],...], 'text': str, 'conf': float}

        支持的常见返回格式：
        - list of [box, (text, score)]
        - dict 包含 'boxes'/'texts' 或 'rec_texts'
        - 对象形式（尝试使用 repr 后回退）
        """
        out: List[Dict[str, Any]] = []
        try:
            for entry in result:
                bbox = None
                text = None
                conf = None

                # case: list like [box, (text, score)]
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    maybe_box = entry[0]
                    maybe_txt = entry[1]
                    # box as list of points
                    try:
                        bbox = [list(map(float, p)) for p in maybe_box]
                    except Exception:
                        bbox = None

                    if isinstance(maybe_txt, (list, tuple)) and len(maybe_txt) >= 1:
                        text = str(maybe_txt[0])
                        try:
                            conf = float(maybe_txt[1]) if len(maybe_txt) > 1 else None
                        except Exception:
                            conf = None
                    else:
                        text = str(maybe_txt)

                # case: dict-like
                elif isinstance(entry, dict):
                    # common keys: 'box', 'bbox', 'points', 'boxes'
                    for k in ('box', 'bbox', 'points'):
                        if k in entry:
                            try:
                                bbox = [list(map(float, p)) for p in entry[k]]
                            except Exception:
                                bbox = None
                            break

                    # text
                    for k in ('rec_texts', 'texts', 'text', 'transcription'):
                        if k in entry and entry[k]:
                            if isinstance(entry[k], (list, tuple)):
                                # pick first or join
                                text = ' '.join([str(x) for x in entry[k]])
                            else:
                                text = str(entry[k])
                            break

                    # confidence
                    for k in ('confidence', 'conf', 'score'):
                        if k in entry:
                            try:
                                conf = float(entry[k])
                                break
                            except Exception:
                                conf = None

                # case: object with attributes
                else:
                    # try attributes
                    if hasattr(entry, 'boxes') and hasattr(entry, 'rec_texts'):
                        try:
                            bbox = [list(map(float, p)) for p in entry.boxes]
                        except Exception:
                            bbox = None
                        try:
                            text = ' '.join([str(t) for t in entry.rec_texts])
                        except Exception:
                            text = None

                out.append({'bbox': bbox, 'text': text, 'conf': conf, 'raw': repr(entry)})
        except Exception:
            # 极端容错：将整个 result 的 repr 返回
            out = [{'bbox': None, 'text': None, 'conf': None, 'raw': repr(result)}]

        return out

    def annotate_image(self, image_path: str, result: Any, out_path: str):
        """根据 result 在图像上绘制 bbox 与文字并保存为 out_path。尽量兼容不同返回格式。"""
        try:
            data = self.serialize_result(result)
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError(f"无法读取图片: {image_path}")

            h, w = img.shape[:2]
            for i, line in enumerate(data):
                bbox = line.get('bbox')
                text = line.get('text') or ''
                conf = line.get('conf')

                if bbox:
                    pts = np.array(bbox, dtype=np.int32)
                    if pts.ndim == 2 and pts.shape[1] == 2:
                        cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
                        # put text near top-left of bbox
                        tx, ty = int(pts[0][0]), int(pts[0][1]) - 10
                        if ty < 10:
                            ty = int(pts[0][1]) + 20
                        label = text if len(text) <= 60 else text[:57] + '...'
                        if conf is not None:
                            label = f"{label} ({conf:.2f})"
                        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # 保存
            cv2.imwrite(str(out_path), img)
            return True
        except Exception as e:
            print(f"标注图片失败: {e}")
            return False

    def save_raw_result(self, image_path: str, output_dir: str = "ocr_output"):
        """保存 OCR 的原始返回结果与提取到的文本为 JSON 文件。"""
        try:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            result = self.ocr.predict(image_path)
            extracted_text = self.extract_text(image_path)

            # 尝试把 result 转为可序列化结构（列表的 dict，每个包含 bbox/text/conf）
            serializable = self.serialize_result(result)

            out = {
                'image': str(image_path),
                'extracted_text': extracted_text,
                'lines': serializable,
            }

            # 以图片文件名为前缀保存到独立子目录，支持批量处理时不覆盖
            base = Path(image_path).stem
            save_dir = out_dir / base
            save_dir.mkdir(parents=True, exist_ok=True)

            out_path = save_dir / 'ocr_result.json'
            with out_path.open('w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            # 额外：保存带标注的图片（如可能）
            try:
                ann_path = save_dir / f"{base}_annotated.jpg"
                self.annotate_image(image_path, result, str(ann_path))
                print(f"带标注的图片已保存到: {ann_path}")
            except Exception:
                # 标注失败不要中断流程
                pass

            # 如果 PaddleOCR 的返回对象每一项提供了 save_to_img/save_to_json 方法，则调用它们（兼容你的原脚本）
            for i, entry in enumerate(result):
                try:
                    if hasattr(entry, 'print'):
                        try:
                            entry.print()
                        except Exception:
                            pass
                    if hasattr(entry, 'save_to_img'):
                        try:
                            entry.save_to_img(str(save_dir))
                        except Exception:
                            pass
                    if hasattr(entry, 'save_to_json'):
                        try:
                            entry.save_to_json(str(save_dir))
                        except Exception:
                            pass
                except Exception:
                    pass

            print(f"OCR结果已保存到: {out_path}")
            return out

        except Exception as e:
            print(f"保存 OCR 结果失败: {e}")
            return None


def create_sample_image():
    """创建一个示例图片用于测试"""
    try:
        # 创建一个空白图片
        img = np.ones((500, 700, 3), dtype=np.uint8) * 255  # 白色背景

        # 添加文本
        text_lines = [
            "The Importance of Learning English",
            "",
            "English is an international language that",
            "is widely used around the world. Learning",
            "English can open up many opportunities for",
            "people. It helps in communication with",
            "people from different countries and cultures.",
            "",
            "Moreover, English is the language of",
            "science and technology. Many books and",
            "research papers are written in English.",
            "Therefore, learning English is essential",
            "for academic success.",
            "",
            "In conclusion, English is very important",
            "in today's globalized world."
        ]

        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 0, 0)  # 黑色
        thickness = 1

        # 在图片上写文本
        y = 40
        for line in text_lines:
            if line:  # 非空行
                cv2.putText(img, line, (30, y), font, font_scale, font_color, thickness)
            y += 35

        # 保存图片
        cv2.imwrite('sample_essay.jpg', img)
        print("✅ 已创建示例图片: sample_essay.jpg")
        return 'sample_essay.jpg'

    except Exception as e:
        print(f"创建示例图片失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="英语作文 OCR 识别工具 (改进版)")
    parser.add_argument('input', nargs='?', default='essay1.png', help='图片路径或目录，默认 essay1.png')
    parser.add_argument('--out', '-o', default='ocr_output', help='结果保存目录')
    parser.add_argument('--lang', default='en', help='PaddleOCR 语言模型，例如 en、ch 等')
    parser.add_argument('--gpu', action='store_true', help='如果可用，启用 GPU')
    parser.add_argument('--debug-structure', action='store_true', help='打印 OCR 返回结构用于调试')
    args = parser.parse_args()

    print("初始化 OCR 系统...")
    ocr_tool = EnglishOCR(lang=args.lang, use_gpu=args.gpu)

    input_path = Path(args.input)

    # 如果输入是目录，则批量处理目录下的图片文件
    targets: List[Path] = []
    if input_path.is_dir():
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp'):
            targets.extend(sorted(input_path.glob(ext)))
    else:
        if not input_path.exists():
            print("未找到测试图片，正在创建示例图片...")
            sample = create_sample_image()
            input_path = Path(sample)

        if not input_path.exists():
            print(f"❌ 图片文件不存在: {input_path}")
            return
        targets = [input_path]

    for img in targets:
        try:
            print(f"\n处理图片: {img}")
            if args.debug_structure:
                ocr_tool.debug_ocr_structure(str(img))

            text = ocr_tool.extract_text(str(img))
            if not text:
                print("未能提取到文本")
            else:
                print("识别到的文本:\n")
                print(text)

            print(f"正在保存到: {args.out} ...")
            ocr_tool.save_raw_result(str(img), args.out)

        except Exception as e:
            print(f"处理 {img} 过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()