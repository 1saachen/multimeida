from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import json
import os
from typing import Dict, List


class EnglishEssayGrader:
    def __init__(self):
        # 使用新版PaddleOCR初始化参数
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",  # 使用英文模型
            # device="gpu"  # 如果有GPU可以启用
        )

        # 评分权重配置
        self.weights = {
            'grammar': 0.3,
            'vocabulary': 0.2,
            'structure': 0.2,
            'content': 0.3
        }

        # 词汇库
        self.advanced_vocab = {
            'excellent', 'outstanding', 'remarkable', 'significant',
            'consequently', 'furthermore', 'moreover', 'nevertheless',
            'perspective', 'dilemma', 'phenomenon', 'contemporary'
        }

    def extract_text_from_image(self, image_path: str) -> str:
        """
        使用新版PaddleOCR API从图片中提取文本 - 修复版本
        """
        try:
            # 使用新版predict方法
            result = self.ocr.predict(image_path)
            full_text = []

            # 处理结果 - 修复：使用正确的属性名
            for res in result:
                # 从OCRResult对象中提取文本，使用rec_texts而不是txt
                if hasattr(res, 'rec_texts') and res.rec_texts:
                    for text in res.rec_texts:
                        if text and text.strip():  # 只保留非空文本
                            full_text.append(text.strip())
                else:
                    # 备用方法：尝试从字典中获取
                    if isinstance(res, dict) and 'rec_texts' in res:
                        for text in res['rec_texts']:
                            if text and text.strip():
                                full_text.append(text.strip())

            return ' '.join(full_text) if full_text else ""

        except Exception as e:
            print(f"OCR识别错误: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def debug_ocr_structure(self, image_path: str):
        """
        调试函数：打印OCR结果的结构
        """
        try:
            result = self.ocr.predict(image_path)
            print("OCR结果结构调试:")
            print(f"结果类型: {type(result)}")

            for i, res in enumerate(result):
                print(f"\n--- 第{i + 1}个结果 ---")
                print(f"结果类型: {type(res)}")
                print(f"所有属性: {dir(res)}")

                # 检查常见属性
                for attr in ['rec_texts', 'txt', 'text', 'boxes', 'scores']:
                    if hasattr(res, attr):
                        value = getattr(res, attr)
                        print(f"{attr}: {type(value)} - {value}")

                # 如果是字典类型
                if isinstance(res, dict):
                    print("字典键:", res.keys())

            return result
        except Exception as e:
            print(f"调试失败: {e}")
            return None

    def extract_text_robust(self, image_path: str) -> str:
        """
        更健壮的文本提取方法
        """
        try:
            result = self.ocr.predict(image_path)
            full_text = []

            for res in result:
                # 方法1: 尝试rec_texts属性
                if hasattr(res, 'rec_texts') and res.rec_texts:
                    full_text.extend([t.strip() for t in res.rec_texts if t and t.strip()])

                # 方法2: 尝试直接访问文本数据
                elif hasattr(res, '__dict__'):
                    res_dict = res.__dict__
                    if 'rec_texts' in res_dict:
                        full_text.extend([t.strip() for t in res_dict['rec_texts'] if t and t.strip()])

                # 方法3: 如果是字典
                elif isinstance(res, dict) and 'rec_texts' in res:
                    full_text.extend([t.strip() for t in res['rec_texts'] if t and t.strip()])

            return ' '.join(full_text) if full_text else ""

        except Exception as e:
            print(f"文本提取失败: {e}")
            return ""

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = self.correct_common_errors(text)
        return text.strip()

    def correct_common_errors(self, text: str) -> str:
        """纠正常见OCR识别错误"""
        corrections = {
            'rn': 'm', 'cl': 'd', 'vv': 'w',
            'I O': '10', 'l O': '10', '|': 'I',
            '0': 'O', '1': 'I', 'acaderic': 'academic'  # 修正你日志中的错误
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def analyze_grammar(self, text: str) -> Dict:
        """语法分析"""
        if not text:
            return {'score': 0, 'total_sentences': 0, 'avg_sentence_length': 0, 'errors': {}}

        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_sentences = len(sentences)
        words = text.split()
        avg_sentence_length = len(words) / max(total_sentences, 1)

        errors = {
            'capitalization': len(re.findall(r'[a-z][.!?]\s+[a-z]', text)),
            'double_spaces': len(re.findall(r'  ', text)),
            'subject_verb_agreement': self.check_subject_verb_agreement(text)
        }

        total_errors = sum(errors.values())
        grammar_score = max(0, 100 - total_errors * 2)

        return {
            'score': round(grammar_score, 2),
            'total_sentences': total_sentences,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'errors': errors
        }

    def check_subject_verb_agreement(self, text: str) -> int:
        """检查主谓一致错误"""
        errors = 0
        patterns = [
            r'\b(he|she|it)\s+(do|have)\b',
            r'\b(I|you|we|they)\s+(does|has)\b'
        ]
        for pattern in patterns:
            errors += len(re.findall(pattern, text.lower()))
        return errors

    def analyze_vocabulary(self, text: str) -> Dict:
        """词汇分析"""
        if not text:
            return {'score': 0, 'total_words': 0, 'unique_words': 0, 'lexical_diversity': 0}

        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        total_words = len(words)
        if total_words == 0:
            return {'score': 0, 'total_words': 0, 'unique_words': 0, 'lexical_diversity': 0}

        unique_words = len(set(words))
        lexical_diversity = unique_words / total_words

        advanced_words_used = [word for word in words if word in self.advanced_vocab]
        advanced_ratio = len(advanced_words_used) / total_words

        vocabulary_score = min(100, (lexical_diversity * 60 + advanced_ratio * 40) * 100)

        return {
            'score': round(vocabulary_score, 2),
            'total_words': total_words,
            'unique_words': unique_words,
            'lexical_diversity': round(lexical_diversity, 3),
            'advanced_words_used': advanced_words_used
        }

    def analyze_structure(self, text: str) -> Dict:
        """结构分析"""
        if not text:
            return {'score': 0, 'sentences_count': 0, 'transitions_used': []}

        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        transition_words = [
            'first', 'second', 'finally', 'however', 'therefore',
            'moreover', 'furthermore', 'consequently', 'in conclusion'
        ]

        transitions_used = []
        for word in transition_words:
            if word in text.lower():
                transitions_used.append(word)

        structure_score = min(100, len(sentences) * 3 + len(transitions_used) * 5)

        return {
            'score': round(structure_score, 2),
            'sentences_count': len(sentences),
            'transitions_used': transitions_used
        }

    def analyze_content(self, text: str) -> Dict:
        """内容质量分析"""
        if not text:
            return {'score': 0, 'word_count': 0, 'feedback': '未识别到文本内容'}

        words = text.split()
        word_count = len(words)

        if word_count < 50:
            content_score = 50
        elif word_count < 100:
            content_score = 60
        elif word_count < 200:
            content_score = 75
        elif word_count < 300:
            content_score = 85
        else:
            content_score = 90

        return {
            'score': round(content_score, 2),
            'word_count': word_count,
            'feedback': self.generate_content_feedback(word_count)
        }

    def generate_content_feedback(self, word_count: int) -> str:
        if word_count < 50:
            return "文章过短，建议大幅扩展内容。"
        elif word_count < 100:
            return "文章较短，建议扩展内容。"
        elif word_count < 200:
            return "文章长度适中。"
        else:
            return "文章内容丰富。"

    def calculate_overall_score(self, scores: Dict) -> float:
        total = 0
        for category, score_info in scores.items():
            if category in self.weights:
                total += score_info['score'] * self.weights[category]
        return round(total, 2)

    def grade_essay(self, image_path: str) -> Dict:
        """主评分函数"""
        if not os.path.exists(image_path):
            return {"error": f"图片文件不存在: {image_path}"}

        print("正在识别图片中的文本...")

        # 先调试OCR结构
        self.debug_ocr_structure(image_path)

        # 使用健壮的文本提取方法
        raw_text = self.extract_text_robust(image_path)

        if not raw_text:
            # 备用方法：尝试直接提取
            raw_text = self.extract_text_from_image(image_path)

        if not raw_text:
            return {"error": "无法从图片中识别出文本"}

        print("识别到的文本:")
        print(raw_text)
        print("\n" + "=" * 50)

        processed_text = self.preprocess_text(raw_text)

        grammar_analysis = self.analyze_grammar(processed_text)
        vocabulary_analysis = self.analyze_vocabulary(processed_text)
        structure_analysis = self.analyze_structure(processed_text)
        content_analysis = self.analyze_content(processed_text)

        analysis_results = {
            'grammar': grammar_analysis,
            'vocabulary': vocabulary_analysis,
            'structure': structure_analysis,
            'content': content_analysis
        }

        overall_score = self.calculate_overall_score(analysis_results)

        return {
            'original_text': raw_text,
            'processed_text': processed_text,
            'overall_score': overall_score,
            'detailed_analysis': analysis_results,
            'word_count': len(processed_text.split())
        }

    def save_ocr_result(self, image_path: str, output_dir: str = "output"):
        """保存OCR的可视化结果和JSON数据"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            result = self.ocr.predict(image_path)

            for i, res in enumerate(result):
                # 保存可视化图片
                res.save_to_img(output_dir)
                # 保存JSON数据
                res.save_to_json(output_dir)
                # 打印结果
                res.print()

            print(f"OCR结果已保存到 {output_dir} 目录")

        except Exception as e:
            print(f"保存OCR结果失败: {e}")

    def print_results(self, results: Dict):
        """格式化输出结果"""
        if 'error' in results:
            print(f"错误: {results['error']}")
            return

        print("\n" + "=" * 60)
        print("           英语作文评分结果")
        print("=" * 60)

        print(f"\n📝 识别到的文本:")
        print(f"   {results['processed_text']}")

        print(f"\n📊 基本统计:")
        print(f"   总字数: {results['word_count']}")

        print(f"\n🎯 综合评分: {results['overall_score']}/100")

        print(f"\n📖 详细分析:")
        analysis = results['detailed_analysis']

        for category, details in analysis.items():
            print(f"\n  {category.upper()}分析:")
            for key, value in details.items():
                if key != 'score' and value:
                    if isinstance(value, list):
                        if value:
                            print(f"    {key}: {', '.join(value)}")
                    else:
                        print(f"    {key}: {value}")
            print(f"    评分: {details['score']}/100")

        print(f"\n💡 改进建议:")
        self.generate_improvement_suggestions(analysis)

    def generate_improvement_suggestions(self, analysis: Dict):
        suggestions = []
        if analysis['grammar']['score'] < 80:
            suggestions.append("• 注意语法准确性")
        if analysis['vocabulary']['score'] < 70:
            suggestions.append("• 尝试使用更多高级词汇")
        if analysis['structure']['score'] < 75:
            suggestions.append("• 加强文章结构，使用过渡词")
        if analysis['content']['score'] < 80:
            suggestions.append("• 丰富文章内容")

        if not suggestions:
            suggestions.append("• 继续保持，文章质量很好！")

        for suggestion in suggestions:
            print(suggestion)


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
    print("=" * 60)
    print("        英语作文自动评分系统 (修复版)")
    print("=" * 60)

    # 初始化评分器
    print("初始化评分系统...")
    grader = EnglishEssayGrader()

    # 检查是否有测试图片，如果没有则创建
    image_path = "sample_essay.jpg"
    if not os.path.exists(image_path):
        print("未找到测试图片，正在创建示例图片...")
        image_path = create_sample_image()
        if not image_path:
            custom_path = input("请手动输入作文图片路径: ")
            image_path = custom_path.strip() if custom_path.strip() else "sample_essay.jpg"

    if not os.path.exists(image_path):
        print(f"❌ 图片文件不存在: {image_path}")
        return

    print(f"使用图片: {image_path}")

    try:
        # 保存OCR的可视化结果
        print("正在生成OCR可视化结果...")
        grader.save_ocr_result(image_path, "ocr_output")

        # 进行作文评分
        results = grader.grade_essay(image_path)
        grader.print_results(results)

        # 保存评分结果
        with open('essay_score_result.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 评分结果已保存到 essay_score_result.json")

    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()