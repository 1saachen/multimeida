"""
英文手写作文评分系统 - 评语生成模块
集成通义千问API，生成个性化双语评语
"""

import json
import re
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CommentGenerator")


@dataclass
class ScoringResult:
    """评分结果数据类"""
    overall: int
    vocabulary: int
    grammar: int
    structure: int
    content: int


@dataclass
class EssayAnalysis:
    """作文分析结果数据类"""
    strengths: List[str]
    weaknesses: List[str]
    specific_errors: List[Dict]
    topic_relevance: float = 0.0
    vocabulary_diversity: float = 0.0


@dataclass
class WritingPrompt:
    """写作题目数据类"""
    topic: str
    requirements: str = ""


class QianWenAPI:
    """通义千问API调用封装类"""

    def __init__(self, api_key: str,
                 base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def call(self, prompt: str, model: str = "qwen-turbo", temperature: float = 0.7) -> str:
        """
        调用通义千问API

        Args:
            prompt: 输入的提示文本
            model: 模型名称
            temperature: 生成温度

        Returns:
            API返回的文本内容
        """
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的英语写作教师，擅长给出具体、建设性的写作反馈。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": 2000
            }
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            if "output" in result and "text" in result["output"]:
                return result["output"]["text"]
            else:
                logger.error(f"API响应格式异常: {result}")
                raise Exception(f"API响应格式异常: {result}")

        except requests.exceptions.RequestException as e:
            logger.error(f"API调用失败: {e}")
            raise Exception(f"API调用失败: {e}")

    def call_with_retry(self, prompt: str, max_retries: int = 3, delay: float = 1.0) -> str:
        """带重试机制的API调用"""
        for attempt in range(max_retries):
            try:
                return self.call(prompt)
            except Exception as e:
                logger.warning(f"API调用失败，第{attempt + 1}次重试: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # 指数退避
                else:
                    raise e


class PromptEngineer:
    """Prompt工程类"""

    @staticmethod
    def build_comment_prompt(essay_text: str, scores: ScoringResult, prompt: WritingPrompt,
                             analysis: EssayAnalysis) -> str:
        """构建评语生成Prompt"""

        # 根据分数范围确定作文水平
        score_level = "高分" if scores.overall >= 85 else "中等" if scores.overall >= 70 else "低分"

        prompt_template = f"""
你是一位英语写作专家。请根据以下完整信息，为这篇{score_level}作文生成专业、具体、建设性的反馈。

【作文题目】
{prompt.topic}
{prompt.requirements if prompt.requirements else "无特殊要求"}

【学生作文原文】
{essay_text}

【评分详情】
- 综合分数：{scores.overall}/100
- 词汇维度：{scores.vocabulary}/100
- 语法维度：{scores.grammar}/100  
- 结构维度：{scores.structure}/100
- 内容维度：{scores.content}/100

【具体分析结果】
主要优点：
{PromptEngineer._format_list(analysis.strengths)}

需要改进的问题：
{PromptEngineer._format_list(analysis.weaknesses)}

{PromptEngineer._format_errors(analysis.specific_errors)}

请严格按照以下结构和要求生成反馈：

1. 英文评语（150-200字）：
   - 开头给予积极的肯定
   - 具体指出2-3个优点，要引用原文中的实际例子
   - 详细分析2-3个主要问题，提供具体的错误示例和改进建议
   - 结尾给予鼓励和期望

2. 中文翻译版本：
   - 准确翻译英文评语内容
   - 保持专业性和教育性

3. 具体行动建议（3条）：
   - 每条建议要具体可行
   - 针对作文中的实际问题

要求：
- 评语必须基于作文原文，引用具体的句子或表达
- 语言要专业、建设性、鼓励性
- 避免使用模板化语言，要个性化
- 中文翻译要准确自然

请按以下格式输出：
[ENGLISH_COMMENT]
这里放置英文评语
[/ENGLISH_COMMENT]

[CHINESE_COMMENT]
这里放置中文评语
[/CHINESE_COMMENT]

[SUGGESTIONS]
1. 第一条建议
2. 第二条建议  
3. 第三条建议
[/SUGGESTIONS]
"""
        return prompt_template

    @staticmethod
    def _format_list(items: List[str]) -> str:
        """格式化列表项"""
        return "\n".join(f"- {item}" for item in items) if items else "- 无"

    @staticmethod
    def _format_errors(errors: List[Dict]) -> str:
        """格式化错误信息"""
        if not errors:
            return "具体错误：\n- 无明显语法错误"

        error_text = "具体错误示例：\n"
        for i, error in enumerate(errors[:5], 1):  # 最多显示5个错误
            error_type = error.get('type', '未知')
            detail = error.get('detail', '')
            example = error.get('example', '')
            count = error.get('count', 1)

            error_text += f"{i}. [{error_type}] {detail}"
            if example:
                error_text += f" 示例: '{example}'"
            if count > 1:
                error_text += f" (出现{count}次)"
            error_text += "\n"

        return error_text


class CommentParser:
    """评语响应解析器"""

    @staticmethod
    def parse_response(response_text: str) -> Dict[str, str]:
        """解析API响应，提取各部分的评语"""

        # 定义提取模式
        patterns = {
            'english': r'\[ENGLISH_COMMENT\](.*?)\[/ENGLISH_COMMENT\]',
            'chinese': r'\[CHINESE_COMMENT\](.*?)\[/CHINESE_COMMENT\]',
            'suggestions': r'\[SUGGESTIONS\](.*?)\[/SUGGESTIONS\]'
        }

        parsed_comments = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                # 清理可能的标记残留
                content = re.sub(r'^\s*-\s*', '', content)  # 清理行首的-
                parsed_comments[key] = content
            else:
                logger.warning(f"未找到{key}部分，尝试启发式提取")
                parsed_comments[key] = CommentParser._heuristic_extract(response_text, key)

        return parsed_comments

    @staticmethod
    def _heuristic_extract(text: str, part: str) -> str:
        """启发式提取（当正则匹配失败时使用）"""
        if part == 'english':
            # 尝试提取第一段英文文本
            english_match = re.search(r'([A-Z][^。！？!?]*[.?!]){3,}', text)
            return english_match.group(0) if english_match else "未能提取英文评语"

        elif part == 'chinese':
            # 尝试提取中文字符段落
            chinese_match = re.search(r'([\u4e00-\u9fff]{10,}。[^A-Za-z]*)+', text)
            return chinese_match.group(0) if chinese_match else "未能提取中文评语"

        else:  # suggestions
            # 尝试提取编号列表
            suggestions_match = re.findall(r'\d+\.\s*([^\n]+)', text)
            if suggestions_match:
                return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(suggestions_match[:3]))
            return "1. 仔细检查语法错误\n2. 丰富词汇使用\n3. 改善文章结构"


class QualityChecker:
    """评语质量检查器"""

    @staticmethod
    def check_comment_quality(comments: Dict[str, str], essay_text: str, scores: ScoringResult) -> Dict:
        """检查生成的评语质量"""

        english_comment = comments.get('english', '')
        chinese_comment = comments.get('chinese', '')
        suggestions = comments.get('suggestions', '')

        quality_score = 0
        feedback = []

        # 1. 检查英文评语长度
        en_word_count = len(english_comment.split())
        if 100 <= en_word_count <= 250:
            quality_score += 25
            feedback.append("英文评语长度合适")
        else:
            feedback.append(f"英文评语长度不合适: {en_word_count}词")

        # 2. 检查中文评语存在性
        if chinese_comment and len(chinese_comment) > 20:
            quality_score += 25
            feedback.append("中文评语完整")
        else:
            feedback.append("中文评语不完整")

        # 3. 检查建议数量
        suggestion_count = len(re.findall(r'\d+\.', suggestions))
        if suggestion_count >= 3:
            quality_score += 25
            feedback.append("建议数量充足")
        else:
            feedback.append(f"建议数量不足: {suggestion_count}条")

        # 4. 检查原文引用
        reference_score = QualityChecker._check_essay_references(english_comment, essay_text)
        quality_score += reference_score * 25
        if reference_score > 0:
            feedback.append(f"引用了原文具体内容(得分: {reference_score:.2f})")
        else:
            feedback.append("未充分引用原文具体内容")

        # 5. 检查与分数的一致性
        consistency_score = QualityChecker._check_score_consistency(english_comment, scores)
        quality_score += consistency_score * 25
        if consistency_score > 0.5:
            feedback.append("评语与分数一致")
        else:
            feedback.append("评语与分数可能存在不一致")

        return {
            'quality_score': min(100, quality_score),
            'feedback': feedback,
            'word_count': en_word_count,
            'suggestion_count': suggestion_count,
            'references_essay': reference_score > 0
        }

    @staticmethod
    def _check_essay_references(comment: str, essay_text: str) -> float:
        """检查评语中引用原文的程度"""
        # 提取原文中的关键词（去除常见功能词）
        words = re.findall(r'\b[a-zA-Z]{4,}\b', essay_text.lower())
        unique_words = set(words)

        # 检查评语中是否包含原文词汇
        reference_count = 0
        for word in list(unique_words)[:20]:  # 检查前20个独特词汇
            if word in comment.lower():
                reference_count += 1

        return min(1.0, reference_count / 10)  # 最多1.0分

    @staticmethod
    def _check_score_consistency(comment: str, scores: ScoringResult) -> float:
        """检查评语语气与分数的一致性"""
        positive_words = ['excellent', 'good', 'well', 'strong', 'effective', 'impressive', 'commendable']
        negative_words = ['weak', 'poor', 'need improvement', 'should improve', 'problem', 'error']

        positive_count = sum(1 for word in positive_words if word in comment.lower())
        negative_count = sum(1 for word in negative_words if word in comment.lower())

        # 高分作文应该有更多积极评价
        expected_ratio = scores.overall / 100
        actual_ratio = positive_count / (positive_count + negative_count + 1)  # +1避免除零

        return 1.0 - min(1.0, abs(expected_ratio - actual_ratio))


class CommentGenerator:
    """评语生成主类"""

    def __init__(self, api_key: str):
        self.api = QianWenAPI(api_key)
        self.prompt_engineer = PromptEngineer()
        self.parser = CommentParser()
        self.quality_checker = QualityChecker()

    def generate_comments(self, essay_text: str, scores: ScoringResult,
                          writing_prompt: WritingPrompt, analysis: EssayAnalysis,
                          max_retries: int = 2) -> Dict:
        """
        生成个性化双语评语

        Args:
            essay_text: 作文文本
            scores: 评分结果
            writing_prompt: 写作题目
            analysis: 分析结果
            max_retries: 最大重试次数

        Returns:
            包含评语和元数据的字典
        """

        logger.info(f"开始生成评语，作文长度: {len(essay_text)}字符，总分: {scores.overall}")

        # 1. 构建Prompt
        prompt = self.prompt_engineer.build_comment_prompt(essay_text, scores, writing_prompt, analysis)

        # 2. 调用API（带重试）
        try:
            raw_response = self.api.call_with_retry(prompt, max_retries=max_retries)
            logger.info("API调用成功")
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return self._generate_fallback_comments(scores, analysis)

        # 3. 解析响应
        parsed_comments = self.parser.parse_response(raw_response)

        # 4. 质量检查
        quality_report = self.quality_checker.check_comment_quality(parsed_comments, essay_text, scores)

        # 5. 组装结果
        result = {
            'success': True,
            'english_comment': parsed_comments.get('english', ''),
            'chinese_comment': parsed_comments.get('chinese', ''),
            'suggestions': parsed_comments.get('suggestions', ''),
            'quality_report': quality_report,
            'raw_response': raw_response,
            'prompt_used': prompt
        }

        logger.info(f"评语生成完成，质量评分: {quality_report['quality_score']}")

        return result

    def _generate_fallback_comments(self, scores: ScoringResult, analysis: EssayAnalysis) -> Dict:
        """生成降级方案的评语（当API调用失败时）"""
        logger.warning("使用降级方案生成评语")

        # 简单的模板化评语
        level = "优秀" if scores.overall >= 85 else "良好" if scores.overall >= 70 else "需要改进"

        english_comment = f"""Your essay shows {level.lower()} writing skills. """

        if analysis.strengths:
            english_comment += f"Strengths include: {', '.join(analysis.strengths[:2])}. "

        if analysis.weaknesses:
            english_comment += f"Areas for improvement: {', '.join(analysis.weaknesses[:2])}. "

        english_comment += "Keep practicing and you will make good progress."

        chinese_comment = f"""你的作文表现出{level}的写作水平。"""
        if analysis.strengths:
            chinese_comment += f"优点包括：{'，'.join(analysis.strengths[:2])}。"
        if analysis.weaknesses:
            chinese_comment += f"需要改进的方面：{'，'.join(analysis.weaknesses[:2])}。"
        chinese_comment += "继续练习，你会取得更大进步。"

        suggestions = "1. 多阅读范文积累语感\n2. 注意检查语法错误\n3. 尝试使用更多样的句式结构"

        return {
            'success': False,
            'english_comment': english_comment,
            'chinese_comment': chinese_comment,
            'suggestions': suggestions,
            'quality_report': {
                'quality_score': 60,
                'feedback': ['使用降级模板生成'],
                'word_count': len(english_comment.split()),
                'suggestion_count': 3,
                'references_essay': False
            },
            'raw_response': '',
            'prompt_used': '降级方案'
        }


# 使用示例
def main():
    """使用示例"""

    # 初始化生成器（需要替换为你的API Key）
    generator = CommentGenerator(api_key="your_qianwen_api_key_here")

    # 准备测试数据
    essay_text = """
In my opinion, technology have changed our lives in many ways. First, it make communication easier. We can talk to people from different countrys. Second, technology help us learn new things. For example, we can find informations on the internet. However, some people spends too much time on their phones. This is not good for their health. Overall, technology is useful but we should use it wisely.
"""

    scores = ScoringResult(
        overall=72,
        vocabulary=70,
        grammar=65,
        structure=80,
        content=75
    )

    writing_prompt = WritingPrompt(
        topic="Technology's Impact on Our Lives",
        requirements="Discuss both positive and negative effects of technology"
    )

    analysis = EssayAnalysis(
        strengths=[
            "文章结构清晰，有明确的开头、主体和结尾",
            "使用了恰当的连接词(First, Second, However, Overall)",
            "观点明确，有基本的论证结构"
        ],
        weaknesses=[
            "存在多处主谓一致错误",
            "部分词汇拼写错误",
            "论点展开不够充分，缺乏具体例子"
        ],
        specific_errors=[
            {"type": "grammar", "detail": "主谓一致错误", "example": "technology have"},
            {"type": "grammar", "detail": "主谓一致错误", "example": "it make"},
            {"type": "spelling", "detail": "拼写错误", "example": "countrys → countries"},
            {"type": "vocabulary", "detail": "用词不当", "example": "informations → information"}
        ],
        topic_relevance=0.85,
        vocabulary_diversity=0.62
    )

    # 生成评语
    result = generator.generate_comments(essay_text, scores, writing_prompt, analysis)

    # 输出结果
    print("=" * 50)
    print("英文评语:")
    print(result['english_comment'])
    print("\n" + "=" * 50)
    print("中文评语:")
    print(result['chinese_comment'])
    print("\n" + "=" * 50)
    print("改进建议:")
    print(result['suggestions'])
    print("\n" + "=" * 50)
    print(f"质量报告: {result['quality_report']}")


if __name__ == "__main__":
    main()