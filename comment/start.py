from comment_generator import CommentGenerator, ScoringResult, WritingPrompt, EssayAnalysis
import config

def test_comment_generation():
    """测试评语生成功能"""

    # 1. 初始化生成器 - 使用你的有效API密钥
    generator = CommentGenerator(api_key=config.Config.QIANWEN_API_KEY)

    # 2. 准备更真实的测试数据
    essay_text = """
I like summer the most. Summer is my favorite season because the weather is warm and sunny. I can go swimming in the pool and eat ice cream. Last summer, I go to the beach with my family. We play in the water and build sandcastles. I also enjoy summer vacation because I don't have to go to school. I can sleep late and play with my friends. Sometimes it is too hot, but I still love summer. Overall, summer is the best season for me.
"""

    scores = ScoringResult(
        overall=75,
        vocabulary=70,
        grammar=65,  # 语法分数较低，因为文中有时态错误
        structure=80,
        content=80
    )

    prompt = WritingPrompt(
        topic="My Favorite Season",
        requirements="Write about your favorite season and explain why you like it"
    )

    analysis = EssayAnalysis(
        strengths=[
            "文章结构清晰，有明确的开头、主体和结尾",
            "使用了丰富的个人经历和具体例子",
            "观点明确，情感表达真实"
        ],
        weaknesses=[
            "存在时态不一致的问题",
            "部分句子结构简单，可以更丰富",
            "词汇可以更加多样化"
        ],
        specific_errors=[
            {"type": "grammar", "detail": "时态错误", "example": "I go to the beach",
             "correction": "I went to the beach"},
            {"type": "grammar", "detail": "时态错误", "example": "We play in the water",
             "correction": "We played in the water"},
            {"type": "vocabulary", "detail": "重复使用简单词汇", "example": "like", "suggestion": "prefer, enjoy, love"}
        ],
        topic_relevance=0.9,
        vocabulary_diversity=0.65
    )

    # 3. 生成评语
    print("正在生成评语，请稍候...")
    result = generator.generate_comments(
        essay_text=essay_text,
        scores=scores,
        writing_prompt=prompt,
        analysis=analysis
    )

    # 4. 美化输出结果
    print("\n" + "=" * 80)
    print("📝 作文评语生成结果")
    print("=" * 80)

    print("\n🔤 英文评语:")
    print("-" * 40)
    print(result['english_comment'])

    print("\n🀄 中文评语:")
    print("-" * 40)
    print(result['chinese_comment'])

    print("\n💡 改进建议:")
    print("-" * 40)
    print(result['suggestions'])

    print("\n📊 生成质量报告:")
    print("-" * 40)
    quality_report = result.get('quality_report', {})
    print(f"质量评分: {quality_report.get('quality_score', 'N/A')}/100")
    print(f"使用的API: {result.get('api_source', 'unknown')}")
    print(f"生成状态: {'成功' if result.get('success', False) else '失败'}")

    if 'feedback' in quality_report:
        print("\n详细反馈:")
        for fb in quality_report['feedback']:
            print(f"  • {fb}")

    return result


if __name__ == "__main__":
    # 运行测试
    test_comment_generation()