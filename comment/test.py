import os


def check_environment():
    """检查环境变量设置"""
    api_key = os.getenv("QIANWEN_API_KEY")

    if not api_key:
        print("❌ 环境变量 QIANWEN_API_KEY 未设置")
        return False

    if api_key == "your_api_key_here":
        print("❌ 环境变量 QIANWEN_API_KEY 仍然是默认值，请替换为真实的API密钥")
        return False

    print(f"✅ 环境变量 QIANWEN_API_KEY 已设置: {api_key[:10]}...")
    return True


# 在运行主程序前先检查
if __name__ == "__main__":
    check_environment()
