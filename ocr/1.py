# create_test_image.py
import cv2
import numpy as np


def create_test_essay_image():
    # 创建白色背景
    img = np.ones((500, 700, 3), dtype=np.uint8) * 255

    # 作文内容
    essay_text = [
        "My Favorite Season",
        "",
        "My favorite season is spring. During spring,",
        "the weather becomes warm and flowers bloom.",
        "The trees turn green and birds sing happily.",
        "",
        "I enjoy going for walks in the park in spring.",
        "The air is fresh and the scenery is beautiful.",
        "Sometimes I have picnics with my family.",
        "",
        "Spring is a season of new beginnings. It",
        "makes me feel happy and energetic. I think",
        "spring is the best season of the year."
    ]

    # 添加文本到图片
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40
    for line in essay_text:
        if line.strip():
            cv2.putText(img, line, (30, y), font, 0.6, (0, 0, 0), 2)
        y += 35

    # 保存图片
    cv2.imwrite('test_essay.jpg', img)
    print("测试图片已创建: test_essay.jpg")


if __name__ == "__main__":
    create_test_essay_image()