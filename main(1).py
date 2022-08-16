def Preprocess(image):
    """
    对图像进行预处理（二值化、去除噪声等）
    :param image: 原始图片（数据数组形式）
    :return: processed_image
    """
    pass


def ExtractExpression(processed_image):
    """
    提取图片里的算式
    :param processed_image: 处理后的图片（数据数组形式）
    :return: expression_images -> (location, split_image)
        location -> (x, y, width, height) 是该表达式坐标， split_image是提取出的表达式图片（数据数组形式）
    """
    pass


def CharaSplit(split_image):
    """
    将表达式分割为字符集
    :param split_image: 表达式图片
    :return: chara_images -> 分割后的字符集
    """
    pass


def ExpRecognition(model, chara_images):
    """
    对表达式进行预测
    :param model: 预先训练好的机器学习模型，要有 predict(chara_image)方法，返回该图片的字符表示
    :param chara_images: 字符图片集（按表达式从左到右顺序）
    :return: answer -> (right_answer, is_right) 等号左边表达式的正确答案及等式结果
    """
    pass


def Mark(image, location, answer):
    """
    框出图片中的表达式并添加结果
    :param image: 原始图片
    :param location:表达式的外框信息
    :param answer:结果
    :return:result_image -> 处理后的图片，提交给前端
    """


