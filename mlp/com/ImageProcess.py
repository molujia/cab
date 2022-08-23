import numpy as np
import cv2 as cv
import os
import shutil

from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from PIL import *

import pre_with_foldername
import time


def Preprocess(image):
    """
    对图像进行预处理
    :param image: 原始图片（单个，ndarray）
    :return: processed_image -> 处理后的图像
    """

    # 灰度二值化图像
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)[1]
    # 表达式一般由黑字组成，颠倒黑白，方便后续计算
    target = cv.bitwise_not(binary_image)

    return target


def RotateByText(image):
    """
    根据图片中文本的最小外接矩形进行倾斜矫正，只允许 5度以内的微调
    :param image:需要旋转的图片(已经过预处理，ndarray)
    :return: upright_image -> 旋转后的图片
    """

    # 求出最小外接矩形的倾斜角
    coords = np.column_stack(np.where(image > 0))
    angle = cv.minAreaRect(coords)[-1]

    # 对偏角进行更正
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 10:
        # 仿射变换，旋转图片
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        upright_image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        return upright_image
    else:
        return image


def ExtractExpression(processed_image):
    """
    提取图片里的算式
    :param processed_image: 处理后的单个图片 (ndarray)
    :return: expression_images -> (location, split_image)
        location -> (x, y, width, height) 是该表达式坐标 （ x、y 是矩形左上角坐标 ），
        split_image是提取出的表达式图片（已经经过了预处理，ndarray）
    """

    locations = []

    # 按行投影，rate是确定为空白区域的阈值比率，internal可以调整按行分割后子图的上下或左右距离
    ROW_RATE = 0.18
    COL_RATE = 0.2
    ROW_INTERNAL = 10
    COL_INTERNAL = 5

    # 水平方向得到的投影，列向量
    horizontal = np.sum(processed_image, axis=1)
    threshold_h = np.max(horizontal) * ROW_RATE

    # 遍历时需要的临时变量
    row_flag = False
    y_begin, y_end = [], []

    for i in range(len(horizontal)):
        if horizontal[i] > threshold_h and not row_flag:
            row_flag = True
            y_begin.append(i - ROW_INTERNAL)
        if horizontal[i] < threshold_h and row_flag:
            row_flag = False
            y_end.append(i + ROW_INTERNAL)

    # 膨胀需要的核（白底黑字，计算数值可能会偏大）
    kernel = np.ones((5, 5), dtype=np.uint8)

    # 进行列的遍历，得到单个表达式的位置信息
    for i in range(len(y_begin)):
        # 经过膨胀后的子图片
        line_image = cv.dilate(processed_image[y_begin[i]:y_end[i], :], kernel, iterations=8)
        # 按列投影得到的行向量
        vertical = np.sum(line_image, axis=0)
        threshold_v = np.max(vertical, axis=0) * COL_RATE
        # 按列遍历需要的临时变量
        col_flag = False
        x_begin, x_end = [], []

        for j in range(len(vertical)):
            if vertical[j] > threshold_v and not col_flag:
                col_flag = True
                x_begin.append(j - COL_INTERNAL)
            if vertical[j] < threshold_v and col_flag:
                col_flag = False
                x_end.append(j + COL_INTERNAL)
        for k in range(len(x_begin)):
            locations.append((x_begin[k], y_begin[i], x_end[k] - x_begin[k], y_end[i] - y_begin[i]))

    expression_images = [(loc, processed_image[loc[1]:loc[1] + loc[3], loc[0]:loc[0] + loc[2]]) for loc in locations]
    return expression_images


def CharaSplit(exp_image, location=None, image=None):
    """
    将表达式分割为字符集，利用投影操作
    :param image: 原始图片
    :param location: 如果输入父表达式位置信息，则显示分割框（测试用）
    :param exp_image: 经过预处理和旋转之后的表达式图片
    :return: chara_images -> 分割后的字符集图片
    """

    RATE = 0.05
    INTERNAL = 2

    # 提取字符竖向边界信息
    line = np.sum(exp_image, axis=0)
    threshold = np.max(line, axis=0) * RATE
    flag = False
    begin, end = [], []
    for i in range(len(line)):
        if line[i] > threshold and not flag:
            flag = True
            begin.append(i - INTERNAL)
        if line[i] < threshold and flag:
            flag = False
            end.append(i + INTERNAL)

    chara_images = [exp_image[:, x1:x2] for x1, x2 in zip(begin, end)]

    if location is not None:
        f = True
        for i in range(len(begin)):
            x1 = location[0] + begin[i]
            x2 = location[0] + end[i]
            y1 = location[1]
            y2 = location[1] + location[3]
            if f:
                # cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
                f = False
            else:
                # cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
                f = True

    return chara_images


def cv2ImgAddText(img, text, left, top, textColor=(255, 0, 0), textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("static/fonts/FRADM.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)


def Mark(image, location, answer):
    """
    框出图片中的表达式并添加结果
    :param image: 原始图片
    :param location:表达式的外框信息
    :param answer:表达式结果
    :return: result_image -> 处理后的图片，提交给前端
    """

    result_image = image.copy()
    for i in range(0, len(location)):
        cv.rectangle(result_image, (location[i][0], location[i][1]),
                     (location[i][0] + location[i][2], location[i][1] + location[i][3]), (0, 255, 255), 1)

    # answer应该是一个一维的字符数组，记录的是每个表达式对应的结果，有三种可能结果 '√' ’x‘ 'right_answer'

    for i in range(0, len(location)):
        # 字体大小
        textSize = location[i][3]
        # 根据结果设置字体颜色
        if answer[i] == "√":
            color = (0, 255, 0)
        elif answer[i] == "×":
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        left = location[i][0] + location[i][2]
        top = location[i][1]

        # 将结果写到原图上
        result_image = cv2ImgAddText(result_image, answer[i], left, top, color, textSize)
    return result_image


def Analyse(equation):
    """
    分析表达式的正误
    :param equation: 由图片转化的单个表达式
    :return: "√"、 "×"、"right_answer"
    """

    # equation应为字符数组格式，如 ['1','+','1','=','2']
    # 返回的表达式结果为一个字符 result
    estr = ''.join(equation)
    print(estr)
    result = ''
    global res
    if '=' in estr:
        str_arra = estr.split('=')
        e_str = str_arra[0]
        # print(e_str)
        r_str = str_arra[1]
        # print(r_str)
        if '÷' in e_str:
            s = e_str.split('÷')
            s1 = s[0]
            s2 = s[1]
            res = int((int(s1)) / (int(s2)))

        else:
            res = int(eval(e_str))

        if r_str == '':
            result = str(res)
        else:
            if str(res) == r_str:
                result = '√'
            else:
                result = '×'
    else:
        result = 'wrong'
    return result


def getResult(filename):
    # 导入图片，转化为 ndarray
    images_dir = 'static' + os.sep + 'images'
    images_path = images_dir + os.sep + filename

    image = cv.imread(images_path)

    # 进行图片预处理
    processed_image = Preprocess(image)

    # 对图片整体进行旋转
    rota_image = RotateByText(processed_image)

    # 将图片里的表达式提取出来，得到表达式矩阵（包含位置信息）
    expressions = ExtractExpression(rota_image)

    # 位置信息
    locations = [i[0] for i in expressions]
    # # 分割得到的表达式数据集
    split_image = [i[1] for i in expressions]

    # 将单个表达式进行分割  chara_sets.shape ->（表达式，单个字符，字符数据行，字符数据列）四维
    chara_sets = [CharaSplit(exp) for exp in split_image]
    for i in range(len(split_image)):
        temp = CharaSplit(split_image[i], locations[i], image)

    filename = r'expressiontest'  # 加r
    isExists = os.path.exists(filename)
    if isExists:
        shutil.rmtree(filename)
    picture_name = 'test'

    ans = []
    # 将所有分割后的字符暂存到一个文件中，做训练用
    for i in range(len(chara_sets)):
        path = 'expression' + picture_name + os.sep + str(i)
        os.makedirs(path)
        for j in range(len(chara_sets[i])):
            cv.imwrite(f'{path}{os.sep}{picture_name}_{i}_{j}.png', chara_sets[i][j])
        tempres = pre_with_foldername.predict_with_folder(path)
        ans.append(tempres)

    shutil.rmtree(path)

    result = []
    for i in ans:
        r = ''
        r = Analyse(i)
        result.append(r)

    accuracy = round(float(result.count('√')) / len(result), 2)
    # 最终结果
    result_image = Mark(image, locations, result)

    cv.imwrite(images_dir + os.sep + 'result.jpg', result_image)
    return accuracy
