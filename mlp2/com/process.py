# -*-coding:utf-8-*-
"""
>该模块于来批改简单的手写作业（加减乘除）
>要处理图片，你需要调用此模块中  ImageProcess类中的 get_result()函数
>详情请看 ImageProcess类的 docstring
"""
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import com.settings as st
import numpy as np
import cv2 as cv
import joblib
import time
import os


class ImageProcess:
    """
    此类用于处理手写作业图片
    处理方式：
        读取默认路径下的目标图片，经过处理切分成单个字符图片，然后预测比较结果，
        最后在原图的基础上作出对错和补全的标识，同样将结果保存在默认路径下
    """

    def __init__(self):
        """
        读取目标图片，作为类属性
        """
        # --------------加载模型-----------
        self.__print_model = joblib.load(st.PRINT_MODEL)
        self.__hand_model = joblib.load(st.HANDWRITE_MODEL)
        # --------------加载图片-----------
        self.__image = cv.imread(st.IMAGE_HOME + os.sep + st.TARGET_IMAGE_NAME)
        # --------------设置参数-----------
        self.__ROW_RATE = 0.18  # 影响切割整行表达式组的效果
        self.__COL_RATE = 0.2  # 影响切割单个表达式时的效果
        self.__ITERATION = 8  # 在切割单个表达式时，图片膨胀时的迭代数，影响膨胀效果
        self.__ROW_INTERNAL = 10  # 调整最后切除表达式的上下偏移
        self.__COL_INTERNAL = 5  # 调整最后切除表达式的左右偏移
        self.__CHAR_RATE = 0.05  # 调整切割单个字符时的效果
        self.__CHAR_INTERNAL = 2  # 调整单个字符的左右偏移
        self.__LABEL_DICT = [  # 预测字符时的编码，相当于数字到字符的字典
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '=', '+', '-', '*', '/']

    def __preprocess(self):
        """
        对用户图整体进行预处理，得到二值化图像
        :return: 处理后的图像
        """
        # 转灰度图像并二值化（大津法）
        gray_image = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)[1]

        # 题卡主要是白底黑字，转换为黑底白字，减少计算量
        processed_image = cv.bitwise_not(binary_image)

        return processed_image

    @staticmethod
    def __rotate_by_text(image):
        """
        根据图片中文本的最小外接矩形进行倾斜矫正，但只允许 5度以内的微调，防止过度矫正
        :param image:需要旋转的图片(已 经过预处理，ndarray)
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

        if abs(angle) < 5:
            # 仿射变换，旋转图片
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv.getRotationMatrix2D(center, angle, 1.0)
            upright_image = cv.warpAffine(image, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
            return upright_image
        else:
            return image

    @staticmethod
    def __split_by_projection(vector, threshold, INTERNAL):
        """
        投影算法分割 vector，用于切割表达式和字符
        :param vector: 图片投影后的行向量或列向量
        :param threshold: 切割依据的阈值
        :param INTERNAL: 偏移量
        :return: begin 和 end 数组，存储切割的起始位置和结束位置
        """
        flag = False            # 判断是否开始切割
        begin, end = [], []     # 存储位置信息
        for i in range(len(vector)):
            if vector[i] > threshold and not flag:
                flag = True
                begin.append(i - INTERNAL)
            if vector[i] < threshold and flag:
                flag = False
                end.append(i + INTERNAL)
        return begin, end

    def __extract_expression(self, image):
        """
        提取图片里的每个等式，利用投影算法切割表达式
        :param image: 处理后的单个图片 (ndarray)
        :return: expressions -> (location, split_image)
            location -> (x, y, width, height) 是该表达式坐标 （ x、y 是矩形左上角坐标）
            split_image是提取出的表达式图片（已 经过了预处理，ndarray）
        """
        locations = []
        # 水平方向得到的投影，列向量
        horizontal = np.sum(image, axis=1)
        threshold_h = np.max(horizontal) * self.__ROW_RATE
        # 将图片切割成单行
        y_begin, y_end = self.__split_by_projection(horizontal, threshold_h, self.__ROW_INTERNAL)
        # 初始化膨胀核，将单个表达式连成一片
        kernel = np.ones((5, 5), dtype=np.uint8)

        # 进行列的遍历，得到单个表达式的位置信息
        for i in range(len(y_begin)):
            # 将图片切成单行
            line_image = cv.dilate(image[y_begin[i]:y_end[i], :], kernel,
                                   iterations=self.__ITERATION)

            vertical = np.sum(line_image, axis=0)  # 按列投影
            threshold_v = np.max(vertical, axis=0) * self.__COL_RATE
            # 将单行切割成单个表达式
            x_begin, x_end = self.__split_by_projection(vertical, threshold_v, self.__ROW_INTERNAL)
            # 提取位置信息
            for k in range(len(x_begin)):
                locations.append((x_begin[k], y_begin[i], x_end[k] - x_begin[k], y_end[i] - y_begin[i]))
        # 得到表达式子图
        expressions = [(loc, image[loc[1]:loc[1] + loc[3], loc[0]:loc[0] + loc[2]]) for loc in locations]
        return expressions

    def __chara_split(self, exp_image):
        """
        将表达式分割为字符集，同样利用投影算法
        :param exp_image: 单个表达式图片
        :return: chara_images -> 分割后的字符集图片
        """
        # 提取字符竖向边界信息
        line = np.sum(exp_image, axis=0)
        threshold = np.max(line, axis=0) * self.__CHAR_RATE
        # 切割单个字符
        begin, end = self.__split_by_projection(line, threshold, self.__CHAR_INTERNAL)
        # 得到单个字符集
        chara_images = [exp_image[:, x1:x2] for x1, x2 in zip(begin, end)]
        return chara_images

    @staticmethod
    def __add_text(img, text, left, top, textColor=(255, 0, 0), textSize=20):
        """
        在图片上的指定位置添加文本
        :param img: 欲添加文本的图片
        :param text: 需要添加的文本
        :param left: 添加位置的横坐标
        :param top: 添加位置的纵坐标
        :param textColor: 文本的颜色
        :param textSize: 文本大小
        :return: 添加文本后的图片
        """
        # 转化图像数据格式
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        fontStyle = ImageFont.truetype("static/fonts/msyhl.ttc", textSize, encoding="utf-8")
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

    def __mark(self, location, answer):
        """
        框出图片中的表达式并添加结果
        :param image: 原始图片
        :param location: 表达式的外框信息
        :param answer: 题目结果，分为三种 1.“√” 2.“×” 3. “answer”
        :return: result_image -> 处理后的图片，提交给前端
        """
        result_image = self.__image.copy()
        for i in range(len(location)):
            cv.rectangle(result_image, (location[i][0], location[i][1]),
                         (location[i][0] + location[i][2], location[i][1] + location[i][3]), (0, 255, 255))

        for i in range(len(location)):
            textSize = location[i][3]
            if answer[i] == "√":
                color = (0, 255, 0)  # 正确为绿色
            elif answer[i] == "×":
                color = (255, 0, 0)  # 错误为红色
            else:
                color = (0, 0, 255)  # 没写为蓝色
            left = location[i][0] + location[i][2]
            top = location[i][1]
            # 将结果写到原图上
            result_image = self.__add_text(result_image, answer[i], left, top, color, textSize)

        return result_image

    @staticmethod
    def __analyse(equation):
        """
        判断表达式的正误
        :param equation: 由图片转化的单个表达式字符串
        :return: "√"、 "×"、"right_answer"
        """
        if '=' in equation:
            part = equation.split('=')
            e_str = part[0]
            r_str = part[1]

            try:
                result = eval(e_str)
            except:  # 预测表达式时出错
                return 'error'

            if r_str == '':
                return str(result)
            else:
                if str(result) == r_str:
                    return '√'
                else:
                    return '×'
        else:
            return 'error'

    def __predict_chara(self, chara_image, model):
        """
        预测字体的内容
        :param chara_image: 字符图像
        :param model: 使用的模型
        :return: 预测值，字符形式
        """
        resized = cv.resize(chara_image, (20, 20))  # 统一为 20×20 大小
        normalized_image = (resized - resized.mean()) / resized.max()  # 标准化
        ravel_image = np.array([normalized_image.ravel()])  # 展平
        predicts = model.predict(ravel_image)  # 预测
        return self.__LABEL_DICT[predicts[0]]

    def __predict_chars(self, chars_image):
        """
        利用预测函数，将字符图像数组转化为字符串
        :param chars_image: 字符图像数组
        :return: 字符串
        """
        string = ''
        flag = 0
        while flag < len(chars_image):
            temp = self.__predict_chara(chars_image[flag], self.__print_model)
            string = string + temp
            flag += 1
            if temp == '=':  # 检测到等号后转换模型
                break
        while flag < len(chars_image):
            temp = self.__predict_chara(chars_image[flag], self.__hand_model)
            string = string + temp
            flag += 1
        # 返回结果
        return string

    def getResult(self):
        """
        对原始图片进行处理，得到结果图片，并存储到默认路径下
        :return: 所有题目的正确率，将显示到前端
        """
        processed_image = self.__preprocess()  # 图片二值化
        rota_image = self.__rotate_by_text(processed_image)  # 对图片整体进行旋转
        expressions = self.__extract_expression(rota_image)  # 提取所有表达式
        locations = [i[0] for i in expressions]  # 获取位置信息
        split_image = [i[1] for i in expressions]  # 获得分割得到的表达式集
        chara_sets = [self.__chara_split(exp) for exp in split_image]  # 分割为单个字符
        result = [self.__analyse(self.__predict_chars(i)) for i in chara_sets]  # 预测表达式
        accuracy = round(float(result.count('√')) / len(result), 2)  # 计算正确率
        result_image = self.__mark(locations, result)  # 结果图片
        cv.imwrite(st.IMAGE_HOME + os.sep + st.RESULT_IMAGE_NAME, result_image)  # 保存
        return accuracy


if __name__ == "__main__":
    # 检查算法是否能正常工作，需要提前在指定目录下放置目标图片
    start = time.time()        # 测试图片批改时间
    accuracy = ImageProcess().getResult()
    end = time.time()
    print("accuracy: " + str(accuracy))
    print("Running time: " + str(end - start))
