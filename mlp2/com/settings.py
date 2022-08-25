# -*-coding:utf-8-*-

# ----------------   处理图片需要的参数    ---------------
IMAGE_HOME = r'static/images'               # 访问需要处理的图片的文件夹
HISTORY_HOME = r'static/history/'           # 存放历史图片的文件夹
RESULT_IMAGE_NAME = 'result.jpg'            # 结果文件名
TARGET_IMAGE_NAME = 'target.jpg'            # 需要处理的图片名
PRINT_MODEL = r'model/svm_print.m'          # 识别打印字体的 svm模型
HANDWRITE_MODEL = r'model/svm_handwrite.m'  # 识别手写字体的 svm模型


# ----------------     flask 配置类      ----------------
class BasicConfig:
    UPLOAD_FOLDER = IMAGE_HOME                                               # 默认的静态图片路径
    ALLOWED_EXTENSIONS = set(['.png', '.jpg', 'bmp'])                        # 允许上传的文件类型
    SQLALCHEMY_DATABASE_URI = 'mysql://myuser:wwxxtt2285@127.0.0.1:3306/flask_homework'  # 设置数据库连接
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

