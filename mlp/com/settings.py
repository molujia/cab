IMAGE_HOME = 'static/images'  # 访问需要处理的图片的文件夹
HISTORY_HOME = 'static/history/'
RESULT_IMAGE_NAME = 'result.jpg'   # 批改结果图片名
TARGET_IMAGE_NAME = 'test.jpg'   # 上传的图片名


# flask 配置类
class BasicConfig:
    UPLOAD_FOLDER = IMAGE_HOME
    ALLOWED_EXTENSIONS = set(['.png', '.jpg', 'bmp'])
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    # 设置数据库连接
    SQLALCHEMY_DATABASE_URI = 'mysql://myuser:wwxxtt2285@127.0.0.1:3306/flask_homework'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

