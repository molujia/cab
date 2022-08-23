import time
from flask import Flask
from flask import render_template, request
from flask_sqlalchemy import SQLAlchemy
import ImageProcess
from PIL import Image
import datetime
import os
import settings
import pymysql

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# 导入基本配置类
app.config.from_object(settings.BasicConfig)

db = SQLAlchemy(app)


class History(db.Model):
    # 创建用户历史记录表
    __tablename__ = 'users_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(64), index=True)
    pwd = db.Column(db.String(64))
    # 存储历史结果的 url
    history = db.Column(db.String(255))
    # 以字符串形式存储时间，精确到秒
    date = db.Column(db.String(30))
    # 存储正确率
    accuracy = db.Column(db.Float)


@app.route('/')
def introduction():
    return render_template('intro.html')


@app.route('/<username>')
def user_introduction(username):
    return render_template('intro.html', is_success='pass', username=username)


@app.route('/intro_login', methods=['GET', 'POST'])
def intro_login():
    if request.method == 'POST':
        user_info = request.form.to_dict()
        name = user_info.get('username')
        query_out = History.query.filter_by(name=name, pwd=user_info.get('password')).all()
        if len(query_out) is not 0:
            return render_template('intro.html', is_success='pass', username=name)
        else:
            return render_template('intro.html', is_success='refuse')

    return render_template('intro.html')


# 非登录状态响应
@app.route('/apply', methods=['GET', 'POST'])
def product():

    if request.method == 'POST':

        uploadFile = request.files['img']
        fileName = uploadFile.filename
        extName = os.path.splitext(fileName)[1]

        if extName in app.config['ALLOWED_EXTENSIONS']:
            uploadFile.save(settings.IMAGE_HOME + os.sep + settings.TARGET_IMAGE_NAME)
            accuracy = ImageProcess.getResult(settings.TARGET_IMAGE_NAME)
            return render_template('uploaded.html', msg='正确率:' + str(accuracy))
        else:
            return render_template('application.html', msg='文件类型错误, 希望接受到(.bmp |.png |.jpg)文件')
    else:
        return render_template('application.html')


# 登录状态响应，将记录添加到数据库
@app.route('/apply/<username>', methods=['GET', 'POST'])
def user_product(username):

    if request.method == 'POST':

        uploadFile = request.files['img']
        fileName = uploadFile.filename
        extName = os.path.splitext(fileName)[1]
        if extName in app.config['ALLOWED_EXTENSIONS']:
            # 暂存到图片目录
            uploadFile.save(settings.IMAGE_HOME + os.sep + settings.TARGET_IMAGE_NAME)
            # 处理图片，响应客户端
            accuracy = ImageProcess.getResult(settings.TARGET_IMAGE_NAME)
            # 保存图片至数据库，文件名由时间和用户名组成
            query = History.query.filter_by(name=username).first()
            pwd = query.pwd
            now_time = time.localtime(time.time())
            now = time.strftime("%Y-%m-%d_%H_%M_%S", now_time)
            history_path = settings.HISTORY_HOME + now + f'_{username}.png'
            image = Image.open(settings.IMAGE_HOME + os.sep + settings.RESULT_IMAGE_NAME)
            image.save(history_path)
            recording = History(name=username, pwd=pwd, history=history_path, date=now, accuracy=accuracy)
            db.session.add(recording)
            db.session.commit()

            return render_template('uploaded.html', msg='正确率:' + str(accuracy), username=username)
        else:
            # 响应客户端
            return render_template('application.html', msg='文件类型错误, 希望接受到(.bmp |.png |.jpg)文件', username=username)
    else:
        return render_template('application.html', username=username)


@app.route('/history/<username>')
def history(username):
    # 得到所有历史记录
    search_result = History.query.filter_by(name=username).all()
    # 存储文件路径和时间
    result = []
    for i in search_result:
        if i.history is not None:
            # 转化为时间类
            date_time = datetime.datetime.strptime(i.date, "%Y-%m-%d_%H_%M_%S")
            result.append((i.history, date_time, i.accuracy))
    # 结果按时间排序
    result.sort(key=lambda x: x[1])
    if len(result) is not 0:
        return render_template('history.html', result=result, username=username)
    else:
        return render_template('history.html', result='没有检测到历史记录！', username=username)


if __name__ == '__main__':
    # 重新生成表
    db.drop_all()
    db.create_all()

    # 暂时不考虑注册功能，生成一位虚拟用户
    sign_one = History(name='张三', pwd='123')
    db.session.add(sign_one)
    db.session.commit()
    app.run()
