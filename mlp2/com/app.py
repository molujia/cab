# -*- coding:utf-8 -*-
from flask import render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from PIL import Image
import settings as st
import pymysql
import process
import datetime
import time
import os

app = Flask(__name__)
pymysql.install_as_MySQLdb()  # 链接 mysql 数据库
app.config.from_object(st.BasicConfig)  # 导入基本配置类
db = SQLAlchemy(app)


class History(db.Model):
    """
    创建用户历史记录表，有六个字段，存储用户的注册信息和提交记录
    """
    __tablename__ = 'users_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)  # 生成主码 id
    name = db.Column(db.String(64), index=True)  # 存储用户名
    pwd = db.Column(db.String(64))  # 存储用户密码
    history = db.Column(db.String(255))  # 存储历史结果的 url
    date = db.Column(db.String(30))  # 以字符串形式存储时间，精确到秒
    accuracy = db.Column(db.Float)  # 存储正确率


@app.route('/')
def home():
    """
    进入客户端主页面
    """
    return render_template('home_page.html')


@app.route('/<username>')
def user_home(username):
    """
    进入用户登录状态下的客户端主页面
    """
    return render_template('home_page.html', is_success='pass', username=username)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    处理登录表单
    """
    if request.method == 'POST':
        # 验证用户信息
        user_info = request.form.to_dict()
        name = user_info.get('username')
        query_out = History.query.filter_by(name=name, pwd=user_info.get('password')).all()
        if len(query_out) is not 0:
            return render_template('home_page.html', is_success='pass', username=name)
        else:
            return render_template('home_page.html', is_success='refuse')

    return render_template('home_page.html')


@app.route('/apply', methods=['GET', 'POST'])
def product():
    """
    接收图片处理请求，此时为未登录状态
    """
    if request.method == 'POST':
        # 接收上传的图片并验证后缀是否符合规范
        uploadFile = request.files['img']
        fileName = uploadFile.filename
        extName = os.path.splitext(fileName)[1]

        if extName in app.config['ALLOWED_EXTENSIONS']:
            uploadFile.save(st.IMAGE_HOME + os.sep + st.TARGET_IMAGE_NAME)
            accuracy = process.ImageProcess().getResult()       # 进行图片处理
            return render_template('result_page.html', msg='正确率:' + str(accuracy))
        else:
            return render_template('apply_page.html', msg='文件类型错误, 希望接受到(.bmp |.png |.jpg)文件')

    else:
        return render_template('apply_page.html')


@app.route('/apply/<username>', methods=['GET', 'POST'])
def user_product(username):
    """
    登录状态下的上传图片处理
    """
    if request.method == 'POST':
        print("enter")

        uploadFile = request.files['img']
        fileName = uploadFile.filename
        extName = os.path.splitext(fileName)[1]

        if extName in app.config['ALLOWED_EXTENSIONS']:

            uploadFile.save(st.IMAGE_HOME + os.sep + st.TARGET_IMAGE_NAME)
            accuracy = process.ImageProcess().getResult()

            # 保存图片至数据库，文件名由时间和用户名组成
            query = History.query.filter_by(name=username).first()
            pwd = query.pwd
            now_time = time.localtime(time.time())
            now = time.strftime("%Y-%m-%d_%H_%M_%S", now_time)
            history_path = st.HISTORY_HOME + now + f'_{username}.png'
            # 将图片保存到历史文件夹下面
            Image.open(st.IMAGE_HOME + os.sep + st.RESULT_IMAGE_NAME).save(history_path)
            recording = History(name=username, pwd=pwd, history=history_path, date=now, accuracy=accuracy)
            # 提交数据
            db.session.add(recording)
            db.session.commit()
            return render_template('result_page.html', msg='正确率:' + str(accuracy), username=username)
        else:
            return render_template('apply_page.html', msg='文件类型错误, 希望接受到(.bmp |.png |.jpg)文件', username=username)

    else:
        return render_template('apply_page.html', username=username)


@app.route('/history/<username>')
def history(username):
    """
    响应历史记录页面，将在数据库中查询到的数据发送到客户端
    """
    # 得到所有历史记录
    search_result = History.query.filter_by(name=username).all()
    result = []
    for i in search_result:
        if i.history is not None:
            # 转化为时间类
            date_time = datetime.datetime.strptime(i.date, "%Y-%m-%d_%H_%M_%S")
            result.append((i.history, date_time, i.accuracy))
    # 结果按时间排序
    result.sort(key=lambda x: x[1])
    if len(result) is not 0:
        return render_template('history_page.html', result=result, username=username)
    else:
        return render_template('history_page.html', result='没有检测到历史记录！', username=username)


if __name__ == '__main__':
    # 每次用户进入都会重新生成表
    db.drop_all()
    db.create_all()

    # 因为暂时不考虑注册功能，生成一位虚拟用户
    sign_one = History(name='张三', pwd='123')
    db.session.add(sign_one)
    db.session.commit()
    app.run()
