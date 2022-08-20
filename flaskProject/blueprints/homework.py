from flask import Blueprint,render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2 as cv
import time
import ImageProcess
bp=Blueprint("homework",__name__,url_prefix="/")

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@bp.route("/")
def index():
    return render_template("index.html")

@bp.route("/homework/correct", methods=['POST', 'GET'])
def correct():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        #basepath = os.path.dirname(__file__)  # 当前文件所在路径
        basepath="./static/images"
        upload_path = os.path.join(basepath, secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv.imread(upload_path)
        cv.imwrite(os.path.join(basepath, 'test.jpg'), img)

        return render_template('upload_ok.html', userinput=user_input, val1=time.time())

    #return render_template('upload.html')
    return render_template("correct.html")

@bp.route("/homework/correct/result")
def show():
    ImageProcess.getResult('test.jpg')

    return render_template('result.html')
