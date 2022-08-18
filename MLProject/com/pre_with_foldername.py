def load_file_dir(folder_dir):
    import os
    file_dir = []
    for i in os.listdir(folder_dir):
        i_path = os.path.join(folder_dir, i)
        file_dir.append(i_path)
    return file_dir

def svm_predict(image_path):
	import joblib
	import ml_predict_utility
	# import ml_predict_utility
	# 打印体字符识别，包含符号
	# 需要输入20×20的数字图像，输出对应的字典
	# 模型地址填写模型所在位置
	LABEL_DICT = [
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
		'=', '+', '-', '*', '÷']
	MODEL_PATH = r"E:\Python\Workspace\PycharmProjects\MLProject\com\svm_new.m"#更改这里！
	IMAGE_WIDTH = 20
	IMAGE_HEIGHT = 20
	digit_image = ml_predict_utility.load_image(image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
	model = joblib.load(MODEL_PATH)
	predicts = model.predict(digit_image)
	return(LABEL_DICT[predicts[0]])

def knn_predict(image_path):
	import joblib
	import ml_predict_utility
	import cv2
	# 手写体字符识别，不包含符号
	# 需要输入28×28的数字图像，输出对应的字典
	# 模型地址填写模型所在位置
	LABEL_DICT = [
		'0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	MODEL_PATH = r"E:\Python\Workspace\PycharmProjects\MLProject\com\GBDT.m"#更改这里！
	IMAGE_WIDTH = 28
	IMAGE_HEIGHT = 28
	digit_image = ml_predict_utility.load_image(image_path, IMAGE_WIDTH, IMAGE_HEIGHT)
	model = joblib.load(MODEL_PATH)
	predicts = model.predict(digit_image)
	return(LABEL_DICT[predicts[0]])

def predict_with_folder(folder_dir):
    #输入文件夹的路径，文件夹内理应是顺序排列的图片
    answer=[]#最终的字符列表
    file_dir=load_file_dir(folder_dir)
    i=0
    while(i<len(file_dir)):
        temp=svm_predict(file_dir[i])
        answer.append(temp)
        i+=1
        if(temp=='='):break

    while(i<len(file_dir)):
        temp=knn_predict(file_dir[i])
        answer.append(temp)
        i+=1

    return answer

# ans=predict_with_folder(r"E:\Python\Workspace\PycharmProjects\MLProject\com\charss")
# print(ans)
# 经测试无误，需要注意的是没有=号也能运行，就是不鲁棒