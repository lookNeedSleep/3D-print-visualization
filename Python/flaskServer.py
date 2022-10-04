from asyncio.windows_events import NULL
import datetime
from email import message
import random
from turtle import right
from PIL import Image
import cv2
from flask import Flask, jsonify, make_response, render_template, request, template_rendered
from numpy import size
import numpy
import flask_config
from datetime import timedelta
import matplotlib.pyplot as plt
import os
import edgeGetServer
import base64


app = Flask(__name__)
app.config.from_object(flask_config)
imageUploadPath = "./upload/images/"
imageSaveExetension = ".bmp"
contourSavePath = "./contourImg/"
finalContourPath = './midLine/'
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# 主页
@app.route('/index', methods=['GET', 'POST'])
def GoToIndex():
    sliderValue = 0
    fileList = os.listdir(imageUploadPath)
    imageList = imageListInit('bmp')
    contourImages = list(filter(countours_filter, fileList))
    contourImageData = ""
    if(size(contourImages) == 0):
        contourImageData = NULL
        imgName = NULL
        message = NULL
    else:
        contourImageData = imageList[sliderValue][1]
        imgName = imageList[sliderValue][0]
        message = "OK"

    context = {
        "contourImageData": contourImageData,
        "imgName": imgName,
        "sliderValue": sliderValue,
        "min": 1,
        "max": getMax(),
        "message": message,
    }
    return render_template("serverVersion-2.html", **context)


def getMax():
    '''
    获取图片数量
    '''
    return size(list(filter(countours_filter, os.listdir(imageUploadPath))))-1


def allowed_file(filename):
    '''
    文件后缀名合法性判定
    '''
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 图片转为字节流
def return_img_stream(filePath):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    with open(filePath,  'rb',) as img_f:
        img_stream = base64.b64encode(img_f.read()).decode('ascii')
    return img_stream


class Pic_str:
    '''
    （暂时弃用）
    图片名唯一生成
    '''
    def create_uuid(self):  # 生成唯一的图片的名称字符串，防止图片显示时的重名问题
        nowTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # 生成当前时间
        randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
        if randomNum <= 10:
            randomNum = str(0) + str(randomNum)
        uniqueNum = str(nowTime) + str(randomNum)
        return uniqueNum


def countours_filter(f):
    '''
    轮廓图片件前缀名判定
    '''
    if f[0:3] in ['sil']:
        return True
    else:
        return False


def jpg_filter(f):
    '''
    上传原图片后缀名判定
    '''
    if f[-4:] in ['.jpg'] and (f[0:3] not in ['sil']):
        return True
    else:
        return False


def imageListInit(fileExetension):
    '''
    获取图片列表
    fileExetension：图片类型
    '''
    imageList = []
    fileList = os.listdir(imageUploadPath)
    if fileExetension == 'jpg':
        contourImages = list(filter(jpg_filter, fileList))
    else:
        contourImages = list(filter(countours_filter, fileList))
    for i in range(size(contourImages)):
        imageList.append(
            [contourImages[i], return_img_stream(imageUploadPath + contourImages[i])])
    return imageList

# 滑块数值获取轮廓图片
@app.route('/getContourImage', methods=['GET', 'POST'])
def getContourImage():
    sliderValue = request.form['data']
    fileList = os.listdir(imageUploadPath)
    contourImages = list(filter(countours_filter, fileList))
    imageList = imageListInit('contour')
    if int(sliderValue) > size(contourImages):
        return "error"
    context = {
        "contourImageData": imageList[int(sliderValue)][1],
        "imgName": imageList[int(sliderValue)][0],
        "sliderValue": int(sliderValue),
    }
    return jsonify(context)

# 文件上传
@app.route('/uploadFile', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # 通过file标签获取文件
        f = request.files['file']
        fileName = (f.filename).rsplit('.', 1)[0]
        fileExtension = (f.filename).rsplit('.', 1)[1]
        if not (f and allowed_file(f.filename)):
            return jsonify({"message": "Type Error", "msg": "图片类型：png、PNG、jpg、JPG、bmp"})
        img_path = imageUploadPath + fileName + "." + fileExtension
        f.save(img_path)
        # edgeGetServer.CannyThreshold(fileName, fileExtension, imageUploadPath)
        edgeGetServer.getSilhouette(fileName, fileExtension, imageUploadPath)

        context = {
            "message": "IT IS OK!",
            "msg": "",
            # "sliderValue": int(request.form['sliderValue']),
        }
        return jsonify(context)

# 读取图像列表，并返回字符流显示
@app.route('/getImgList', methods=['GET', 'POST'])
def getImgList():
    if request.method == 'POST':
        # 通过file标签获取文件
        jpgList = []
        jpgList = imageListInit('jpg')
        bmpList = imageListInit('bmp')
        context = {
            "jpgList": jpgList,
            "bmpList": bmpList,
            "message": "IT IS OK!",
        }
        return jsonify(context)

# 返回点击点相对应坐标
@app.route('/coordinationDispose', methods=['GET', 'POST'])
def coordinationDispose():

    x = float(request.form['x'])
    y = float(request.form['y'])
    Height = float(request.form["imgHeight"][:-2])
    Width = float(request.form["imgWidth"][:-2])
    name = request.form['name']
    img = cv2.imread(imageUploadPath+name)
    # print("position img size",img.shape)
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]
    actualWidth = x/Width*imgWidth
    actualHeight = imgHeight - y/Height*imgHeight
    # contourImageRedraw( actualHeight,actualWidth)
    context = {
        "actualHeight": actualHeight,
        "actualWidth": actualWidth,
    }
    return jsonify(context)

# 滑块更新
@app.route('/updateSlider', methods=['GET', 'POST'])
def updateSlider():
    context = {
        "min": 1,
        "max": getMax(),
    }
    return jsonify(context)

    # remove filename
    # os.remove(fileName)
    # url_for(函数，参数)

# 轮廓拟合
@app.route('/drawFixContour', methods=['GET', 'POST'])
def drawFixContour():
    fileName = request.form['fileName']
    leftTopP = [int(request.form.getlist('leftTopP[]')[0]),
                int(request.form.getlist('leftTopP[]')[1])]
    leftBottomP = [int(request.form.getlist('leftBottomP[]')[0]), int(
        request.form.getlist('leftBottomP[]')[1])]
    rightBottomP = [int(request.form.getlist('rightBottomP[]')[0]), int(
        request.form.getlist('rightBottomP[]')[1])]
    fy1, fy2, fy3, message = edgeGetServer.getFinalContour(
        imageUploadPath, fileName, leftTopP, leftBottomP, rightBottomP)
    fileName = fileName[:-4]+".jpg"
    content = {
        "contourImage": return_img_stream('./midLine/final'+fileName+'.jpg'),
        "fy1": str(fy1),
        "fy2": str(fy2),
        "fy3": str(fy3),
        "message": message
    }
    return jsonify(content)

# 计算半径数据
@app.route('/dataExport', methods=['GET', 'POST'])
def dataExport():
    fy1 = strToNdarray(request.form['fy1'])
    fy2 = strToNdarray(request.form['fy2'])
    midLineFactor = strToNdarray(request.form['fy3'])
    yList = numpy.array([600.0,550.0, 500.0,450.0, 400.0])
    rList = []
    for y in yList:
        dis1 = -1.0
        dis2 = -1.0
        dis = [[], []]
        normalL, x = normalLine(midLineFactor, y)
        leftLineF=fy1.copy()
        rightLineF=fy2.copy()
        p = edgeGetServer.getIntersection(leftLineF, normalL)
        dis1 = getDistance([x, y], p)
        p = edgeGetServer.getIntersection(rightLineF, normalL)
        dis2 = getDistance([x, y], p)
        if(dis1 > 0 and dis2 > 0):
            dis[0].append(dis1)
            dis[1].append(dis2)
        rList.append(dis)

    content = {
        "rList": rList,
        "message": "ok",
    }

    return jsonify(content)


def normalLine(factor, y):
    f = numpy.poly1d(factor)
    x = f(y)
    derF = f.deriv(1)
    derX = derF(y)
    ky = -1/derX
    const = x-ky*y
    lineFactor = numpy.array([ky, const])
    return lineFactor, x


def getDistance(p1, p2):
    return (((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5).item()


def strToNdarray(string):
    string = string[1:-1]
    strArray = string.split(' ')
    numA = []
    for i in range(len(strArray)):
        if strArray[i] in ['']:
            continue
        if strArray[i][-2:] in ['\n']:
            strArray[i] = strArray[i][:-3]
        numA.append(float(strArray[i]))
    arry = numpy.array(numA)
    return arry


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
