from server import app, utilities as fn
from flask import request
import os
import cv2
from PIL import Image
import json




@app.route('/uploadImage',  methods=['POST'])
def uploadImage():

    if request.method == 'POST':

        file1 = request.files["Image_File1"]
        file1.save(os.path.join(
            'server/static/assests/Image1.jpg'))
        img= cv2.imread('server//static//assests//image1.jpg')
        fn.rgbHistogram(img)
        fn.histogram(img)
        fn.cumm_dist(img)
        img = fn.rgbtogray(img)

       
        
        norm = fn.normalize(img)
        eq = fn.equalization('server//static//assests//image1.jpg')
        glbl = fn.globalThresholding(img,200)
        lcl = fn.localThresholding(img,40)

        cv2.imwrite('server//static//assests//normalize.jpg', norm)
        cv2.imwrite('server//static//assests//equalize.jpg', eq)
        cv2.imwrite('server//static//assests//global.jpg', glbl)
        cv2.imwrite('server//static//assests//local.jpg', lcl)


       

    return []


@app.route('/imgProcessing' ,  methods=['POST'])
def imgProcessing():
    if request.method == 'POST':
        jsonData = request.get_json()  
        data = jsonData['formElement']

        if (data[0]== False):
            return []
                
        else:
            img= cv2.imread('server//static//assests//image1.jpg')
            img = fn.rgbtogray(img)
            if (data[1]=='None'):
                output = img
                cv2.imwrite('server//static//assests//output.jpg', output)
            else:
                if(data[1]=='Uniform'):
                    output = fn.uniform_noise(img)
                elif(data[1]=='Gaussian'):
                    output = fn.gaussian_noise(img)
                elif(data[1]=='salt'):
                    output = fn.salt_and_pepper(img)
                cv2.imwrite('server//static//assests//output.jpg', output)

            img= cv2.imread('server//static//assests//output.jpg')
            img = fn.rgbtogray(img)
            
            if (data[2]=='None'):
                output = img
                cv2.imwrite('server//static//assests//output.jpg', output)
            else:
                if(data[2]=='Average'):
                    output = fn.meanLowPass(img)
                elif(data[2]=='Gaussian'):
                    output = fn.GaussianLowFilter(img)
                elif(data[2]=='Median'):
                    output = fn.median_filter(img)
                cv2.imwrite('server//static//assests//output.jpg', output)


            img= cv2.imread('server//static//assests//output.jpg')
            img = fn.rgbtogray(img)

            if (data[3]=='None'):
                output = img
                cv2.imwrite('server//static//assests//output.jpg', output)
            else:
                if(data[3]=='Sobel'):
                    output = fn.sobel(img)
                elif(data[3]=='Robert'):
                    output = fn.robert(img)
                elif(data[3]=='Prewitt'):
                    output = fn.prewit(img)
                else:
                    output=fn.canny(img)
                cv2.imwrite('server//static//assests//output.jpg', output)  

            img= cv2.imread('server//static//assests//output.jpg')
            img = fn.rgbtogray(img)                  
            if (data[4]=='None'):
                output = img
                cv2.imwrite('server//static//assests//output.jpg', output)
            else:
                if(data[4]=="high"):
                    output = fn.IdealHighPass(img)
                elif(data[4]=="low"):
                    output = fn.idealLowPass(img)
                cv2.imwrite('server//static//assests//output.jpg', output)
    

            return data    








@app.route('/hybrid' ,  methods=['POST'])                 
def hybrid():

    if request.method == 'POST':
        jsonData = request.get_json()  
        data = jsonData['formElement']

        if (data[0]== True and data[1]== True):
                img1 = cv2.imread('server/static/assests/hybridimg1.jpg')
                img2 = cv2.imread('server/static/assests/hybridimg2.jpg')

                images=[img1 , img2]
                final=fn.hybrid_image(images, [7,7], 1)
                cv2.imwrite('server//static//assests//hybridoutput.jpg', final*255)  
                return data
                            
        else: 
            return  data




@app.route('/uploadHybrid',  methods=['POST'])
def uploadHybrid():

    if request.method == 'POST':
                
        try:
            file2 = request.files["hybrid_img1"]
            file2.save(os.path.join(
                'server/static/assests/hybridimg1.jpg'))
        except:
            file3 = request.files["hybrid_img2"]
            file3.save(os.path.join(
                'server/static/assests/hybridimg2.jpg')) 

    return []