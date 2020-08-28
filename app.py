import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage import filters, morphology, color, exposure, img_as_float
from skimage import io
from skimage.morphology import dilation, opening, erosion
from skimage.morphology import disk
from skimage.filters.rank import median, mean
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from flask import Flask, jsonify
from flask import send_file
from flask import request
from flask_restful import Resource, Api
# from flask_socketio import SocketIO, Namespace, emit
from PIL import Image
from guppy import hpy
import os, psutil, gc, errno
import tracemalloc
import time
import base64
import calendar
from pathlib import Path
matplotlib.use('Agg')

application = app = Flask(__name__)
api = Api(application)


class RequestReceiver(Resource):
    # def get(self):
    #     obj = Glass()
    #     opened = obj.adjust_img()
    #     return obj.get_labels(opened)

    def post(self):
        requested_method = request.form['method']
        print(requested_method)
        print("Recieved request")
        if requested_method == "make_labels":
            image_file = Image.open(request.files['image'])
            obj = Glass(image_file)
            opened = obj.adjust_img()
            return obj.get_labels(opened)
        elif requested_method == "label_data":
            file_name = request.form['name']
            label_number = request.form['label_number']
            label_obj = LabelData(file_name)
            return label_obj.get_label_data(int(label_number))
        elif requested_method == "area_data":
            file_name = request.form['name']
            label_number = request.form['label_number']
            sort_type = request.form['sort_type']
            obj = AreaData(file_name)
            if sort_type == 'large':
                return obj.get_label_data(10 + int(label_number))
            else:
                return obj.get_label_data(int(label_number))
        elif requested_method == "delete":
            file_name = request.form['name']
            obj = DeleteClutter(file_name)
            return obj.delete_file()


class Glass:
    def __init__(self, received_image):
        self.start_time = time.time()
        self.h = hpy()
        self.d1 = self.make_disk(1)
        self.d2 = self.make_disk(10)
        self.d3 = self.make_disk(5)
        self.d4 = self.d3
        self.d5 = self.make_disk(2)
        self.bimg = received_image
        self.image = img_as_float(self.bimg)
        self.file_name = str(calendar.timegm(time.gmtime()))
        self.current_path = str(Path().absolute())
        self.indices_path = self.current_path + "/indices_files/"

    def make_disk(self, n):
        return disk(n)

    def adjust_img(self):
        start_time = time.time()
        gamma_corrected = exposure.adjust_gamma(self.image, 1)
        gamma_corrected = exposure.equalize_hist(gamma_corrected)
        gamma_corrected = exposure.equalize_adapthist(gamma_corrected)
        gamma_corrected = exposure.rescale_intensity(gamma_corrected)

        gamma_corrected = color.rgb2gray(gamma_corrected)
        imgg1 = mean(gamma_corrected, self.d4)
        imgg1 = median(imgg1, self.d3)
        dilated = dilation(imgg1, self.d5)

        eroded = erosion(dilated, self.d1)
        thresh = filters.threshold_adaptive(eroded, 253, method='gaussian')
        del eroded
        BW = gamma_corrected >= thresh
        BW_smt = remove_small_objects(BW, 500)
        opened = opening(BW_smt, self.d2)
        del BW, BW_smt, thresh, gamma_corrected, imgg1
        gc.collect()
        return opened
        # size = opened.shape[::-1]
        # databytes = np.packbits(opened, axis=1)
        # opened_img = Image.frombytes(mode='1', size=size, data=databytes)
        # opened_img.save('opened_img.jpg')
        # print(self.h.heap())
        # return send_file('opened_img.jpg', mimetype='image/gif')

    def get_labels(self, img):
        labels = morphology.label(~img, background=-1)
        labels = morphology.remove_small_objects(labels, 500)
        plt.imshow(labels, cmap='jet', interpolation='nearest')
        a = np.unique(labels)
        ans = a.shape

        props = regionprops(labels)
        END = []
        label_list = list()

        for i in range(len(a) - 1):
            temp = list()
            temp.append(i)
            temp.append(int(props[i].area))
            label_list.append(temp)
            if (1 in props[i].coords.T[0]) or (labels.shape[0] - 1 in props[i].coords.T[0]) or (
                    1 in props[i].coords.T[1]) or (labels.shape[1] - 1 in props[i].coords.T[1]):
                END.append(i)
            else:
                continue

        label_overlay = label2rgb(labels, image=self.image)
        label_list.sort(key=lambda x: x[1])
        large_small_list = list()
        num_labels = len(label_list)
        for i in range(0, 10):
            large_small_list.append(label_list[i][0])
        for i in range(0, 10):
            large_small_list.append(label_list[num_labels - 11 + i][0])

        with open(self.indices_path + self.file_name + '_indices', "w") as f:
            for s in large_small_list:
                f.write(str(s) + "\n")

        label_path = self.current_path + '/label_files/'
        np.save(label_path + self.file_name + '_label_arr', labels)
        del labels, label_list, large_small_list
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(label_overlay)

        for i in range(len(props)):
            if props[i].area < 100:
                continue
            if i in END:
                minr, minc, maxr, maxc = props[i].bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='white', linewidth=1)
            else:
                minr, minc, maxr, maxc = props[i].bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=1)

            ax.add_patch(rect)

        del props, label_overlay,
        file_path = self.current_path + '/img_files/'
        plt.savefig(file_path + self.file_name + '_final.jpg', box_inches='tight')
        del fig, ax
        return_data = {}
        with open(file_path + self.file_name + '_final.jpg', mode='rb') as file:
            img_file = file.read()
        plt.close()
        return_data['img'] = base64.encodebytes(img_file).decode("utf-8")
        return_data['labels'] = len(a)
        return_data['end_pieces'] = len(END)
        return_data['file_name'] = self.file_name
        print(time.time() - self.start_time)
        print(self.h.heap())
        # return send_file('final.jpg', mimetype='image/gif')
        return jsonify(return_data)


class LabelData:
    def __init__(self, file_name):
        self.current_path = str(Path().absolute())
        self.img_path = self.current_path + '/img_files/'
        self.label_path = self.current_path + '/label_files/'
        self.area_path = self.current_path + '/area_files/'
        self.file_name = file_name
        self.label_arr = np.load(self.label_path + file_name + '_label_arr.npy')

    def get_label_data(self, label_number):
        if label_number > len(np.unique(self.label_arr)):
            return "Out of range"
        props = regionprops(self.label_arr)
        plt.imshow(self.label_arr == np.unique(self.label_arr)[label_number + 1])
        area = int(props[label_number].area)
        perimeter = int(props[label_number].perimeter)
        plt.savefig(self.area_path + self.file_name + '_test.jpg', box_inches='tight')
        with open(self.area_path + self.file_name + '_test.jpg', mode='rb') as file:
            img_file = file.read()
        return_data = {}
        return_data['img'] = base64.encodebytes(img_file).decode("utf-8")
        return_data['area'] = area
        return_data['perimeter'] = perimeter
        return jsonify(return_data)


class AreaData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.current_path = str(Path().absolute())
        self.indices_path = self.current_path + '/indices_files/'
        self.area_path = self.current_path + '/area_files/'
        self.label_path = self.current_path + '/label_files/'
        self.indices_list = list()
        self.label_arr = np.load(self.label_path + self.file_name + '_label_arr.npy')
        with open(self.indices_path + file_name + '_indices', "r") as f:
            for line in f:
                self.indices_list.append(int(line.strip()))

    def get_label_data(self, index_number):
        print(self.indices_list)
        print(index_number)
        label_number = self.indices_list[index_number]
        print(label_number)
        if label_number > len(np.unique(self.label_arr)):
            return "Out of range"
        props = regionprops(self.label_arr)
        plt.imshow(self.label_arr == np.unique(self.label_arr)[label_number + 1])
        area = int(props[label_number].area)
        perimeter = int(props[label_number].perimeter)
        plt.savefig(self.area_path + self.file_name + '_test.jpg', box_inches='tight')
        with open(self.area_path + self.file_name + '_test.jpg', mode='rb') as file:
            img_file = file.read()
        return_data = {}
        return_data['img'] = base64.encodebytes(img_file).decode("utf-8")
        return_data['area'] = area
        return_data['perimeter'] = perimeter
        return jsonify(return_data)


class DeleteClutter:
    def __init__(self, filename):
        self.current_path = str(Path().absolute())
        self.label_path = self.current_path + '/label_files/' + filename + '_label_arr.npy'

    def delete_file(self):
        try:
            os.remove(self.label_path)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        return "Success"


api.add_resource(RequestReceiver, '/')

application.run(host='0.0.0.0', port=8080, debug=True)
