from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model

# Create your views here.

from sympy.printing.tests.test_tensorflow import tf
from keras.preprocessing import image
import numpy as np
from tensorflow import Graph
from tensorflow.python.client.session import Session
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 180, 180
batch_size = 32

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('modelpath/ICModel.h5')


def index(request):
    return render(request, 'index.html')


def imageProcess(filePath, x):
    val_data = filePath
    val_ds = x.flow_from_directory(
        val_data,
        color_mode='grayscale',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    return val_ds


def predictMalaria(request):
    print(request)
    print(request.POST.dict())
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x / 255
    x = x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            predictions = model.predict(x)
            if predictions[0][0] > predictions[0][1]:
                pos = "Not Infected"
            else:
                pos = "Infected"
            result = 100 * np.max(predictions)
            result = round(result, 2)
    # x = x/255
    # x = x.reshape(1, img_height, img_width, 3)
    # with model_graph.as_default():
    #   with tf_session.as_default():
    #      predi = model.predict(x)

    context = {'filePathName': filePathName,
               'pos': pos,
               'percentage': result}
    return render(request, 'index.html', context)
