from django.apps import AppConfig
from tensorflow import keras

class DetectMalariaConfig(AppConfig):
        name = "DetectMalaria"
        model = keras.models.load_model('modelpath/deneme.h5')
        if (model):
                print("Model yüklendi")