
from django.apps import AppConfig
from tensorflow import keras

class AppConfig(AppConfig):
    name = 'app'
    #model = keras.models.load_model('C:/Users/user/Desktop/EnginProject/malaria/models/')
    #if (model):
    #    print("model çalışıyor")