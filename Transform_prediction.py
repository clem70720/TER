import matplotlib.pyplot as plt
import numpy as np

from os.path import splitext
from keras.saving import load_model
from skimage.io import imread
from skimage.transform import resize
from joblib import load

def transform_image(path_image):
    image = imread(path_image)
    image = np.array(image).astype("float32")
    image= resize(image,(28,28,1),preserve_range=True)
    image = 255 - image
    image = image / np.max(image)
    return image

def prediction(model_path,image_to_predict):
    if splitext(model_path)[1] == ".sav":
        load_model_var = load(model_path)
        image_to_predict = transform_image(image_to_predict)
        image_to_predict = image_to_predict.reshape(1,784)
        model_prediction = load_model_var.predict(image_to_predict)
        print(model_prediction[0])
        #model_prediction_proba = load_model_var.predict_log_proba(image_to_predict)
        #print(model_prediction_proba[0])
        return model_prediction[0]
    elif splitext(model_path)[1] == ".keras":
        load_model_var = load_model(model_path)
        image_to_predict = transform_image(image_to_predict)
        image_to_predict = image_to_predict.reshape(1,28,28,1)
        model_prediction_proba = load_model_var.predict(image_to_predict)
        category_predicted = np.where(model_prediction[0] == max(model_prediction[0]))[0][0]
        #print(max(model_prediction[0]))
        #print(np.where(model_prediction[0] == max(model_prediction[0]))[0][0])
        return (category_predicted,model_prediction_proba[0])
    else:
        print(splitext(model_path)[1])
        return print("Votre fichier n'est pas au format .sav ou .keras")



"""image_to_show = transform_image("s oliver pullover-181jnn.jpg")
print(image_to_show.shape)
plt.imshow(image_to_show,cmap='gray')
plt.show()"""

prediction("SVC.sav","T-shirt_grey.jpeg")




