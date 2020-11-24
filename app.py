from flask import Flask,render_template,request,send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation


# MODEL
image_shape = (130,130,3)

model = Sequential()                                

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Dropouts help reduce overfitting by randomly turning neurons off during training.
# Here we say randomly turn off 50% of neurons.
model.add(Dropout(0.5))

# Last layer, remember its binary so we use sigmoid
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('static/malaria_detector.h5')

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1



@app.route('/')
def mymain():
    return render_template('index.html')


@app.route('/home',methods=['POST'])
def home():                                          # For Prediction 
    global COUNT
    img = request.files['image']  # loading img

    img.save(f'static/{COUNT}.jpg')    # saving img
    img_arr = cv2.imread(f'static/{COUNT}.jpg')    # converting into array

    img_arr = cv2.resize(img_arr, (130,130))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 130,130,3)
    prediction = model.predict_classes(img_arr)  # it will give class
    prediction2 = model.predict_proba(img_arr)  # it will give probabillty
    COUNT += 1
    return render_template('prediction.html',data1=prediction,data2=prediction2)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', f"{COUNT-1}.jpg")



if __name__ == "__main__":
    app.run(debug=True)