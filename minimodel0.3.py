import operator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
import cv2 as cv
import os

def createAndSaveModel():

    classifier = Sequential()

    # add convolution
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='elu'))

    # add dropout
    classifier.add(Dropout(2.5))

    # add max pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # add dropout
    classifier.add(Dropout(2.5))

    # add flattening
    classifier.add(Flatten())

    # full connection

    # hidden layer
    classifier.add(Dense(units=128, activation='elu'))

    classifier.add(Dense(units=64,activation='elu'))

    classifier.add(Dense(units=32, activation='elu'))

    # output layer
    classifier.add(Dense(units=4, activation='softmax'))

    # compile the layers
    # create the model
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    from keras.preprocessing.image import ImageDataGenerator

    # 0 - 1
    # RGB = 255, 0, 0 => 1, 0, 0
    x_train_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    x_test_generator = ImageDataGenerator(rescale=1./255)

    # generate train and test data
    x_train = x_train_generator.flow_from_directory('/home/sunbeam/Desktop/dataset0.3/train', target_size=(64, 64), batch_size=32, class_mode='categorical',color_mode='rgb')
    x_test = x_test_generator.flow_from_directory('/home/sunbeam/Desktop/dataset0.3/test', target_size=(64, 64), batch_size=32, class_mode='categorical',color_mode='rgb')

    # fit the images to the model
    classifier.fit_generator(x_train, steps_per_epoch=4646, epochs=2, validation_data=x_test, validation_steps=1000)

    # save the model
    json = classifier.to_json()
    file = open('my_model.json', 'w')
    file.write(json)
    file.close()

    # save the weights
    classifier.save_weights('weights.h5', True)


#createAndSaveModel()

def classify(testImageFile):
    from keras.models import model_from_json

    # read the json model
    file = open('my_model.json', 'r')
    data = file.read()
    print(data)

    file.close()

    # classifier will load the model from the data
    # data -> contents of the my_model.json file
    classifier = model_from_json(data)

    # load waits
    classifier.load_weights('weights.h5')

    # load the test image
    from keras.preprocessing import image

    test_image = image.load_img(testImageFile, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    result = classifier.predict(test_image)
    print(result)
    prediction = {'Finch': result[0][0],
                  'Humming_bird': result[0][1],
                  'Sparrow': result[0][2],
                  'Woodpecker': result[0][3],
                  }
    print(prediction)
    prediction = sorted(prediction.items(),key = operator.itemgetter(1),reverse = True)
    for k,v in prediction :
        if v == True:
            filename = ''
            if (k == 'Finch'):
                filename = testImageFile
                image = cv.imread(filename)

            if (k == 'Humming_bird'):
                filename = testImageFile
                image = cv.imread(filename)

            if (k == 'Sparrow'):
                filename = testImageFile
                image = cv.imread(filename)

            if (k == 'Woodpecker'):
                filename = testImageFile
                image = cv.imread(filename)

    cv.putText(image,prediction[0][0],(5,35),cv.FONT_HERSHEY_COMPLEX , 1,(0,0,255), 2)
    cv.imshow('img',image)





    if cv.waitKey(0) & 0xFF == ord('q'):
        cv.destroyAllWindows()


# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/0a496fd6793f4a7d86d8dce219dd0256.jpg') #sparrow
# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/0b0c932691be4ff99994d11884f3a7d4.jpg')  #humming
# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/3c416070003340a69baa34e220b77a09.jpg')  #humming
# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/e11b5e132bcd468288ca7aa2bac41c36.jpg')  #sparrow
# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/f8cd008f8f5d4b38b9a929ac2db79760.jpg')  #woodpecker
# classify('/home/sunbeam/Desktop/dataset0.3/test/Finch/f21bfe5facd943eda5ab03f1bf38a533.jpg')  #humming

# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/1f9075d8c8f2426f8fc9fab1f982aad7.jpg')  #sparrow
# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/2d04b19230fe402c82898afd60dcc156.jpg')
# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/2de32aae5fda4cf48c700a5de2d652bc.jpg') #wood
# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/02fd6f7599454e21ae9ff4bf5f0d3296.jpg') #sparr
# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/3defb11fd4b44c069a7e64410b7774b1.jpg') #sparr
# classify('/home/sunbeam/Desktop/dataset0.3/test/Humming_bird/4a13435ec2754f9684c4953ae253bcdd.jpg')

# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/03c07b983f184b71bb70ecff564a78d1.jpg')
# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/5b78236b2ed346aeb0eaa8fe6228a186.jpg')  #w
# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/6e360fd0ca29405db28b7911ce8dc82d.jpg')  #h
# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/7fb082e0ce79485c910904367fb32275.jpg')  #h
# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/20cb15c89aef4602a53d4145c6727ffe.jpg')
# classify('/home/sunbeam/Desktop/dataset0.3/test/Sparrow/064a3db8a69b411ea28d9936da0ee585.jpg')  #w

# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/3df2b20fcb2e406f8a3a093a6eecf9ad.jpg') #f
# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/6e23fd17da764cdc82c6f677fef41a0c.jpg')
# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/85d2d632eb9b47c78a65172929eed971.jpg') #h
# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/60219f5c1dce4da395c59ce1b71391a9.jpg') #s
# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/d91c5088371c4b0dab529bd3cb8083b1.jpg') #h
# classify('/home/sunbeam/Desktop/dataset0.3/test/Woodpecker/f361eb8be1e04537b30f567943697b28.jpg')