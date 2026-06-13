import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

cnn = Sequential()

cnn.add(Conv2D(32,(3,3), activation= "relu", input_shape = (64,64,3)))
cnn.add(MaxPool2D(pool_size= (2,2)))
cnn.add(Conv2D(64,(3,3), activation= "relu"))
cnn.add(MaxPool2D(pool_size= (2,2)))
cnn.add(Conv2D(128,(3,3), activation= "relu"))
cnn.add(MaxPool2D(pool_size= (2,2)))

cnn.add(Flatten())

cnn.add(Dense(128, activation= "relu"))
cnn.add(Dense(64, activation= "relu"))
cnn.add(Dense(32, activation= "relu"))
cnn.add(Dense(1, activation= "sigmoid"))

cnn.compile(optimizer= "adam", loss= "binary_crossentropy", metrics= ["accuracy"])

train_data_generate = ImageDataGenerator(rescale= 1./255, shear_range= 0.2, zoom_range= 0.2, horizontal_flip= True, validation_split= 0.2)

train_data = train_data_generate.flow_from_directory("Deep Learning Data", target_size= (64,64), batch_size= 32, class_mode= "binary", subset= "training")
validation_data = train_data_generate.flow_from_directory("Deep Learning Data", target_size=(64,64), batch_size=32, class_mode="binary", subset="validation")

history = cnn.fit(train_data, validation_data= validation_data, epochs= 10)

print("Train Accuracy :", history.history["accuracy"][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])

print("Train Loss:", history.history['loss'][-1])
print("Validation Loss:", history.history['val_loss'][-1])

cnn.save("cats_dogs_classifier.keras")