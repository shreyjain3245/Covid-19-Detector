# %%
# Dataset : http://cb.lk/covid_19

# %%
TRAIN_PATH = "CovidDataset/Train"
VAL_PATH = "CovidDataset/Val"

# %%
import numpy as np
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# %%
# CNN Based Model in Keras
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # To check overfitting

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,
                activation='sigmoid'))  # Output layer we need single neuron i.e and since binary classification so we use sigmoid function

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
# adam optimizer for gradient descent and accurary metrics as classification matrix


# %%
model.summary()

# %%
# Train from scratch

train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_dataset = image.ImageDataGenerator(
    rescale=1. / 255,
)

# %%
train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train',
    target_size=(224, 224),  # same as input size in mode, to reshape it
    batch_size=32,
    class_mode='binary',  # Since Binary classification, if multiple classes then set to categorical

)

# %%
train_generator.class_indices

# %%
validation_generator = test_dataset.flow_from_directory(
    'CovidDataset/Val',
    target_size=(224, 224),  # same as input size in mode, to reshape it
    batch_size=32,
    class_mode='binary',  # Since Binary classification, if multiple classes then set to categorical

)

# %%
hist = model.fit_generator(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)

# %%
# Class Activation Maps
# Grad-CAM

# %%
model.save('model_CNN')

# %%
model.evaluate_generator(train_generator)

# %%
model.evaluate_generator(validation_generator)

# %%
"""
 Test Images
"""

# %%
model = load_model('model_CNN.h5')

# %%
import os

# %%
train_generator.class_indices

# %%
y_actual = []
y_test = []

# %%
for i in os.listdir("./CovidDataset/Val/Normal/"):
    img = image.load_img("./CovidDataset/Val/Normal/" + i, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = (model.predict(img) > 0.5).astype("int32")  # Since sigmoid function
    y_test.append(p[0, 0])
    y_actual.append(1)

# %%
for i in os.listdir("./CovidDataset/Val/Covid/"):
    img = image.load_img("./CovidDataset/Val/Covid/" + i, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = (model.predict(img) > 0.5).astype("int32")  # Since sigmoid function
    y_test.append(p[0, 0])
    y_actual.append(0)

# %%
y_actual = np.array(y_actual)
y_test = np.array(y_test)

# %%
from sklearn.metrics import confusion_matrix

# %%
cm = confusion_matrix(y_actual, y_test)

# %%
import seaborn as sns

# %%
sns.heatmap(cm, cmap="plasma", annot=True)

# %%
