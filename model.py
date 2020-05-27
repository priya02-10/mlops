import keras
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten 
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D 
from keras.layers.normalization import BatchNormalization 
from keras.regularizers import l2 
from keras.datasets import mnist 
from keras.utils import np_utils 



model = Sequential()

# 2 sets of CRP (Convolution, RELU, Pooling)
model.add(Conv2D(20, (5, 5),
                 padding = "same", 
                 input_shape = (512,512,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2) ))

model.add(Conv2D(50, (5, 5),
                 padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(10))
model.add(Activation("softmax"))
           
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
    
print(model.summary())
