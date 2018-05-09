import os
import keras
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

classes = ('lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor')
activation_func = 'relu'

index = [CIFAR100_LABELS_LIST.index(el) for el in classes]

X_train = X_train[((y_train==index[0]) | (y_train==index[1]) | (y_train==index[2]) | (y_train==index[3]) | (y_train==index[4])).squeeze()]
y_train = y_train[((y_train==index[0]) | (y_train==index[1]) | (y_train==index[2]) | (y_train==index[3]) | (y_train==index[4])).squeeze()]

for i in range(len(index)):
    y_train[y_train==index[i]] = i

X_test = X_test[((y_test==index[0]) | (y_test==index[1]) | (y_test==index[2]) | (y_test==index[3]) | (y_test==index[4])).squeeze()]
y_test = y_test[((y_test==index[0]) | (y_test==index[1]) | (y_test==index[2]) | (y_test==index[3]) | (y_test==index[4])).squeeze()]

for i in range(len(index)):
    y_test[y_test==index[i]] = i

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

np.random.seed(42)
batch_size = 16
num_classes = len(classes)
epochs = 12
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation(activation_func))
model.add(Conv2D(32, (3, 3)))
model.add(Activation(activation_func))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation(activation_func))
model.add(Conv2D(64, (3, 3)))
model.add(Activation(activation_func))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation(activation_func))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()


# initiate Adam optimizer
opt = keras.optimizers.adam()

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

history = model.fit(X_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_data=(X_test, y_test), 
                        shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss: {0:.3f}'.format(scores[0]))
print('Test accuracy: {0:.3f}'.format(scores[1]))