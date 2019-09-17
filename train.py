from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, \
                         Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle


NUM_CLASSES = 4
IMG_SIZE = 48
EPOCHS = 32
BATCH_SIZE = 64


'''
Prepare data for neural network.
'''

#Open serialized image and label data.
with open("./data.pkl", "rb") as f:
    faces, labels = pickle.load(f)
    f.close()

#Split data into testing and training sets and preprocess image data.
X_train, X_test, Y_train, Y_test = train_test_split(faces,
                                                    labels,
                                                    test_size = 0.1,
                                                    shuffle = True)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], IMG_SIZE, IMG_SIZE, 1)
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)


'''
Define model structure.
'''

#Initialize model.
model = Sequential()

#Convolutional layers with pooling.
model.add(Conv2D(32, (5, 5), activation = "relu",
    input_shape = (IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size = (5, 5), strides = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(AveragePooling2D(pool_size = (3, 3), strides = (2, 2)))

#Fully connected layer followed by 0.5 dropout.
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))

#Output layer.
model.add(Dense(NUM_CLASSES, activation = "softmax"))


#Compile model.
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

print ("\n\nTraining....\n\n")

#Declare callbacks.
filepath = "./models/model-{epoch:02d}-{val_acc:.2f}.h5"
callbacks = [EarlyStopping(monitor = "val_loss", patience = 0),
             ModelCheckpoint(filepath = filepath,
                             monitor = "val_loss",
                             save_best_only = False,
                             mode = "auto",
                             period = 1)]

#Fit model and store statistics.
stats = model.fit(np.array(X_train),
                  Y_train,
                  epochs = EPOCHS,
                  batch_size = BATCH_SIZE,
                  validation_data = (X_test, Y_test),
                  shuffle = True,
                  verbose = 1,
                  callbacks = callbacks)


'''
Evaluate models.
'''

acc = stats.history["acc"]
val_acc = stats.history["val_acc"]
loss = stats.history["loss"]
val_loss = stats.history["val_loss"]
x = range(len(acc))

#Plot accuracy vs epoch.
plt.subplot(2, 1, 1)
plt.plot(x, acc, "ko", label = "Training", linewidth = 3)
plt.plot(x, val_acc, "k", label = "Validation", linewidth = 3)
plt.ylabel("Accuracy")
plt.ylim((0,1))
plt.legend()

#Plot loss vs epoch.
plt.subplot(2, 1, 2)
plt.plot(x, loss, "ko", label = "Training", linewidth = 3)
plt.plot(x, val_loss, "k", label = "Validation", linewidth = 3)
plt.ylabel("Loss")
plt.xlabel("Epoch")

#Save figure.
plt.savefig("evaluation.png")
