{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's construct LeNet in Keras!\n",
    "\n",
    "#### First let's load and prep our MNIST data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.regularizers import l2\n",
    "from keras.datasets import mnist\n",
    "from keras.utils import np_utils\n",
    "import keras\n",
    "\n",
    "# loads the MNIST dataset\n",
    "(x_train, y_train), (x_test, y_test)  = mnist.load_data()\n",
    "\n",
    "# Lets store the number of rows and columns\n",
    "img_rows = x_train[0].shape[0]\n",
    "img_cols = x_train[1].shape[0]\n",
    "\n",
    "# Getting our date in the right 'shape' needed for Keras\n",
    "# We need to add a 4th dimenion to our date thereby changing our\n",
    "# Our original image shape of (60000,28,28) to (60000,28,28,1)\n",
    "x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n",
    "x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n",
    "\n",
    "# store the shape of a single image \n",
    "input_shape = (img_rows, img_cols, 1)\n",
    "\n",
    "# change our image type to float32 data type\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "\n",
    "# Normalize our data by changing the range from (0 to 255) to (0 to 1)\n",
    "x_train /= 255\n",
    "x_test /= 255\n",
    "\n",
    "# Now we one hot encode outputs\n",
    "y_train = np_utils.to_categorical(y_train)\n",
    "y_test = np_utils.to_categorical(y_test)\n",
    "\n",
    "num_classes = y_test.shape[1]\n",
    "num_pixels = x_train.shape[1] * x_train.shape[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now let's create our layers to replicate LeNet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 28, 28, 20)        520       \n",
      "_________________________________________________________________\n",
      "activation_1 (Activation)    (None, 28, 28, 20)        0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 14, 14, 20)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 14, 14, 50)        25050     \n",
      "_________________________________________________________________\n",
      "activation_2 (Activation)    (None, 14, 14, 50)        0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 7, 7, 50)          0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 2450)              0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 500)               1225500   \n",
      "_________________________________________________________________\n",
      "activation_3 (Activation)    (None, 500)               0         \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 10)                5010      \n",
      "_________________________________________________________________\n",
      "activation_4 (Activation)    (None, 10)                0         \n",
      "=================================================================\n",
      "Total params: 1,256,080\n",
      "Trainable params: 1,256,080\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# create model\n",
    "model = Sequential()\n",
    "\n",
    "# 2 sets of CRP (Convolution, RELU, Pooling)\n",
    "model.add(Conv2D(20, (5, 5),\n",
    "                 padding = \"same\", \n",
    "                 input_shape = input_shape))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))\n",
    "\n",
    "model.add(Conv2D(50, (5, 5),\n",
    "                 padding = \"same\"))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))\n",
    "\n",
    "# Fully connected layers (w/ RELU)\n",
    "model.add(Flatten())\n",
    "model.add(Dense(500))\n",
    "model.add(Activation(\"relu\"))\n",
    "\n",
    "# Softmax (for classification)\n",
    "model.add(Dense(num_classes))\n",
    "model.add(Activation(\"softmax\"))\n",
    "           \n",
    "model.compile(loss = 'categorical_crossentropy',\n",
    "              optimizer = keras.optimizers.Adadelta(),\n",
    "              metrics = ['accuracy'])\n",
    "    \n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now let us train LeNet on our MNIST Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/10\n",
      "60000/60000 [==============================] - 56s 940us/step - loss: 0.1870 - accuracy: 0.9412 - val_loss: 0.0469 - val_accuracy: 0.9849\n",
      "Epoch 2/10\n",
      "60000/60000 [==============================] - 69s 1ms/step - loss: 0.0474 - accuracy: 0.9852 - val_loss: 0.0328 - val_accuracy: 0.9884\n",
      "Epoch 3/10\n",
      "60000/60000 [==============================] - 65s 1ms/step - loss: 0.0307 - accuracy: 0.9905 - val_loss: 0.0259 - val_accuracy: 0.9908\n",
      "Epoch 4/10\n",
      "60000/60000 [==============================] - 57s 953us/step - loss: 0.0224 - accuracy: 0.9929 - val_loss: 0.0269 - val_accuracy: 0.9912\n",
      "Epoch 5/10\n",
      "60000/60000 [==============================] - 65s 1ms/step - loss: 0.0159 - accuracy: 0.9951 - val_loss: 0.0220 - val_accuracy: 0.9920\n",
      "Epoch 6/10\n",
      "60000/60000 [==============================] - 58s 963us/step - loss: 0.0118 - accuracy: 0.9966 - val_loss: 0.0329 - val_accuracy: 0.9897\n",
      "Epoch 7/10\n",
      "60000/60000 [==============================] - 70s 1ms/step - loss: 0.0086 - accuracy: 0.9974 - val_loss: 0.0244 - val_accuracy: 0.9930\n",
      "Epoch 8/10\n",
      "60000/60000 [==============================] - 70s 1ms/step - loss: 0.0061 - accuracy: 0.9983 - val_loss: 0.0243 - val_accuracy: 0.9924\n",
      "Epoch 9/10\n",
      "60000/60000 [==============================] - 61s 1ms/step - loss: 0.0051 - accuracy: 0.9984 - val_loss: 0.0265 - val_accuracy: 0.9932\n",
      "Epoch 10/10\n",
      "60000/60000 [==============================] - 63s 1ms/step - loss: 0.0031 - accuracy: 0.9993 - val_loss: 0.0273 - val_accuracy: 0.9914\n",
      "10000/10000 [==============================] - 3s 286us/step\n",
      "Test loss: 0.027261309696467294\n",
      "Test accuracy: 0.9914000034332275\n"
     ]
    }
   ],
   "source": [
    "# Training Parameters\n",
    "batch_size = 128\n",
    "epochs = 1\n",
    "\n",
    "history = model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=epochs,\n",
    "          validation_data=(x_test, y_test),\n",
    "          shuffle=True)\n",
    "\n",
    "model.save(\"mnist_LeNet.h5\")\n",
    "\n",
    "# Evaluate the performance of our trained model\n",
    "scores = model.evaluate(x_test, y_test, verbose=1)\n",
    "print('Test loss:', scores[0])\n",
    "print('Test accuracy:', scores[1])\n",
    "accuracy = scores[1]*100\n",
    "\n",
    "\n",
    "accuracyfile = open(\"/ml_task/accuracy.txt\", \"w\")\n",
    "accuracyfile.write(str(acc))\n",
    "accuracyfile.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
