import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

#Carregar os dados de treinamento e teste do MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Pré-processamento dos dados
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Criar a arquitetura da rede neural
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Treinar a rede neural
model.fit(train_images, train_labels, epochs=5, batch_size=64)

#Avaliar o desempenho do modelo nos dados de teste
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)