import numpy as np
import tensorflow as tf

# Veri kümesi yükleme ve işleme
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Döngüsel sinir ağı modeli oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
model.fit(x_train, y_train, epochs=5, batch_size=128)

# Test verileriyle modeli değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
