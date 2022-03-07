from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model


M = 28
N = 28
h = 4
c = 10


inputs = Input((M, N, 1))


x = Conv2D(filters=3, kernel_size=(2, 2), padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

for i in range(h - 1):
    x = Conv2D(filters=3, kernel_size=(2, 2), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
outputs = Dense(c)(x)

model = Model(inputs, outputs)
model.summary()