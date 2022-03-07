from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model


M = 28
N = 28
h = 4
c = 10


# inputs = Input((M, N))
# x = Flatten()(inputs)
# x = Dense(128, activation='softmax')(x)
# for i in range(h - 1):
#     x = Dense(128, activation='softmax')(x)
# outputs = Dense(c)(x)

inputs = Input(2)
outputs = Dense(4)(inputs)
model = Model(inputs, outputs)
model.summary()