from sklearn.datasets import make_moons
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X, y = make_moons(n_samples=500, noise=0.2)

model = Sequential([
    Dense(60, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=40)
print(model.predict(X[:5]))