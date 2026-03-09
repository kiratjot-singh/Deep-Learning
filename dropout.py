import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(42)

X = np.random.rand(1000,1) * 10
y = 3*X + 7 + np.random.randn(1000,1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(128,input_dim=1,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="linear"))
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
model.fit(X_train, y_train, epochs=50, batch_size=32)