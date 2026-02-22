import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class perceptron:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr=lr
        self.epochs=epochs
        self.m=None
        self.b=None
    def fit(self,X_train,y_train):
        self.m=np.ones(X_train.shape[1])
        self.b=0
        for j in range(X_train.shape[0]):
            idx=np.random.randint(0,X_train.shape[0])
            y=y_train[idx]
            y_cap=np.dot(self.m,X_train[idx])+self.b
            if(y*y_cap<0):
                slope1=-(y*X_train[idx])
                slope2=-y
                self.m=self.m-self.lr*slope1
                self.b=self.b-self.lr*slope2

    def predict(self,X_test):
        return np.dot(X_test,self.m)+self.b




def plot_decision_boundary(model, X, y):
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx.shape)

    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

   
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

    
    m1, m2 = model.m
    b = model.b

   
   

    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()





np.random.seed(42)


class_pos = np.random.randn(100, 2) + np.array([2, 2])
y_pos = np.ones(100)


class_neg = np.random.randn(100, 2) + np.array([-2, -2])
y_neg = -np.ones(100)
X= np.vstack((class_pos, class_neg))
y= np.hstack((y_pos, y_neg))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model=perceptron()
model.fit(X_train,y_train)
plot_decision_boundary(model,X_train,y_train)

y_pred=model.predict(X_test)
y_pred_labels=np.where(y_pred >= 0, 1, -1)
accuracy = np.mean(y_pred_labels == y_test)
print("Accuracy:", accuracy)

