import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_decision_boundary(clf, X, y):
    X = np.array(X)
    y = np.array(y)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    w2 = pca.transform(clf.m.reshape(1, -1))[0]
    b = clf.b

   
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    x_points = np.linspace(x_min, x_max, 200)

    if w2[1] != 0:
        y_points = -(w2[0] * x_points + b) / w2[1]
    else:
        x_points = [-b / w2[0]] * 200
        y_points = np.linspace(X2[:,1].min() - 1, X2[:,1].max() + 1, 200)

   
    plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.plot(x_points, y_points, 'k--', linewidth=2)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Perceptron Decision Boundary (PCA Projected)")
    plt.show()


class perceptron_trick:
    def __init__(self,lr=0.1,epochs=1000):
        self.b=None
        self.m=None
        self.lr=lr
        self.epochs=epochs
    def activation(self,z):
        if(z>=0):
            return 1
        if(z<0):
            return 0
    def fit(self,X,y):
        self.m=np.zeros(X.shape[1])
        self.b=0
        for i in range(self.epochs):
            for j in range(X.shape[0]):
                idx=np.random.randint(0,X.shape[0])
                z=np.dot(X[idx],self.m)+self.b
                y_hat=self.activation(z)
                self.m=self.m+self.lr*(y[idx]-y_hat)*X[idx]
                self.b=self.b+self.lr*(y[idx]-y_hat)
        return self.m,self.b

    def predict(self,X_test):
        return self.activation(np.dot(X_test,self.m)+self.b)




x = np.array([
    [1.0, 2.0, 1.5, 3.2],
    [1.2, 1.8, 1.3, 2.9],
    [2.0, 1.0, 2.2, 3.1],
    [6.0, 5.5, 5.8, 5.0],
    [7.1, 6.2, 6.0, 6.1],
    [8.0, 7.5, 6.9, 7.2]
])


y = np.array([0, 0, 0, 1, 1, 1])


clf = perceptron_trick(lr=0.1, epochs=20)
clf.fit(x, y)
y_pred = []
for sample in x:
    y_pred.append(clf.predict(sample))

y_pred = np.array(y_pred)
plot_decision_boundary(clf,x,y)

print("Weights:", clf.m)
print("Bias:", clf.b)
print("Predictions:", y_pred)
print("Accuracy:", np.mean(y_pred == y))