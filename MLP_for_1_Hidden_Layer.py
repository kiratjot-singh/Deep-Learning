class MLP:
    def __init__(self,hidden_size,lr=0.01,epochs=1000):
        self.w1=None
        self.w2=None
        self.lr=lr
        self.epochs=epochs
    def forward_prop():

    def activation(x,w):
        np.insert(x,0,1,axis=1)
        return np.dot(x,w)

    def fit(X_train,y_train):
        w1=np.ones(X_train.shape[1],self.hidden_size)
        w2=np.ones(hidden_size+1,1)
        for i in range(epochs):
            for j in range(X_train[0].shape[0]):
                idx=np.random.randint(0,X_train.shape[0])
                O1=activation(X_train[idx],w1)
                O21=activation(O1,w2)
                




    def predict():


       

        

