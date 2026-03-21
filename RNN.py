import numpy as np


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01   #wi
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  #wh
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  #w0
        
        
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_hs = {0: h}
        
        
        for t, x in enumerate(inputs):
            x = x.reshape(-1, 1)
            h = tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[t + 1] = h
        
        y = self.Why @ h + self.by
        return y, h
    
    def train(self, inputs, target, lr=0.01):
        
        y, h = self.forward(inputs)
        
        
        loss = np.square(y - target).sum()
        

        dWhy = (y - target) @ h.T
        dby = (y - target)
        
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        
        dh_next = np.zeros_like(h)
        
        
        for t in reversed(range(len(inputs))):
            dh = self.Why.T @ (y - target) + dh_next
            temp = tanh_deriv(self.last_hs[t+1]) * dh
            
            dbh += temp
            dWxh += temp @ inputs[t].reshape(1, -1)
            dWhh += temp @ self.last_hs[t].T
            
            dh_next = self.Whh.T @ temp
        
        
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby
        
        return loss