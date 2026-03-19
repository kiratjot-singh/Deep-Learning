import numpy as np
from sklearn.datasets import load_digits

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def sigmoid(x): return 1/(1+np.exp(-x))
def loss_fn(p,y): return -(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))

def conv2d(x,k):
    h,w=x.shape
    kh,kw=k.shape
    out=np.zeros((h-kh+1,w-kw+1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j]=np.sum(x[i:i+kh,j:j+kw]*k)
    return out

def conv2d_backward(dout,x,k):
    kh,kw=k.shape
    dk=np.zeros_like(k)
    for i in range(dout.shape[0]):
        for j in range(dout.shape[1]):
            dk+=dout[i,j]*x[i:i+kh,j:j+kw]
    return dk

def maxpool(x):
    h,w=x.shape
    out=np.zeros((h//2,w//2))
    mask=np.zeros_like(x)
    for i in range(0,h,2):
        for j in range(0,w,2):
            patch=x[i:i+2,j:j+2]
            m=np.max(patch)
            out[i//2,j//2]=m
            idx=np.unravel_index(np.argmax(patch),patch.shape)
            mask[i+idx[0],j+idx[1]]=1
    return out,mask

def maxpool_backward(dout,mask):
    dx=np.zeros_like(mask)
    h,w=mask.shape
    for i in range(0,h,2):
        for j in range(0,w,2):
            dx[i:i+2,j:j+2]=mask[i:i+2,j:j+2]*dout[i//2,j//2]
    return dx

data = load_digits()
x_train = data.images[:200] / 16.0
y_train = (data.target[:200] % 2 == 0).astype(int)

k = np.random.randn(3,3)*0.1
W = np.random.randn(9,1)*0.1
b = 0.0
lr = 0.02

for epoch in range(100):
    total_loss=0
    for i in range(len(x_train)):
        x=x_train[i]
        y=y_train[i]

        conv=conv2d(x,k)
        rel=relu(conv)
        pool,mask=maxpool(rel)
        flat=pool.flatten().reshape(-1,1)

        z=np.dot(W.T,flat)+b
        p=sigmoid(z)[0][0]

        loss=loss_fn(p,y)
        total_loss+=loss

        dz=p-y
        dW=dz*flat
        db=dz

        dflat=dz*W
        dpool=dflat.reshape(pool.shape)
        drel=maxpool_backward(dpool,mask)
        dconv=drel*relu_deriv(conv)
        dk=conv2d_backward(dconv,x,k)

        W-=lr*dW
        b-=lr*db
        k-=lr*dk

    print("epoch",epoch,"loss",total_loss/len(x_train))