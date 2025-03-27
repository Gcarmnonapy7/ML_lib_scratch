import numpy as np
import matplotlib.pyplot as plt
import torch
from sympy.abc import alpha

if torch.cuda.is_available() :
    device = 'cuda'
else :
    device = 'cpu'

def boundary_frontier(model,X,y):
    '''
    :param model: your model
    :param X: data
    :param y: target
    :return: plot a decision boundary frontier//more for DLearning models
    '''
    #min and max points with a extra merge (0.1)
    x_min,x_max = X[:,0].min()-0.1,X[:,0].max()+0.1
    y_min,y_max = X[:,1].min()-0.1,X[:,1].max()+0.1

    #mininum space for x and y divide by 100 to calculate the thinest space between the points(grid)
    spacing = min(x_max-x_min,y_max-y_min)/100

    #Create a grid for the points
    XX,YY = np.meshgrid(np.arrange(x_min,x_max,spacing),np.arrange(y_min,y_max,spacing))

    # turning the data into 1D -> ravel() and reshape it after data concatenate the data into 2D
    data = np.hstack((XX.ravel().reshape(-1,1),YY.ravel(-1,1)))
    # Turning into a torch tensor and casting on GPU
    db_prob = model(torch.Tensor(data).to(device))

    clf = np.argmax(db_prob.cpu().data.numpy(),axis=1)  #picking the highest points turning into one line and numpy
    #Reshape the clf on XX.shape
    Z = clf.reshape(XX.shape)
    #Ploting the points and the boundary frontier
    plt.contourf(XX,YY,Z,cmap=plt.cm.brg,alpha=0.5)
    plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k',s=25,cmap=plt.cm.brg)