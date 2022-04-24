import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.express as px

def make_plot(x_from, x_to, n_points, alpha, beta):
    def make_coefficient(i: int, j: int) -> float:
        u = 50*(i/np.pi - np.floor(i/np.pi))
        v = 50*(j/np.pi - np.floor(j/np.pi))
        w = u*v*(u+v)
        return 2*(w - np.floor(w)) - 1

    def height_coefficients(x_from,x_to):
        flat_array = np.array([make_coefficient(ii,jj) for ii in range(x_from,x_to+1) for jj in range(x_from,x_to+1)])
        return flat_array.reshape((x_to-x_from+1,x_to-x_from+1))

    x = np.linspace(x_from,x_to,n_points)
    z = x
    i = np.floor(x).astype(int)
    j = np.floor(z).astype(int)
    X,Z = np.meshgrid(x,z)


    a = height_coefficients(x_from,x_to)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    d = np.zeros_like(a)
    b[0:-1,:] = a[1:,:] #b[i-1,j] = a[i,j]
    c[:,0:-1] = a[:,1:] #c[i,j-1] = a[i,j]
    d[0:-1,0:-1] = a[1:,1:] #d[i-1,j-1] = a[i,j]



    def S(x: float, alpha: float, beta: float) -> float:

        lamda = np.clip(np.clip((x-alpha)/(beta-alpha),0,np.inf),-np.inf,1)
        return lamda*lamda*(3 - 2*lamda)




    first = a[i,j]+(b[i,j]-a[i,j])*S(X-i,alpha,beta)+(c[i,j]-a[i,j])*S(Z-j,alpha,beta)+(a[i,j] -b[i,j] - c[i,j] + d[i,j])*S(X-i,alpha,beta)*S(Z-j,alpha,beta)
    return X,Z,first


def graph_update(x_from,x_to,n_points,alpha,beta):
    print(n_points)
    X,Z,first = make_plot(x_from,x_to,n_points,alpha,beta)
    fig = go.Figure(data=[go.Surface(z=first)])
    return fig  
if __name__ == '__main__':
        # vectorized
    n_points = 140
    alpha = 0.6
    beta = 0.1
    x_from = 0
    x_to = 5 
    fig = graph_update(x_from,x_to,n_points,alpha,beta)
    fig.show()

    """     fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X, Z, first, cmap='viridis')
        plt.show() """
    

