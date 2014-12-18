'''Kalman filtering / smoothing test module.'''


from .. import kalman


class EulerDiscretizedAtmosphericReentry:
    nx = 5
    nu = 0
    nw = 2
    ny = 2
    
    w_cov = np.diag([2.4064e-5, 2.4064e-5])
    v_cov = np.diag([0.017, 0.001]) ** 2
    
    def f(self, k, x, u=None, w=None):
        [x1, x2, x3, x4, x5] = x
        [w1, w2] = w if w is not None else [0, 0]
        
        beta0 = -0.59783
        H0 = 13.406
        Gm0 = 3.9860e5
        R0 = 6374
        
        R = np.hypot(x1, x2)
        V = np.hypot(x3, x4)
        beta = beta0 * np.exp(x5)
        D = beta * np.exp((R0 - R) / H0) * V
        G = -Gm0 / R ** 3
        
        dx1 = x3
        dx2 = x4
        dx3 = D * x3 + G * x1
        dx4 = D * x4 + G * x2
        dx5 = np.zeros_like(x5)
        
        T = 0.05
        f = np.array([dx1, dx2, dx3, dx4, dx5])
        perturb = np.array([0, 0, w1, w2, 0])
        return x + f * T + perturb * np.sqrt(T)
    
    def h(self, t, x, u=None):
        [x1, x2, x3, x4, x5] = x
        
        xr = 6374
        yr = 0
        
        rr = np.hypot(x1 - xr, x2 - yr)
        theta = np.atan2(x2 - yr, x1 - xr)
        
        return np.array([rr, theta])


def sim():
    model = EulerDiscretizedAtmosphericReentry()
    
    x = np.zeros((4000, model.nx))
    x[0] = [6500.4, 349.14, -1.8093, -6.7967, 0.6932]
    
    for k, xk in enumerate(x[:-1]):
        wk = np.random.multivariate_normal([0, 0], model.w_cov)
        x[k+1] = model.f(k, xk, [], wk)
