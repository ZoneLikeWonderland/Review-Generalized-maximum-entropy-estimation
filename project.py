import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("seaborn-whitegrid")

M = 3
r = 5
n_bins = 100
xs = np.linspace(0, 1, n_bins)

A_norm = M**2


def Algorithm_1(yita):
    def L():
        return (1/yita[0]+A_norm+yita[1]).item()

    def x_z_star(z):

        r = z/yita[0]-y

        if np.linalg.norm(r, ord=np.inf) < delta:
            return y+r

        r /= np.linalg.norm(r)
        r *= delta
        return y+r

    def A_star_z(z, x):
        ret = 0
        for i in range(1, M+1):
            ret += z[i-1].item()*x**i
        return ret

    def A_mu_z_star(z):
        ret = np.matrix(np.zeros((M, 1)))
        for i in range(1, M+1):
            ret[i-1] = 0
            nom = 0
            denom = 0
            for x in xs:
                e = 2**-A_star_z(z, x)
                nom += x**i*e
                denom += e
            ret[i-1] = nom/denom
        return ret

    count = 0

    def mu_z(z):

        prs = []
        for x in xs:
            e = 2**-A_star_z(z, x)
            prs.append(e)
        prs = np.array(prs)
        prs /= prs.sum()
        prs *= n_bins
        return prs

    def save(z, force=False):
        nonlocal count
        count += 1
        if count % 100 != 0 and not force:
            return

        prs = mu_z(z)
        # print(prs)
        plt.ion()
        plt.cla()
        plt.plot(xs, mu0)
        plt.plot(xs, prs)
        # plt.ylim(0, None)
        plt.ylim(0.6, 1.6)
        plt.xlim(0, 1)
        plt.title(f"{count} {z.T}")
        plt.draw()
        plt.show()
        plt.pause(0.000001)

        import json
        json.dump(
            {
                "U": U0,
                "eps": eps,
                "p": prs.tolist(),
                "count": count,
                "Fy": F_yita(z).item(),
                "time": time.time()-start
            },
            open(f"U_{U0}_eps_{eps}.json", "w")
        )

    def F_yita(z):

        x_star = x_z_star(z)
        mu = mu_z(z)

        return -(x_star.T*z-yita[0]/2*np.linalg.norm(x_star)**2)+D_KL_niu(mu)+loA(mu).T*z-yita[1]/2*np.linalg.norm(z)**2

    def Gf_yita(z):

        a = -x_z_star(z)
        b = A_mu_z_star(z)
        c = -yita[1].item()*z

        g = a+b+c
        # print(a.T, b.T, c.T, end=" ")

        return g

    import time
    start = time.time()

    w_k = np.matrix(np.zeros((M, 1)))
    y_k = np.matrix(np.zeros((M, 1)))

    ll = 1/L()
    lr = ((L())**0.5-(yita[1].item())**0.5)/((L())**0.5+(yita[1].item())**0.5)
    for k in range(99999):
        y_k1 = w_k+ll*Gf_yita(w_k)
        w_k1 = y_k1+lr*(y_k1-y_k)

        if np.abs(w_k-w_k1).max() < 1e-6 and np.abs(y_k-y_k1).max() < 1e-6:
            save(y_k, True)
            print(f"F_yita = {F_yita(y_k)}")
            break

        w_k = w_k1
        y_k = y_k1
        # print(y_k1.T, lr, yita.T, k)
        save(y_k)

    return y_k1


def slater_point():
    beta = np.matrix(np.ones((M+1, 1)))
    beta[0] = 1
    beta[1:] = y

    A = np.matrix(np.zeros((M+1, r)))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 1/(i+j+1)

    print(A)
    print(A.shape, beta.shape)

    u, s, v = np.linalg.svd(A)
    print(s)
    print(v)
    zero = v[4:].T

    alpha = np.linalg.pinv(A)*beta
    print(alpha)

    def pr(alpha, x):
        s = 0
        for i in range(r):
            s += alpha[i]*x**i
        return s

    alpha0 = alpha
    print(alpha)
    print(zero)
    k = 0
    alpha = alpha0 + zero*k

    prs = []
    for x in xs:
        prs.append(pr(alpha, x).item())
    mu0 = np.array(prs)
    return mu0

    # for k in np.linspace(60, 80):
    for k in np.linspace(-10, 10):
        alpha = alpha0 + zero*k
        assert np.allclose(A*alpha, beta)

        prs = []
        for x in xs:
            prs.append(pr(alpha, x).item())

        mu0 = np.array(prs)

        C = D_KL_niu(mu0)
        print(C, k)

        plt.ion()
        plt.plot(xs, prs)
        plt.show()
        plt.ylim(0.6, 1.6)
        plt.pause(0.001)

    exit()
    return mu0


def loA(mu):
    ret = np.matrix(np.zeros((M, 1)))
    for i in range(M):
        for x, m in zip(xs, mu):
            ret[i] += x**(i+1)*m
    return ret/n_bins


def D_KL_niu(mu):
    ret = 0
    for x, m in zip(xs, mu):
        ret -= m*np.log2(1/m)
        # print(m)
    return ret/n_bins


y = np.matrix(np.array(((1-np.log(2))/np.log(2), (np.log(4)-1)/np.log(4), (5-np.log(64))/np.log(64)))).T

mu0 = slater_point()
print("mu0:")
print(loA(mu0))


json.dump(
    {
        "U": 0,
        "eps": 0,
        "p": mu0.tolist(),
        "count": 0
    },
    open(f"U_{0}_eps_{0}.json", "w")
)
# exit()

# print("D_KL_niu(1)")
# print(D_KL_niu(mu0))

U0 = 1e0
eps = 1e-3
for U0 in [1e0, 1e-1, 1e-2, 0.005]:
    for eps in [1e0, 1e-1, 1e-2, 1e-3]:

        U = [-U0, U0]

        C = D_KL_niu(mu0)
        D = np.linalg.norm(y+U0)/2
        delta = min(-U[0], U[1])

        yita = np.matrix([eps/4/D, eps*delta**2/2/C**2]).T

        print(f"C = {C}")
        print(f"D = {D}")
        print(f"delta = {delta}")
        Algorithm_1(yita)
