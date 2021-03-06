import math, cmath
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import cupy
from cupy.core import core
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import decomposition
from cupy.linalg import util

if cuda.cusolver_enabled:
    from cupy.cuda import cusolver
    print('enable cusolver!')

pi = math.pi
def ixp(x):
    y = cmath.exp(x*complex(0,1))
    z = 4*2*(y-1)
    # print(x,y,z)
    return z
def main():
    Nt, Nit = 16, 25
    N2, N3 = Nt**2, Nt**3
    DIM = 4*N2
    T, U = 2.0, 4.0
    f_5 = 0.5*T*U
    delmu = 0.0
    print("dim =",DIM)
    iw = [ixp(pi*(2.*i+1.)/Nt) for i in range(Nt)]
    g = [-1./i for i in iw]
    h = [complex(0,0) for i in range(Nt)]
    dd = [complex(0,0) for i in range(Nt)]

    nnz1, nnz2, nnz3= 9*N3, 5*N2, 7*N3
    nnz4 = nnz1 + nnz2
    nnz = nnz3 + nnz4
    matcoo1 = [complex(0,0) for i in range(nnz1)]
    matcoo2 = [complex(0,0) for i in range(nnz2)]
    matcoo3 = [complex(0,0) for i in range(nnz3)]
    imatcoo1 = [0 for i in range(nnz1)]
    imatcoo2 = [0 for i in range(nnz2)]
    imatcoo3 = [0 for i in range(nnz3)]
    jmatcoo1 = [0 for i in range(nnz1)]
    jmatcoo2 = [0 for i in range(nnz2)]
    jmatcoo3 = [0 for i in range(nnz3)]
    vec = [complex(0,0) for i in range(DIM)]
    xpar = [complex(0,0) for i in range(DIM)]

    for it in range(Nit):
        print("it =",it+1)
        nh = T*sum(g)
        print("nh =",nh)
        for i in range(Nt): h[i] = -1. / (iw[i]-U/2+U*nh)
        # print("this is h: ",h)
        for b in range(Nt):
            for c in range(Nt):
                # 1.
                for lam in range(Nt):
                    tmp = b+Nt*c+N2*lam
                    F = f_5*h[(b+c-lam)%Nt]*g[b]
                    B = b+Nt*c+1
                    Nnl = lam+Nt*c+1
                    Nnc = c+Nt*lam+1

                    matcoo1[tmp], imatcoo1[tmp], jmatcoo1[tmp] = -F, B, N2+Nnl
                    matcoo1[tmp+N3], imatcoo1[tmp+N3], jmatcoo1[tmp+N3] = F, B, N2*2+Nnl
                    matcoo1[tmp+N3*2], imatcoo1[tmp+N3*2], jmatcoo1[tmp+N3*2] = F, N2+B, N2+Nnl
                    matcoo1[tmp+N3*3], imatcoo1[tmp+N3*3], jmatcoo1[tmp+N3*3] = F, N2+B, N2*2+Nnl
                    matcoo1[tmp+N3*4], imatcoo1[tmp+N3*4], jmatcoo1[tmp+N3*4] = F, N2*2+B, Nnl
                    matcoo1[tmp+N3*5], imatcoo1[tmp+N3*5], jmatcoo1[tmp+N3*5] = -F, N2*2+B, N2+Nnl
                    matcoo1[tmp+N3*6], imatcoo1[tmp+N3*6], jmatcoo1[tmp+N3*6] = F, N2*2+B, N2*3+Nnl
                    matcoo1[tmp+N3*7], imatcoo1[tmp+N3*7], jmatcoo1[tmp+N3*7] = -F, N2*3+B, N2+Nnc
                    matcoo1[tmp+N3*8], imatcoo1[tmp+N3*8], jmatcoo1[tmp+N3*8] = F, N2*3+B, N2*2+Nnc
                # 2.
                sum1, sum2 = complex(0, 0), complex(0, 0)
                for nu in range(Nt): 
                    sum1 += h[nu]*g[(b+c-nu)%Nt]
                    sum2 += h[nu]*(g[(c-b+nu)%Nt]+g[(b-c+nu)%Nt])

                tmp = b+Nt*c
                F1, F2 = f_5*sum1, f_5*sum2
                B = b+Nt*c+1

                matcoo2[tmp], imatcoo2[tmp], jmatcoo2[tmp] = 1+2*F1, B, B
                matcoo2[tmp+N2], imatcoo2[tmp+N2], jmatcoo2[tmp+N2] = 1-F2, N2+B, N2+B
                matcoo2[tmp+N2*2], imatcoo2[tmp+N2*2], jmatcoo2[tmp+N2*2] = -F2, N2+B, N2*2+B
                matcoo2[tmp+N2*3], imatcoo2[tmp+N2*3], jmatcoo2[tmp+N2*3] = 1+F2, N2*2+B, N2*2+B
                matcoo2[tmp+N2*4], imatcoo2[tmp+N2*4], jmatcoo2[tmp+N2*4] = 1+2*F1, N2*3+B, N2*3+B
                # 3.
                for lam in range(Nt):
                    tmp = b+Nt*c+N2*lam
                    F = f_5*h[(b+c-lam)%Nt]*g[c]
                    B = b+Nt*c+1
                    Nnl = lam+Nt*b+1
                    Nnb = b+Nt*lam+1

                    matcoo3[tmp], imatcoo3[tmp], jmatcoo3[tmp] = -F, B, N2+Nnl
                    matcoo3[tmp+N3], imatcoo3[tmp+N3], jmatcoo3[tmp+N3] = F, B, N2*2+Nnl
                    matcoo3[tmp+N3*2], imatcoo3[tmp+N3*2], jmatcoo3[tmp+N3*2] = F, N2+B, N2+Nnb
                    matcoo3[tmp+N3*3], imatcoo3[tmp+N3*3], jmatcoo3[tmp+N3*3] = F, N2+B, N2*2+Nnb
                    matcoo3[tmp+N3*4], imatcoo3[tmp+N3*4], jmatcoo3[tmp+N3*4] = -F, N2*2+B, N2+Nnb
                    matcoo3[tmp+N3*5], imatcoo3[tmp+N3*5], jmatcoo3[tmp+N3*5] = -F, N2*3+B, N2+Nnb
                    matcoo3[tmp+N3*6], imatcoo3[tmp+N3*6], jmatcoo3[tmp+N3*6] = F, N2*3+B, N2*2+Nnb
        # print(matcoo3)
        matcoo1, imatcoo1, jmatcoo1 = np.array(matcoo1), np.array(imatcoo1)-1, np.array(jmatcoo1)-1
        matcoo2, imatcoo2, jmatcoo2 = np.array(matcoo2), np.array(imatcoo2)-1, np.array(jmatcoo2)-1
        matcoo3, imatcoo3, jmatcoo3 = np.array(matcoo3), np.array(imatcoo3)-1, np.array(jmatcoo3)-1
        # create coo matrix
        # print(len(imatcoo1),DIM)
        Acoo = scipy.sparse.coo_matrix((matcoo1,(imatcoo1,jmatcoo1)),shape=(DIM,DIM))
        Bcoo = scipy.sparse.coo_matrix((matcoo2,(imatcoo2,jmatcoo2)),shape=(DIM,DIM))
        Ccoo = scipy.sparse.coo_matrix((matcoo3,(imatcoo3,jmatcoo3)),shape=(DIM,DIM))
        # create csr matrix
        Mat1 = scipy.sparse.csr_matrix(Acoo)
        Mat2 = scipy.sparse.csr_matrix(Bcoo)
        Mat3 = scipy.sparse.csr_matrix(Ccoo)

        # sum up
        Mat = (Mat1+Mat2+Mat3)
        mat = Mat.todense()
        # export csr

        # vec
        sum1, sum2 = complex(0,0), complex(0,0)
        for b in range(Nt):
            for c in range(Nt):
                sum1, sum2 = 0., 0.
                for nu in range(Nt):
                    sum1 += h[nu]*g[(b+c-nu)%Nt]
                    sum2 += h[nu]*(g[(b-c+nu)%Nt]+g[(c-b+nu)%Nt])
                tmp, F = b+Nt*c, f_5*U*g[b]*g[c]
                # print("F =",F,sum1,sum2);
                vec[tmp] = -4.*F*sum1
                vec[N2+tmp] = 0.
                vec[N2*2+tmp] = -2.*F*sum2
                vec[N2*3+tmp] = -4.*F*sum1
                
        # ans = scipy.linalg.solve(mat,vec)
        print("1.",type(mat))
        print(mat.shape)
        print(len(vec))
        mat = torch.tensor(mat)
        vec = torch.tensor(vec)
        print("2.",type(mat))
        ans = torch.gesv(mat,vec)
        
        for n in range(Nt):
            dd[n] = 0.
            for nu in range(Nt):
                dd[n]+=(T/6)*(ans[N2+n+Nt*nu]+2*ans[2*N2+n+Nt*nu]+ans[3*N2+n+Nt*nu])
            g[n]=h[n]*(1+dd[n])
        print(dd)
        print(g[0])

if __name__ == "__main__":
    main()