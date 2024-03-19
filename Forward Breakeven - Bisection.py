#Find the breakeven level of a trigger forward with barrier H and strike K
#Forward's payoff: ST-K if ST>H, 0 otherwise
#Algorithm: bisection

import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from scipy import optimize

#set up initial value:
S0 = 100
r = 0.02
q = 0.01
sigma = 0.2
K = 110
T = 1

def bisect_alg(f, h_ini, xtol, *args):
    
    maxits = 10000
    
    h_low      = h_ini[0]
    h_high     = h_ini[1]
    
    
    # check initialization
    f_low = f(h_low,S0,r,q,sigma,T)
    f_high = f(h_high,S0,r,q,sigma,T)
    
    if f_low*f_high > 0:
        print("initialization failed; fl = %8.4e, fh = %8.4e" % (f_low, f_high))
        x = float('-inf')
        fx = float('-inf')
    else:
        h_mid  = (h_low + h_high)/2
        f_mid = f(h_mid,S0,r,q,sigma,T)
    
        for its in range (1,maxits+1):
            if f_mid*f_low < 0:
                h_high = h_mid
                f_high = f_mid
            else:
                h_low = h_mid
                f_low = f_mid
    
            h_mid  = (h_low + h_high)/2
            f_mid = f(h_mid,S0,r,q,sigma,T)
        
            if (h_high - h_low) <= xtol*(1+abs(h_low)+abs(h_high)):
                if abs(f_mid) <= xtol:
                    print('successful convergence')
                    break
    
        H   = h_mid
        fx  = f_mid
    
        if its>=maxits:
            print('no convergence. function value is ', str(f_mid))
    
    return H, fx

def forward_price(H,S0,r,q,sigma,T):

    d1 = (np.log(S0/H) + (r - q + 0.5*sigma**2) * T)/(sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    N_d1 = si.norm.cdf(d1, 0.0, 1.0)
    N_d2 = si.norm.cdf(d2, 0.0, 1.0)
    
    C = S0 * np.exp(-q*T) * N_d1 - K * np.exp(-r*T) * N_d2
    
    return C

result = bisect_alg(forward_price, (50, 110), 1.0e-6, S0,r,q,sigma,T)
print('The breakeven level of H is', result[0])
