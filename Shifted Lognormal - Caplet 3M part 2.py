#Same setup but now Implied volatilities of the 3.75% strikes is 20.50%

import numpy as np
import scipy.stats as si

#Set up initial values given in the assignment
K = 0.0375
F = 0.04
r = 0.04
t = 1
T = 1.25
delta = T - t

sigma = 0.205

#Calculate price of caplet with strike 3.75% using Black Function
d1 = (np.log(F/K) + (0.5*sigma**2)*t)/(sigma*np.sqrt(t))
d2 = d1 - sigma*np.sqrt(t)
N_d1 = si.norm.cdf(d1, 0.0, 1.0)
N_d2 = si.norm.cdf(d2, 0.0, 1.0)
C = (F*N_d1 - K*N_d2)*np.exp(-r*T)*delta

#-----------------------------------------------------------------------------#

#Calibrate m and sigma m of the shifted lognormal model
m_range = np.arange(0.0001,0.2,0.00001)

for m in m_range:
    Fm = F+m
    Km = K+m
    sigmam = 0.008/(0.04+m)
        
    d1m = (np.log(Fm/Km) + (0.5*sigmam**2)*t)/(sigmam*np.sqrt(t))
    d2m = d1m - sigmam*np.sqrt(t)
    N_d1m = si.norm.cdf(d1m, 0.0, 1.0)
    N_d2m = si.norm.cdf(d2m, 0.0, 1.0)
    Cm = (Fm*N_d1m - Km*N_d2m)*np.exp(-r*T)*delta

    if np.abs(Cm-C)>1.0e-9:
        continue
    else:
        break


print('Shift paramter m =', m)
print('Shifted Lognormal model volatility:', sigmam)
print('Caplet price for strike 3.75%, Black Function:', C)
print('Caplet price for strike 3.75%, Shifted Lognormal:', Cm)    

#-----------------------------------------------------------------------------#
#check with strike = 4%

#Using Black function
K1 = 0.04
F1 = 0.04
sigma1 = 0.2

d1_1 = (np.log(F1/K1) + (0.5*sigma1**2)*t)/(sigma1*np.sqrt(t))
d2_1 = d1_1 - sigma1*np.sqrt(t)
N_d1_1 = si.norm.cdf(d1_1, 0.0, 1.0)
N_d2_1 = si.norm.cdf(d2_1, 0.0, 1.0)
C1 = (F1*N_d1_1 - K1*N_d2_1)*np.exp(-r*T)*delta

#Using shifted lognormal model
K1m = 0.04+m
F1m = 0.04+m

sigma1m = 0.008/(0.04+m)

d1_1m = (np.log(F1m/K1m) + (0.5*sigma1m**2)*t)/(sigma1m*np.sqrt(t))
d2_1m = d1_1m - sigma1m*np.sqrt(t)
N_d1_1m = si.norm.cdf(d1_1m, 0.0, 1.0)
N_d2_1m = si.norm.cdf(d2_1m, 0.0, 1.0)
C1m = (F1m*N_d1_1m - K1m*N_d2_1m)*np.exp(-r*T)*delta

print('Caplet price for strike 4%, Black Function:', C1)
print('Caplet price for strike 4%, Shifted Lognormal:', C1m)