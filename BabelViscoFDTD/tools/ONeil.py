import numpy as np

def calc_axial_pressure(x,p_medium,omega,c,u0,a,A):
        '''
        p_medium:   medium density
        omega:      angular frequency
        c:          propagation velocity
        u0:         max normal velocity
        a:          transducer width / 2
        A:          transducer curvature radius
        '''
        
        h=A-np.sqrt(A**2-a**2)
        k = omega / c
        B = np.sqrt((x - h)**2 + a**2)
        E = 2 / (1 - x/A)
        delta = B - x
        M = (B + x) / 2
        P = E * np.sin(k*delta/2)
        
        abs_axial_pressure = p_medium * c * u0 * P *1j * np.exp(-1j*k*M)
        
        return abs_axial_pressure
