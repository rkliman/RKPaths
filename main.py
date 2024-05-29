import numpy as np
import matplotlib.pyplot as plt
from random import random
import RKPaths

def main():
    phi_max = np.radians(80)
    tractor_length = 4

    k_max = np.tan(phi_max/tractor_length)
    sig_max = np.radians(20)/tractor_length

    path = RKPaths.RKPath(k_max,sig_max)

    # q_s = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 2*k_max*random()-k_max])
    # q_g = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 2*k_max*random()-k_max])
    q_s = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 0])
    q_g = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 0])

    path, length = path.PlanPath(q_s,q_g)
    plt.plot(path[0,:],path[1,:],'k-')
    plt.plot(q_s[0],q_s[1],'g*')
    plt.plot(q_g[0],q_g[1],'r*')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()