import numpy as np
import matplotlib.pyplot as plt
from random import random
import RKPaths
from shapely import Polygon, LinearRing

def main():
    phi_max = np.radians(80)
    tractor_length = 4
    tractor_width = 2

    k_max = np.tan(phi_max/tractor_length)
    sig_max = np.radians(20)/tractor_length

    path = RKPaths.RKPath(k_max,sig_max)


    # Create arbitrary polygon
    poly = generate_random_object(size_range=(-3,3))
    x,y = poly.exterior.xy
    buffer = poly.buffer(tractor_width)
    plt.fill(x,y)
    x,y = buffer.exterior.xy
    plt.plot(x,y,'r--')

    # q_s = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 2*k_max*random()-k_max])
    # q_g = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 2*k_max*random()-k_max])
    q_s = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 0])
    q_g = np.array([(10*random()-5),(10*random()-5),(2*np.pi*random()-np.pi), 0])

    pth_r, length = path.PlanPath(q_s,q_g, [])
    pth, length = path.PlanPath(q_s,q_g, [buffer])
    plt.plot(pth_r[0,:],pth_r[1,:],'b-')
    plt.plot(pth[0,:],pth[1,:],'g-')
    plt.plot(q_s[0],q_s[1],'g*')
    plt.plot(q_g[0],q_g[1],'r*')
    plt.axis('equal')
    plt.show()

def generate_random_object(num_points=5, size_range=(0, 100)):
    # 1. Generate random x and y coordinates
    x = np.random.uniform(size_range[0], size_range[1], num_points) + np.random.uniform(size_range[0], size_range[1])
    y = np.random.uniform(size_range[0], size_range[1], num_points) + np.random.uniform(size_range[0], size_range[1])
    
    # 2. Find the centroid (mean) of the points
    center_x, center_y = np.mean(x), np.mean(y)
    
    # 3. Calculate the angle of each point relative to the centroid
    angles = np.arctan2(y - center_y, x - center_x)
    
    # 4. Sort points by angle to ensure a sequential, non-intersecting path
    sorted_indices = np.argsort(angles)
    points = np.column_stack((x[sorted_indices], y[sorted_indices]))
    
    # 5. Create the Shapely Polygon
    return Polygon(points)


if __name__ == '__main__':
    main()