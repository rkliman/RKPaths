import numpy as np
import scipy.integrate as integrate

def eucdist2(p1, p2):
    return np.linalg.norm(p2-p1)

def wrap_to_pi(angle):
    """
    Wrap angle in radians to the interval [-pi, pi].

    Parameters:
    angle (float or np.ndarray): Angle(s) in radians.

    Returns:
    float or np.ndarray: Wrapped angle(s) in the interval [-pi, pi].
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle

def ensure_valid_arc_left(angle1, angle2):
    # Convert angles to radians and normalize them within 0 to 2*pi
    angle1 = angle1 % (2*np.pi)
    angle2 = angle2 % (2*np.pi)
    
    # Adjust angle2 to ensure it is greater than angle1
    if angle2 <= angle1:
        angle2 += 2*np.pi
    
    return angle1, angle2

def ensure_valid_arc_right(angle1, angle2):
    # Convert angles to radians and normalize them within 0 to 2*pi
    angle1 = angle1 % (2*np.pi)
    angle2 = angle2 % (2*np.pi)
    
    # Adjust angle2 to ensure it is greater than angle1
    if angle1 <= angle2:
        angle1 += 2*np.pi
    
    return angle1, angle2

def euler_spiral_point(L,sig):
    CS, _ = integrate.quad(lambda x: np.cos(sig*x*x/2), 0, L)
    SS, _ = integrate.quad(lambda x: np.sin(sig*x*x/2), 0, L)
    return np.array([CS, SS])

def eulerspiral(L_start, L_end, sig):
    path = np.zeros((2, 20))
    vals = np.linspace(L_start, L_end, 20)
    for i in range(0,20):
        CS, _ = integrate.quad(lambda x: np.cos(sig * x*x / 2), 0, vals[i])
        SS, _ = integrate.quad(lambda x: np.sin(sig * x*x / 2), 0, vals[i])
        path[:, i] = np.array([CS, SS])
    return path

def arc_from_angles(center, radius, theta_start, theta_end, step):
    theta = np.linspace(theta_start, theta_end, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack((x, y))

def straight_path(p1, p2):
    return np.array([np.linspace(p1[0],p2[0],10), np.linspace(p1[1],p2[1],10)]), eucdist2(p1,p2)

def Tf(T,theta):
    return np.array([[np.cos(theta), -np.sin(theta), T[0]],
                     [np.sin(theta),  np.cos(theta), T[1]],
                     [0              , 0               , 1   ]])

def get_new_sigma(delta, r, mu):
    Cf, _ = integrate.quad(lambda x: np.cos(np.pi*x*x/2), 0, np.sqrt(delta/np.pi))
    Sf, _ = integrate.quad(lambda x: np.sin(np.pi*x*x/2), 0, np.sqrt(delta/np.pi))
    return (np.pi*(np.cos(delta/2)*Cf+np.sin(delta/2)*Sf)**2)/(r*r*np.sin((delta/2)+mu)*np.sin((delta/2)+mu))