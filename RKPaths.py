import matplotlib.pyplot as plt
import scipy.integrate as integrate
import numpy as np

import util

class RKPath:
    def __init__(self,k_max,sig_max):
        self.k_max = k_max
        self.sig_max = sig_max
        CS, _ = integrate.quad(lambda x: np.cos(sig_max*x*x/2), 0, k_max/sig_max)
        SS, _ = integrate.quad(lambda x: np.sin(sig_max*x*x/2), 0, k_max/sig_max)
        self.mu = np.arctan2(CS - (1/k_max)*np.sin((k_max*k_max)/(2*sig_max)), SS + (1/k_max)*np.cos((k_max*k_max)/(2*sig_max)))
        
    def PlanPath(self, q_start, q_goal):
        q_s = q_start
        q_g = q_goal
        # Recalculate start and end point according to fraction along line
        # [q_s, startPath] = self.adjustPointForCurvature(q_start,1,self.sig_max)
        # [q_g, endPath] = self.adjustPointForCurvature(q_goal,-1,self.sig_max)
        
        # Pure Dubins Paths
        [path_RSR, L_RSR]  = self.RSR(q_s,q_g)
        [path_LSL, L_LSL]  = self.LSL(q_s,q_g)
        [path_RSL, L_RSL]  = self.RSL(q_s,q_g)
        [path_LSR, L_LSR]  = self.LSR(q_s,q_g)
        [path_RLR, L_RLR]  = self.RLR(q_s,q_g)
        [path_LRL, L_LRL]  = self.LRL(q_s,q_g)

        # # Reed-Shepp Paths
        # [path_RLrR, L_RLrR] = self.RLrR(q_s,q_g,self.k_max,self.sig_max,self.mu)
        # [path_LRrL, L_LRrL] = self.LRrL(q_s,q_g,self.k_max,self.sig_max,self.mu)

        # # Special Paths
        # [path_KRSR, L_KRSR] = self.KRSR(q_s,q_g,self.k_max,self.sig_max,self.mu)
        # [path_KLSL, L_KLSL] = self.KLSL(q_s,q_g,self.k_max,self.sig_max,self.mu)

        plt.plot(path_RSR[0,:],path_RSR[1,:],'-')
        plt.plot(path_LSL[0,:],path_LSL[1,:],'-')
        plt.plot(path_RSL[0,:],path_RSL[1,:],'-')
        plt.plot(path_LSR[0,:],path_LSR[1,:],'-')
        plt.plot(path_RLR[0,:],path_RLR[1,:],'-')
        plt.plot(path_LRL[0,:],path_LRL[1,:],'-')

        lengths = np.array([L_RSR, L_LSL, L_RSL, L_LSR, L_RLR, L_LRL])
        match np.argmin(lengths):
            case 0:
                return path_RSR, L_RSR
            case 1:
                return path_LSL, L_LSL
            case 2:
                return path_RSL, L_RSL
            case 3:
                return path_LSR, L_LSR
            case 4:
                return path_RLR, L_RLR
            case 5:
                return path_LRL, L_LRL
            case _:
                return np.array([0,0]), 1e12

    def RSR(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,self.k_max,self.sig_max)
        [C2, r2] = self.get_spiral_center(q_g,-1,self.k_max,self.sig_max)
        if util.eucdist2(C1,C2) < 2*r1*np.sin(self.mu):
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # Calculate Intermediate Points q1,q2
        th = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        q1 = np.array([C1[0] + r1*np.cos(th+(np.pi/2)-self.mu), C1[1] + r1*np.sin(th+(np.pi/2)-self.mu), th, 0])
        q2 = np.array([C2[0] + r2*np.cos(th+(np.pi/2)+self.mu), C2[1] + r2*np.sin(th+(np.pi/2)+self.mu), th, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_right(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = util.straight_path(q1,q2)
        [arc3, l3] = self.arc_right(C2, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        # path = arc1
        L = l1 + l2 + l3

        return path, L

    def LSL(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,-self.k_max,self.sig_max)
        [C2, r2] = self.get_spiral_center(q_g,-1,-self.k_max,self.sig_max)
        if util.eucdist2(C1,C2) < 2*r1*np.sin(self.mu):
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # Calculate Intermediate Points q1,q2
        th = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        q1 = np.array([C1[0] + r1*np.cos(th-(np.pi/2)+self.mu), C1[1] + r1*np.sin(th-(np.pi/2)+self.mu), th, 0])
        q2 = np.array([C2[0] + r2*np.cos(th-(np.pi/2)-self.mu), C2[1] + r2*np.sin(th-(np.pi/2)-self.mu), th, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_left(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = util.straight_path(q1,q2)
        [arc3, l3] = self.arc_left(C2, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        L = l1 + l2 + l3

        return path, L
    
    def RSL(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,self.k_max,self.sig_max)
        [C2, r2] = self.get_spiral_center(q_g,-1,-self.k_max,self.sig_max)
        if util.eucdist2(C1,C2) < 2*r1:
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # Calculate Intermediate Points q1,q2
        th = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        th_aug = np.arcsin(2*r1*np.cos(self.mu)/util.eucdist2(C1,C2))
        q1 = np.array([C1[0] + r1*np.cos(th+(np.pi/2-th_aug)-self.mu), C1[1] + r1*np.sin(th+(np.pi/2-th_aug)-self.mu), th-th_aug, 0])
        q2 = np.array([C2[0] + r2*np.cos(th-(np.pi/2+th_aug)-self.mu), C2[1] + r2*np.sin(th-(np.pi/2+th_aug)-self.mu), th-th_aug, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_right(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = util.straight_path(q1,q2)
        [arc3, l3] = self.arc_left(C2, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        # path = arc1
        L = l1 + l2 + l3

        return path, L
    
    def LSR(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,-self.k_max,self.sig_max)
        [C2, r2] = self.get_spiral_center(q_g,-1,self.k_max,self.sig_max)
        if util.eucdist2(C1,C2) < 2*r1:
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # Calculate Intermediate Points q1,q2
        th = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        th_aug = np.arcsin(2*r1*np.cos(self.mu)/util.eucdist2(C1,C2))
        q1 = np.array([C1[0] + r1*np.cos(th-(np.pi/2-th_aug)+self.mu), C1[1] + r1*np.sin(th-(np.pi/2-th_aug)+self.mu), th+th_aug, 0])
        q2 = np.array([C2[0] + r2*np.cos(th+(np.pi/2+th_aug)+self.mu), C2[1] + r2*np.sin(th+(np.pi/2+th_aug)+self.mu), th+th_aug, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_left(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = util.straight_path(q1,q2)
        [arc3, l3] = self.arc_right(C2, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        # path = arc1
        L = l1 + l2 + l3

        return path, L

    def LRL(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,-self.k_max,self.sig_max)
        [C3, r3] = self.get_spiral_center(q_g,-1,-self.k_max,self.sig_max)
        th = np.arctan2(C3[1]-C1[1],C3[0]-C1[0])
        d = util.eucdist2(C1,C3)
        if d > 4*r1:
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # get circle 2
        startAng = util.wrap_to_pi(np.arctan2(q_s[1]-C1[1],q_s[0]-C1[0]) - th) # find startpoint angle with respect to C1
        endAng = util.wrap_to_pi(np.arctan2(q_g[1]-C3[1],q_g[0]-C3[0]) - th) # find endpoint angle with respect to C1
        useShortSol = (startAng > np.arccos(d/(4*r1)) or startAng < -np.arccos(d/(4*r1))) and (endAng < np.pi-np.arccos(d/(4*r1)) or endAng > -np.pi+np.arccos(d/(4*r1)))      # boolean of whether we need to take the long path or short path
        if useShortSol:
            th_add = th - np.arccos(d/(4*r1))
        else:
            th_add = th + np.arccos(d/(4*r1))
        C2 = C1 + np.array([2*r1*np.cos(th_add), 2*r1*np.sin(th_add)])
        r2 = r1
        
        # Calculate Intermediate Points q1,q2
        th2 = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        th3 = np.arctan2(C3[1]-C2[1],C3[0]-C2[0])
        th_aug = np.arcsin(2*r1*np.cos(self.mu)/util.eucdist2(C1,C2))
        q1 = np.array([C1[0] + r1*np.cos(th2-(np.pi/2-th_aug)+self.mu), C1[1] + r1*np.sin(th2-(np.pi/2-th_aug)+self.mu), th2+th_aug, 0])
        th_aug = np.arcsin(2*r2*np.cos(self.mu)/util.eucdist2(C2,C3))
        q2 = np.array([C2[0] + r2*np.cos(th3+(np.pi/2-th_aug)-self.mu), C2[1] + r2*np.sin(th3+(np.pi/2-th_aug)-self.mu), th3-th_aug, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_left(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = self.arc_right(C2, q1, q2, self.k_max, self.sig_max, self.mu)
        [arc3, l3] = self.arc_left(C3, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        L = l1 + l2 + l3

        return path, L

    def RLR(self,q_s,q_g):
        # Calculate Spiral Centers
        [C1, r1] = self.get_spiral_center(q_s,1,self.k_max,self.sig_max)
        [C3, r3] = self.get_spiral_center(q_g,-1,self.k_max,self.sig_max)
        th = np.arctan2(C3[1]-C1[1],C3[0]-C1[0])
        d = util.eucdist2(C1,C3)
        if d > 4*r1:
            path = np.array([[0],[0]])
            L = 1e12
            return path, L
        
        # get circle 2
        startAng = util.wrap_to_pi(np.arctan2(q_s[1]-C1[1],q_s[0]-C1[0]) - th) # find startpoint angle with respect to C1
        endAng = util.wrap_to_pi(np.arctan2(q_g[1]-C3[1],q_g[0]-C3[0]) - th) # find endpoint angle with respect to C1
        useShortSol = (startAng > np.arccos(d/(4*r1)) or startAng < -np.arccos(d/(4*r1))) and (endAng < np.pi-np.arccos(d/(4*r1)) or endAng > -np.pi+np.arccos(d/(4*r1)))      # boolean of whether we need to take the long path or short path
        if useShortSol:
            th_add = th + np.arccos(d/(4*r1))
        else:
            th_add = th - np.arccos(d/(4*r1))
        C2 = C1 + np.array([2*r1*np.cos(th_add), 2*r1*np.sin(th_add)])
        r2 = r1
        
        # Calculate Intermediate Points q1,q2
        th2 = np.arctan2(C2[1]-C1[1],C2[0]-C1[0])
        th3 = np.arctan2(C3[1]-C2[1],C3[0]-C2[0])
        th_aug = np.arcsin(2*r1*np.cos(self.mu)/util.eucdist2(C1,C2))
        q1 = np.array([C1[0] + r1*np.cos(th2+(np.pi/2-th_aug)-self.mu), C1[1] + r1*np.sin(th2+(np.pi/2-th_aug)-self.mu), th2-th_aug, 0])
        th_aug = np.arcsin(2*r2*np.cos(self.mu)/util.eucdist2(C2,C3))
        q2 = np.array([C2[0] + r2*np.cos(th3-(np.pi/2-th_aug)+self.mu), C2[1] + r2*np.sin(th3-(np.pi/2-th_aug)+self.mu), th3+th_aug, 0])
    
        # Build Paths, return lengths
        [arc1, l1] = self.arc_right(C1, q_s, q1, self.k_max, self.sig_max, self.mu)
        [arc2, l2] = self.arc_left(C2, q1, q2, self.k_max, self.sig_max, self.mu)
        [arc3, l3] = self.arc_right(C3, q2, q_g, self.k_max, self.sig_max, self.mu)
        path = np.hstack((arc1, arc2, arc3))
        L = l1 + l2 + l3

        return path, L
    
    def get_spiral_center(self, q, dir, k, sig):
        side = -np.sign(k)
        k_max = np.abs(k)
        pt = util.euler_spiral_point(k_max/sig, sig)

        # find center of arc
        O_i = np.array([(pt[0] - (1/k_max)*np.sin((k_max*k_max)/(2*sig)))*dir,
                        (pt[1] + (1/k_max)*np.cos((k_max*k_max)/(2*sig)))*side])
        r = util.eucdist2(np.array([0,0]),O_i)

        # transform q_i to start frame
        cent = util.Tf(q[:2],q[2]).dot(np.array([O_i[0], O_i[1], 1]))
        return np.array([cent[0], cent[1]]), r
    
    def arc_right(self, center, q_start, q_end, k_max, sig_max, mu):
        r = util.eucdist2(q_start[:2], center)
        spiral = util.eulerspiral(-q_start[3] / sig_max, k_max / sig_max, sig_max)
        spiral[1, :] = -spiral[1, :]
        
        qi = util.Tf(q_start[:2], q_start[2]).dot(np.append(spiral[:, -1], 1))
        theta_start = np.arctan2(qi[1] - center[1], qi[0] - center[0])
        delta = (-(q_end[2] - q_start[2])) % (2*np.pi)
        delta_min = k_max**2 / sig_max
        
        if delta == 0:
            path, L = util.straight_path(q_start, q_start + np.array([2*r*np.sin(mu)*np.cos(q_start[2]), 2*r*np.sin(mu)*np.sin(q_start[2]),0,0]))
        elif ((0 < delta < delta_min) or k_max > np.sqrt(2 * sig_max * np.pi)) and (q_start[3] == 0 and q_end[3] == 0):
            new_sig = min(sig_max, util.get_new_sigma(delta, r, mu))
            
            spiral = util.eulerspiral(0, np.sqrt(delta / new_sig),new_sig)
            spiral[1, :] = -spiral[1, :]
            spiral = util.Tf(q_start[:2], q_start[2]).dot(np.vstack((spiral, np.ones(spiral.shape[1]))))
            spiral = spiral[:2, :]
            
            rev_spiral = util.eulerspiral(0, -np.sqrt(delta / new_sig), new_sig)
            rev_spiral = util.Tf(q_end[:2], q_end[2]).dot(np.vstack((rev_spiral, np.ones(rev_spiral.shape[1]))))
            rev_spiral = rev_spiral[:2, :]
            
            path = np.hstack((spiral, np.flip(rev_spiral, axis=1)))
            L = 2 * np.sqrt(delta / new_sig)
        else:
            theta_end = theta_start - (delta - delta_min + (delta < delta_min) * 2 * np.pi)
            theta_start, theta_end = util.ensure_valid_arc_right(theta_start, theta_end)
            
            arc = util.arc_from_angles(center, 1 / k_max, theta_start, theta_end, -np.pi / 20)
            
            spiral = util.Tf(q_start[:2], q_start[2]).dot(np.vstack((spiral, np.ones(spiral.shape[1]))))
            spiral = spiral[:2, :]
            
            rev_spiral = util.eulerspiral(-q_end[3] / sig_max, -k_max / sig_max, sig_max)
            rev_spiral = util.Tf(q_end[:2], q_end[2]).dot(np.vstack((rev_spiral, np.ones(rev_spiral.shape[1]))))
            rev_spiral = rev_spiral[:2, :]
            
            path = np.hstack((spiral, arc, np.flip(rev_spiral, axis=1)))
            L = (1 / k_max) * abs(theta_end - theta_start) + 2 * k_max / sig_max
        
        return path, L
    
    def arc_left(self, center, q_start, q_end, k_max, sig_max, mu):
        r = util.eucdist2(q_start[:2], center)
        spiral = util.eulerspiral(-q_start[3] / sig_max, k_max / sig_max, sig_max)
        qi = util.Tf(q_start[:2], q_start[2]).dot(np.append(spiral[:, -1], 1))
        theta_start = np.arctan2(qi[1] - center[1], qi[0] - center[0])
        delta = (q_end[2] - q_start[2]) % (2*np.pi)
        delta_min = k_max**2 / sig_max
        
        if delta == 0 and q_start[3] == 0 and q_end[3] == 0:
            path, L = util.straight_path(q_start, q_start + np.array([2*r*np.sin(mu)*np.cos(q_start[2]), 2*r*np.sin(mu)*np.sin(q_start[2]),0,0]))
        elif ((0 < delta < delta_min) or k_max > np.sqrt(2 * sig_max * np.pi)) and (q_start[3] == 0 and q_end[3] == 0):
            new_sig = min(sig_max, util.get_new_sigma(delta, r, mu))
            
            spiral = util.eulerspiral(0, np.sqrt(delta / new_sig), new_sig)
            spiral = util.Tf(q_start[:2], q_start[2]).dot(np.vstack((spiral, np.ones(spiral.shape[1]))))
            spiral = spiral[:2, :]
            
            rev_spiral = util.eulerspiral(0, -np.sqrt(delta / new_sig), new_sig)
            rev_spiral[1, :] = -rev_spiral[1, :]
            rev_spiral = util.Tf(q_end[:2], q_end[2]).dot(np.vstack((rev_spiral, np.ones(rev_spiral.shape[1]))))
            rev_spiral = rev_spiral[:2, :]
            
            path = np.hstack((spiral, np.flip(rev_spiral, axis=1)))
            L = 2 * np.sqrt(delta / new_sig)
        else:
            theta_end = theta_start + (delta - delta_min + (delta < 0) * 2 * np.pi)
            theta_start, theta_end = util.ensure_valid_arc_left(theta_start, theta_end)

            arc = util.arc_from_angles(center, 1 / k_max, theta_start, theta_end, np.pi / 20)
            
            spiral = util.Tf(q_start[:2], q_start[2]).dot(np.vstack((spiral, np.ones(spiral.shape[1]))))
            spiral = spiral[:2, :]
            
            rev_spiral = util.eulerspiral(-q_end[3] / sig_max, -k_max / sig_max, sig_max)
            rev_spiral[1, :] = -rev_spiral[1, :]
            rev_spiral = util.Tf(q_end[:2], q_end[2]).dot(np.vstack((rev_spiral, np.ones(rev_spiral.shape[1]))))
            rev_spiral = rev_spiral[:2, :]
            
            path = np.hstack((spiral, arc, np.flip(rev_spiral, axis=1)))
            L = (1 / k_max) * abs(theta_end - theta_start) + 2 * k_max / sig_max
        
        return path, L