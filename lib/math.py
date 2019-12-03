from __future__ import division

import copy
import numpy as np
import scipy
import tensorflow as tf
import IPython

'''
robot rx,ry,rz : axis-angle notation (omega)
rotvec2euler in URscript: X-Y-Z euler angle
our euler : Z-Y-X

ex) given rotvec
eulerXYZ = rotvec2euler(rotvec)
eulerZXY = reverse(eulerXYZ)
euler_to_SO3(eulerZYX) == omega_to_SO3(rotvec)
'''

def np_sigmoid(x):
    sigm = 1./(1.+np.exp(-x))
    return sigm

def euler_to_SO3(euler, sined = False, angle_type = 'radian'):
    '''
    euler[ZYX ] : numpy array (3,)
    if sined is ture, the input is sin(euler) rather than euler
    '''
    so3_a=np.array([
        [ 0,-1, 0, 1, 0, 0, 0, 0, 0],
        [ 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]) 

    so3_b=np.array([
        [ 0, 0, 1, 0, 0, 0,-1, 0, 0],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])

    so3_y=np.array([
        [ 0, 0, 0, 0, 0,-1, 0, 1, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [ 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    if angle_type == 'radian':
        pass
    elif angle_type == 'degree':
        euler = euler*(np.pi/180)
    
    if sined is True:
        sin = np.expand_dims(euler, axis = -1)
        cos = np.sqrt(1-np.square(sin))
    else:
        sin = np.expand_dims(np.sin(euler), axis = -1)
        cos = np.expand_dims(np.cos(euler), axis = -1)
    t = np.concatenate([sin, cos, np.ones_like(sin)], axis = -1 )
    
    t_a = t[0,:]
    t_b = t[1,:]
    t_y = t[2,:]

    # euler : (alpha, beta, gamma)
    soa = np.matmul(t_a, so3_a) # Rz(alpha)
    sob = np.matmul(t_b, so3_b) # Ry(beta)
    soy = np.matmul(t_y, so3_y) # Rz(gamma)

    soa = np.reshape(soa, [3,3])
    sob = np.reshape(sob, [3,3])
    soy = np.reshape(soy, [3,3])
    so3 = np.matmul(soa, np.matmul(sob, soy))
    return so3

def omega_to_SO3(omega):
    omega_w = wedge(omega)
    R = scipy.linalg.expm(omega_w)
    return R

def vee(x):
    try:
        if x.shape == (3,3):
            #assert np.linalg.norm(x[1,0] +x[0,1]) < 1e-3
            #assert np.linalg.norm(x[2,0] +x[0,2]) < 1e-3
            #assert np.linalg.norm(x[1,2] +x[2,1]) < 1e-3
            y2 = x[1,0]
            y1 = x[0,2]
            y0 = x[2,1]
            return np.asarray([y0, y1, y2])
        elif x.shape == (4,4):
            w_hat = x[0:3,0:3]
            w = vee(w_hat)
            v = x[0:3,3]
            assert np.linalg.norm(x[3,:]-np.array([0,0,0,0])) < 1e-5
            xi = np.concatenate([v,w], 0)
            return xi
    except:
        print('assertion error')
        print('input:'+str(x))

def wedge(x):
    assert x.shape == (3,)
    y = [[0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]]    
    return np.asarray(y)

def SO3_to_so3(R):
    thetha = np.arccos((np.trace(R) - 1)/2 )
    ## (Warning!!) if theta = ||w||  > 3.1415; then algorithm fails
    #thetha = (thetha*1e10)%(np.pi*1e10)/(1e10)
    if thetha == 0:
        w = np.zeros(3)
    else:   
        w = thetha * (1./(2.*np.sin(thetha)))*np.asarray([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    return w

def so3_to_SO3(omega, theta):
    assert omega.shape == (3,)
    omega_w = wedge(omega)
    omega_norm = np.linalg.norm(omega)

    y0 = np.eye(3)
    y1 = (omega_w/omega_norm) *np.sin(omega_norm*theta)
    y2 = (np.matmul(omega_w,omega_w)/(omega_norm**2))*(1-np.cos(omega_norm*theta))
    y = y0+y1+y2
    return y

def RT_to_SE3(R,T):
    RT = np.concatenate([R,np.expand_dims(T, axis = 1)], axis = 1)
    bottom = np.array([[0,0,0,1]])
    return np.concatenate([RT, bottom], axis = 0)

def se3_to_SE3(xi):
    '''
    xi = [v,w]
    R = e^wedge(w)
    T = 1/|w| * [(I-R)*(w x v) + w*w^T*v]
    http://karnikram.info/blog/lie/
    '''
    v = np.copy(xi[0:3])
    w = np.copy(xi[3:6])
    
    if (np.linalg.norm(w)<1e-5):
        R = np.eye(3)
        T = v
        SE3 = RT_to_SE3(R,T)
    else:
        ## (Warning!!) if theta = ||w||  > 3.1415; then algorithm fails
        w_norm = np.sqrt( w[0]*w[0]+w[1]*w[1]+w[2]*w[2]) 
        new_w_norm = (w_norm*1e10)%(np.pi*1e10)/(1e10) 
        ######################################3
        new_w = w #(new_w_norm/w_norm) * w  # w ############################################
        
        w_hat = wedge(new_w)
        #R = scipy.linalg.expm(w_hat)
        w_norm = np.sqrt( new_w[0]*new_w[0]+new_w[1]*new_w[1]+new_w[2]*new_w[2])
        #T = (1/w_norm)*np.matmul( np.eye(3)-R, np.cross(w,v))+w_norm*v
        R = np.eye(3) + (w_hat/w_norm) * np.sin(w_norm) + (np.matmul(w_hat,w_hat)/(w_norm*w_norm)) * (1 - np.cos(w_norm))
        T = np.matmul(np.eye(3) + (((1 - np.cos(w_norm)) / (w_norm**2)) * w_hat) + (((w_norm - np.sin(w_norm)) / (w_norm**3)) * np.matmul(w_hat,w_hat)),v)
        SE3 = RT_to_SE3(R,T)
    return SE3

def SE3_to_se3(SE3):
    R = SE3[0:3,0:3]
    T = SE3[0:3,3]
    thetha = np.arccos((np.trace(R) - 1)/2 )
    ## (Warning!!) if theta = ||w||  > 3.1415; then algorithm fails
    ############################################
    #thetha = (thetha*1e10)%(np.pi*1e10)/(1e10) 
    
    if thetha == 0:
        v = T
        w = np.zeros(3)
    else:   
        w = thetha * (1./(2.*np.sin(thetha)))*np.asarray([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        w_hat = wedge(w)
        v = np.matmul(np.eye(3) - (1/2 * (w_hat)) + (((1/(thetha * thetha)) * (1 - ((thetha * np.sin(thetha)) / (2*(1 - np.cos(thetha)))))) * np.matmul(w_hat, w_hat)),T)
    return np.concatenate([v,w],0)

def SE3_to_RT(SE3):
    R = SE3[0:3,0:3]
    T = SE3[0:3,3]
    assert (SE3[3,:] == np.array([0,0,0,1])).all()
    return R,T

def to_homo(x):
    ''' 
    coordinate to homogeneous form
    x: numpy array with shape (3,) or (3,1)
    '''
    ones = np.ones_like(x)
    concat = np.concatenate([x,ones], axis = 0)
    return concat[0:4]

def un_homo(x):
    '''
    homogeneous coorinate to default coordinate
    x: numpy array with shape (4,) 
    '''
    return x[0:3]

def get_SE3_between_se3s(se30, se31):
    SE30 = se3_to_SE3(se30)
    SE31 = se3_to_SE3(se31)
    SE3 = np.matmul(SE31, inv_SE3(SE30))
    return SE3
    
def inv_SE3(SE3):
    R = SE3[0:3,0:3]
    T = SE3[0:3,3]

    R_inv = np.transpose(R)
    T_inv = - np.matmul(R_inv, T)
    SE3_inv = RT_to_SE3(R_inv,T_inv)
    return SE3_inv

def se3_forward_kinematics(se3_state, se3_act):
    SE3_state = se3_to_SE3(se3_state)
    SE3_act = se3_to_SE3(se3_act)
    SE3_next_state = np.matmul(SE3_act, SE3_state)
    se3_next_state = SE3_to_se3(SE3_next_state)
    return se3_next_state

def quaternion_to_R(quater):
    qx = quater[0]
    qy = quater[1]
    qz = quater[2]
    qw = quater[3]
    
    r00 = 1 -2*qy**2-2*qz**2
    r01 = 2*qx*qy-2*qz*qw
    r02 = 2*qx*qz + 2*qy*qw
    r10 = 2*qx*qy + 2*qz*qw
    r11 = 1-2*qx**2-2*qz**2
    r12 = 2*qy*qz - 2*qx*qw
    r20 = 2*qx*qz-2*qy*qw
    r21 = 2*qy*qz+2*qx*qw
    r22 = 1-2*qx**2-2*qy**2

    R = np.asarray([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])
    return R

def quaternion_to_SE3(quater):
    x = quater[0]
    y = quater[1]
    z = quater[2]
    qx = quater[3]
    qy = quater[4]
    qz = quater[5]
    qw = quater[6]
    R = quaternion_to_R(np.asarray([qx,qy,qz,qw]))
    T = np.asarray([x,y,z])
    SE3 = RT_to_SE3(R,T)
    return SE3

def quaternion_to_se3(quater):
    x = quater[0]
    y = quater[1]
    z = quater[2]
    qx = quater[3]
    qy = quater[4]
    qz = quater[5]
    qw = quater[6]

    R = quaternion_to_R(np.asarray([qx,qy,qz,qw]))
    T = np.asarray([x,y,z])
    SE3 = RT_to_SE3(R,T)
    se3 = SE3_to_se3(SE3)
    return se3
    
'''
tensor in tensor out function
'''

def tf_reduce_sum_nan(value, axis = None ):
    '''
    Reduce_sum in tf while considering NaN as zero
    '''
    nan_to_zero = tf.where(tf.is_nan(value),tf.zeros_like(value), value)
    return tf.reduce_sum(nan_to_zero, axis)

def tf_reduce_mean_nan(value, axis = None):
    '''
    Reduce_mean in tf while considering NaN as zero
    '''
    nan_to_zero = tf.where(tf.is_nan(value),tf.zeros_like(value), value)
    return tf.reduce_mean(nan_to_zero, axis)

def tf_repeat_vector(value, repeat_time):
    extended = tf.expand_dims(value, axis = 1)
    tiled = tf.tile(extended, [1,repeat_time, 1])
    return tiled

def tf_multivariate_prob(x, mean, var):
    '''
    x : [N,k]
    mean: [N,k]
    var: [N,k]
    '''
    k = tf.to_float(tf.shape(mean)[1])
    Sigma = tf.linalg.diag(var)
    Z = ((2*np.pi)**(k)*(tf.abs(tf.linalg.det(Sigma))))**(1./2.)
    Sigma_inv = tf.linalg.inv(Sigma)
    x_m = x-mean
    x_m_T = tf.transpose(tf.expand_dims(x_m,-1), [0,2,1])
    y = tf.matmul(tf.matmul(x_m_T, Sigma_inv),tf.expand_dims(x_m,-1))
    y = tf.squeeze(y,[1,2])
    prob = (1./Z)*(tf.exp(-1/2*y))
    return prob

def tf_wedge(w):
    '''
    input : w [-1,3]
    output: w_wedge [-1,3,3]
    '''
    
    w0 = tf.expand_dims(w[:,0],axis = 1) #[-1,1]
    w1 = tf.expand_dims(w[:,1],axis = 1) #[-1,1]
    w2 = tf.expand_dims(w[:,2],axis = 1) #[-1,1]
    
    row0 = tf.concat([tf.zeros_like(w0), -w2, w1], axis = 1) #[-1,3]
    row1 = tf.concat([w2, tf.zeros_like(w1), -w0], axis = 1) #[-1,3]
    row2 = tf.concat([-w1, w0, tf.zeros_like(w2)], axis = 1) #[-1,3]
    
    row0 = tf.expand_dims(row0, axis = 1) #[-1,1,3]
    row1 = tf.expand_dims(row1, axis = 1) #[-1,1,3]
    row2 = tf.expand_dims(row2, axis = 1) #[-1,1,3]
    
    output = tf.concat([row0, row1, row2], axis = 1)
    return output
    
def tf_vee(w_hat):
    '''
    input : w_hat [-1,3,3]
    output: w [-1,3]
    '''
    y2 = w_hat[:,1,0] # [N,]
    y1 = w_hat[:,0,2] # [N,]
    y0 = w_hat[:,2,1] # [N,]
    
    y2 = tf.expand_dims(y2, axis = 1) #[N,1]
    y1 = tf.expand_dims(y1, axis = 1) #[N,1]
    y0 = tf.expand_dims(y0, axis = 1) #[N,1]
    w = tf.concat([y0,y1,y2],axis = 1) #[N,3]
    return w
    
def tf_RT_to_SE3(R,T):
    '''
    input: R [-1,3,3]
           T [-1,3]
    '''
    RT = tf.concat([R, tf.expand_dims(T,2)], axis = 2) # [-1, 3, 4]
    zeros = tf.zeros_like(T) #[-1,3]
    ones = tf.expand_dims(tf.ones_like(T[:,0]), axis = 1) #[-1,1]
    bottom = tf.concat([zeros, ones], axis = 1) #[-1,4]
    bottom = tf.expand_dims(bottom, axis = 1) # [-1, 1, 4]
    SE3 = tf.concat([RT, bottom], axis = 1)
    return SE3
    
def tf_se3_to_SE3(se3):
    '''
    input: se3 [-1, 6 ]
    output: SE3 [-1, 4, 4]
    '''
    v = se3[:,0:3]
    w_old = se3[:,3:6] 
    hope = 1e-5

    batch_size = tf.shape(v)[0]
    w_norm_old = tf.sqrt(tf.reduce_sum((tf.square(w_old)), axis = 1, keepdims = True)) 
    
    tf_pi = tf.constant(np.pi)
    new_w_norm = tf.floormod(w_norm_old, tf_pi)
    w = (new_w_norm/(w_norm_old+hope))*w_old

    w_norm = tf.sqrt(tf.reduce_sum((tf.square(w)), axis = 1, keepdims = True)) 
    isZero = tf.expand_dims(tf_minmax_01(w_norm, 1e-3, 1e3),2)
    
    w_norm = tf.expand_dims(w_norm, 2)
    w_hat = tf_wedge(w)
    R = tf.eye(3) + (w_hat/(w_norm+hope)) * tf.sin(w_norm) + (tf.matmul(w_hat,w_hat)/(w_norm*w_norm+hope)) * (1 - tf.cos(w_norm))
    R = tf.multiply(isZero,R)+tf.multiply(1-isZero, tf.eye(3, batch_shape = [batch_size]))
    
    T = tf.matmul(tf.eye(3) + (((1 - tf.cos(w_norm)) / (tf.pow(w_norm,2)+hope) ) * w_hat) + (((w_norm - tf.sin(w_norm)) / (tf.pow(w_norm,3)+hope)) * tf.matmul(w_hat,w_hat)),tf.expand_dims(v,2))
    T = tf.multiply(isZero,T) + tf.multiply(1-isZero, tf.expand_dims(v,2))
    T = T[:,:,0]
    
    SE3 = tf_RT_to_SE3(R, T)
    return SE3

def tf_SE3_to_se3(SE3):
    '''
    input : SE3 [-1, 4, 4]
    output: se3 [-1,6]
    '''
    # do not use tf.linalg.logm() because it doesn't have gradient graph

    R = SE3[:,0:3,0:3] # [N,3,3]
    T = SE3[:,0:3,3] # [N, 3]
    thetha = tf.expand_dims(tf.acos((tf.trace(R) - 1)/2),1)
    thetha = tf.expand_dims(thetha,2)+1e-5

    R = tf.to_complex64(R)
    w_hat = tf.linalg.logm(R) # [N,3,3]
    w_hat = tf.to_float(w_hat)
    w = tf_vee(w_hat) # [N,3]

    v = tf.matmul(tf.eye(3) - (1/2 * (w_hat)) + (((1/(thetha * thetha)) * (1 - ((thetha * tf.sin(thetha)) / (2*(1 - tf.cos(thetha)))))) * tf.matmul(w_hat, w_hat)), tf.expand_dims(T,2))
    v = v[:,:,0]
    se3 = tf.concat([v,w], axis = 1)
    return se3

def tf_inv_SE3(SE3):
    '''
    input: 
    '''
    R = SE3[:,0:3,0:3]
    T = SE3[:,0:3,3]

    R_inv = tf.transpose(R, [0,2,1])
    T_inv = -tf.matmul(R_inv, tf.expand_dims(T,2))[:,:,0]
    SE3_inv = tf_RT_to_SE3(R_inv,T_inv)
    return SE3_inv


def tf_minmax_quadratic(x,min,max):
    min_relu = tf.square(tf.nn.relu(-x+min))
    max_relu = tf.square(tf.nn.relu(x-max))
    return min_relu+max_relu


def tf_minmax_01(x,minv,maxv):
    '''
    x : [None,1]
    output : [None,1]
    '''
    return 0.5* tf.abs(tf.sign(x+minv)+tf.sign(x+maxv))
