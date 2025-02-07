import pandas as pd
import numpy as np
from cvxopt import matrix, solvers, sparse,spdiag,spmatrix
from collections import defaultdict
import os
# from .misc import flattenList

# TODO
# add try except and put errors/warnings to log
solvers.options['show_progress'] = False
dt = 1/30

def flattenList(nestedList):
 
    # check if list is empty
    if not(bool(nestedList)):
        return nestedList
     # to check instance of list is empty or not
    if isinstance(nestedList[0], list):
        # call function with sublist as argument
        return flattenList(*nestedList[:1]) + flattenList(nestedList[1:])
    # call function with sublist as argument
    return nestedList[:1] + flattenList(nestedList[1:])


def resample(car, dt=0.04, fillnan=False):
    # resample timestamps to 30hz, leave nans for missing data
    '''
    resample the original time-series to uniformly sampled time series in 1/dt Hz
    car: document
    leave empty slop as nan
    '''

    # Select time series only
    time_series_field = ["timestamp", "x_position", "y_position"]
    data = {key: car[key] for key in time_series_field}
    
    # Read to dataframe and resample
    df = pd.DataFrame(data, columns=data.keys()) 
    index = pd.to_timedelta(df["timestamp"], unit='s')
    df = df.set_index(index)
    df = df.drop(columns = "timestamp")
    
    freq = str(dt)+"S"
    df = df.resample(freq).mean() # leave nans
    if fillnan:
        df = df.interpolate(method='linear')
    df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9

    # resample to 25hz
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq()
    # df.index = df.index.values.astype('datetime64[ns]').astype('int64')*1e-9
    # df = df.interpolate(method='linear')
    # df=df.groupby(df.index.floor('0.04S')).mean().resample('0.04S').asfreq() # resample to 25Hz snaps to the closest integer
    
    # do not extrapolate for more than 1 sec
    first_valid_time = pd.Series.first_valid_index(df['x_position'])
    last_valid_time = pd.Series.last_valid_index(df['x_position'])
    first_time = max(min(car['timestamp']), first_valid_time-1)
    last_time = min(max(car['timestamp']), last_valid_time+1)
    df=df[first_time:last_time]
    
    car['x_position'] = df['x_position'].values
    car['y_position'] = df['y_position'].values
    car['timestamp'] = df.index.values
        
    return car






# ==================== CVX optimization for 2d dynamics ================== 
def opt1(car, lam3_x, lam3_y):
    '''
    1/M||z-Hx||_2^2 + \\lam3/N ||D3x||_2^2
    '''
    x = car["x_position"]
    y = car["y_position"]
    # x
    Q, p, H, N, M = _get_qp_opt1(x, lam3_x)
    sol=solvers.qp(P=Q, q=p)
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, N, M = _get_qp_opt1(y, lam3_y)
    sol=solvers.qp(P=Q, q=p)
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt1_l1(car, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    1/M||z-Hx||_2^2 + \\lam3/N ||D3x||_2^2 + \\lam1/M ||e||_1
    '''
    x = car["x_position"]
    y = car["y_position"]
    
    # x
    Q, p, H, G, h, N,M = _get_qp_opt1_l1(x, lam3_x, lam1_x)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, G, h, N,M = _get_qp_opt1_l1(y, lam3_y, lam1_y)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt2(car, lam2_x, lam2_y, lam3_x, lam3_y):
    '''
    1/M||z-Hx||_2^2 + \\lam2/N ||D2x||_2^2 + \\lam3/N ||D3x||_2^2
    '''
    x = car["x_position"]
    y = car["y_position"]
    # x
    Q, p, H, N, M = _get_qp_opt2(x, lam2_x, lam3_x)
    sol=solvers.qp(P=Q, q=p)
    xhat = sol["x"][:N]
    
    # y
    Q, p, H, N, M = _get_qp_opt2(y, lam2_y, lam3_y)
    sol=solvers.qp(P=Q, q=p)
    yhat = sol["x"][:N]
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    xhat_re = np.reshape(xhat, -1) # (N,)
    yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car

def opt2_l1(car, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    1/M||z-Hx||_2^2 + \\lam2/N ||D2x||_2^2 + \\lam3/N ||D3x||_2^2 + \\lam1/M ||e||_1
    "reconciliation_args":{
        "lam2_x": 0,
        "lam2_y": 0,
        "lam3_x": 1e-7,
        "lam3_y": 1e-7,
        "lam1_x": 0,
        "lam1_y": 0
    '''
    x = car["x_position"]
    y = car["y_position"]
    
    # x
    maxax = 99
    minvx = -1
    iter = 0
    max_iter = 10
    dir = car["direction"]
    
    while minvx < 0 and iter <= max_iter:
        Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(x, lam2_x, lam3_x, lam1_x)
        sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
        xhat = sol["x"][:N]
        D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
        vx = D1*xhat*dir
        minvx = min(vx)
        print("minvx ", minvx)
        
        ax = D2*xhat
        maxax = max(abs(ax))
        print("ax: {:.2f}, {:.2f}".format(min(ax), max(ax)))
        lam3_x += 1e-6
        iter += 1
        
    iter = 0   
    while maxax > 6 and iter <= max_iter:
        Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(x, lam2_x, lam3_x, lam1_x)
        sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
        xhat = sol["x"][:N]
        ax = D2*xhat
        maxax = max(abs(ax))
        print("ax: {:.2f}, {:.2f}".format(min(ax), max(ax)))
        lam2_x += 1e-6
        iter += 1
        
    cx_pre = np.nansum(np.abs(H*matrix(x)-H*xhat))/M + 1
    cx = cx_pre-1
    iter = 0
    while cx - cx_pre < 0 and iter <= max_iter:
        # print("iter, ", cx)
        lam1_x += 1e-6
        Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(x, lam2_x, lam3_x, lam1_x)
        sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
        xhat = sol["x"][:N]
        cx_pre = cx
        cx = np.nansum(np.abs(H*matrix(x)-H*xhat))/M
        iter += 1
    if sol["status"]!= "optimal":
        raise Exception("solver status is not optimal")
        
    # print("final")
    # print(f"lam2_x {lam2_x}, lam2_y {lam2_y}, lam3_x {lam3_x}, lam3_y {lam3_y},lam1_x {lam1_x}, lam1_y {lam1_y}")
    # print(sol["status"])
    # y
    Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(y, lam2_y, lam3_y, lam1_y)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    yhat = sol["x"][:N]
    if sol["status"]!= "optimal":
        raise Exception("solver status is not optimal")
    # ax = D2*xhat
    # print("ax: {:.2f}, {:.2f}".format(min(ax), max(ax)))
    # print("ay: {:.2f}, {:.2f}".format(min(D2*yhat), max(D2*yhat)))
    # print(sol["status"])
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat)
    car["y_position"] = list(yhat)
    
    # calculate residual
    # xhat_re = np.reshape(xhat, -1) # (N,)
    # yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    # cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    # cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    
    cx = np.nansum(np.abs(H*matrix(x)-H*xhat))/M
    cy = np.nansum(np.abs(H*matrix(y)-H*yhat))/M
    
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car 


def opt2_l1_constr(car, lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y):
    '''
    1/M||z-Hx||_2^2 + \\lam2/N ||D2x||_2^2 + \\lam3/N ||D3x||_2^2 + \\lam1/M ||e||_1
    s.t. D1x >=0, -10<=D2x<=10, -3 <=D3x<=3
    '''
    x = car["x_position"]
    y = car["y_position"]
    xmin = np.nanmin(x) # shift x to smaller numeral ranges
    x = x-xmin
    
    # x
    max_iter = 1
    dir = car["direction"]
    
    cx_pre = 999
    cx = cx_pre-1
    iter = 0
    while cx - cx_pre < 0 and iter <= max_iter:
        # print("iter, ", cx)
        lam1_x += 1e-3
        Q, p, H, G, h, N, M, D1,D2,D3 = _get_qp_opt2_l1_constr(x, dir, lam2_x, lam3_x, lam1_x)
        sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
        xhat = sol["x"][:N]
        cx_pre = cx
        cx = np.nansum(np.abs(H*matrix(x)-H*xhat))/M
        iter += 1
    
    # vmin = min(abs(D1*xhat))
    # amax = max(abs(D2*xhat))
    # jmax = max(abs(D3*xhat))
    # print("vmin: ", vmin, "amax: ", amax, "jamx: ",jmax)
    # if sol["status"]!= "optimal":
    #     raise Exception("solver status is not optimal")
    # print("x: ",sol["status"])
    
    # print("final")
    # print(f"lam2_x {lam2_x}, lam2_y {lam2_y}, lam3_x {lam3_x}, lam3_y {lam3_y},lam1_x {lam1_x}, lam1_y {lam1_y}")
    # print(sol["status"])
    # y
    Q, p, H, G, h, N, M, D2 = _get_qp_opt2_l1(y, lam2_y, lam3_y, lam1_y)
    sol=solvers.qp(P=Q, q=matrix(p) , G=G, h=matrix(h))
    yhat = sol["x"][:N]
    # if sol["status"]!= "optimal":
    #     raise Exception("solver status is not optimal")
    # print("y: ",sol["status"])
    
    car["timestamp"] = list(car["timestamp"])
    car["x_position"] = list(xhat+xmin)
    car["y_position"] = list(yhat)
    car["starting_x"] = car["x_position"][0]
    car["ending_x"] = car["x_position"][-1]
    
    # calculate residual
    # xhat_re = np.reshape(xhat, -1) # (N,)
    # yhat_re = np.reshape(yhat, -1) # (N,)
    # c1 = np.sqrt(np.nansum((x-xhat_re)**2)/M) # RMSE
    # cx = np.nansum(np.abs(x-xhat_re))/M # MAE
    # cy = np.nansum(np.abs(y-yhat_re))/M # MAE
    
    cx = np.nansum(np.abs(H*matrix(x)-H*xhat))/M
    cy = np.nansum(np.abs(H*matrix(y)-H*yhat))/M
    
    car["x_score"] = cx
    car["y_score"] = cy
    
    return car 





def _get_qp_opt1(x, lam3):
    '''
    rewrite opt1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    input:  x: data array with missing data
    return: Q, p, H, (G, h if l1)
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N <= 3:
        raise ZeroDivisionError
        
    # differentiation operator
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD = lam3 * D3.trans() * D3
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(N), range(N))
    H = I[idx,:]
    DD = lam3 * D3.trans() * D3
    HH = H.trans() * H

    Q = 2*(HH/M+DD/(N-3))
    p = -2*H.trans() * matrix(x)/M

    return Q, p, H, N, M


def _get_qp_opt1_l1(x, lam3, lam1):
    '''
    rewrite opt1_l1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N-3 <= 0:
        raise ZeroDivisionError
    # differentiation operator
    # D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    # D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD = lam3 * D3.trans() * D3
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))
    H = I[idx,:]
    HH = H.trans()*H

    Q = 2*sparse([[HH/M+DD/(N-3),H/M,-H/M], # first column of Q
                [H.trans()/M,IM/M, -H*H.trans()/M], 
                [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
    
    p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
    G = sparse([[H*O,H*O],[-IM,OM],[OM,-IM]])
    h = spmatrix([], [], [], (2*M,1))
    
    return Q, p, H, G, h, N, M



def _get_qp_opt2(x, lam2, lam3):
    '''
    rewrite opt2 to QP form:
    min 1/2 z^T Q x + p^T z + r   
    return: Q, p, H, M
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N <= 3:
        raise ZeroDivisionError
        
    # differentiation operator
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    
    DD2 = lam2 * D2.trans() * D2
    DD3 = lam3 * D3.trans() * D3
    # sol: xhat = (I+delta D'D)^(-1)x
    I = spmatrix(1.0, range(N), range(N))
    H = I[idx,:]
    HH = H.trans() * H

    Q = 2*(HH/M +DD2/(N-2) + DD3/(N-3))
    p = -2*H.trans() * matrix(x)/M

    return Q, p, H, N, M


def _get_qp_opt2_l1(x, lam2, lam3, lam1):
    '''
    rewrite opt2_l1 to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N-3 <= 0:
        raise ZeroDivisionError
    # differentiation operator
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    DD2 = lam2 * D2.trans() * D2
    DD3 = lam3 * D3.trans() * D3
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))
    H = I[idx,:]
    HH = H.trans()*H

    Q = 2*sparse([[HH/M+DD2/(N-2)+DD3/(N-3),H/M,-H/M], # first column of Q
                [H.trans()/M,IM/M, -H*H.trans()/M], 
                [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
    
    p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
    G = sparse([[H*O,H*O],[-IM,OM],[OM,-IM]])
    h = spmatrix([], [], [], (2*M,1))
    
    return Q, p, H, G, h, N, M, D2




def _get_qp_opt2_l1_constr(x, dir, lam2, lam3, lam1):
    '''
    rewrite opt2_l1_constr to QP form:
    min 1/2 z^T Q x + p^T z + r
    s.t. Gz <= h
    '''
    # get data
    N = len(x)
    
    # non-missing entries
    idx = [i.item() for i in np.argwhere(~np.isnan(x)).flatten()]
    x = x[idx]
    M = len(x)
    
    if M == 0 or N-3 <= 0:
        raise ZeroDivisionError
        
    # differentiation operator
    D1 = _blocdiag(matrix([-1,1],(1,2), tc="d"), N) * (1/dt)
    D2 = _blocdiag(matrix([1,-2,1],(1,3), tc="d"), N) * (1/dt**2)
    D3 = _blocdiag(matrix([-1,3,-3,1],(1,4), tc="d"), N) * (1/dt**3)
    DD2 = lam2 * D2.trans() * D2
    DD3 = lam3 * D3.trans() * D3
    
    # define matices
    I = spmatrix(1.0, range(N), range(N))
    IM = spmatrix(1.0, range(M), range(M))
    O = spmatrix([], [], [], (N,N))
    OM = spmatrix([], [], [], (M,M))
    H = I[idx,:]
    HH = H.trans()*H

    Q = 2*sparse([[HH/M+DD2/(N-2)+DD3/(N-3),H/M,-H/M], # first column of Q
                [H.trans()/M,IM/M, -H*H.trans()/M], 
                [-H.trans()/M,-H*H.trans()/M,IM/M]]) 
    
    p = 1/M * sparse([-2*H.trans()*matrix(x), -2*matrix(x)+lam1, 2*matrix(x)+lam1])
    B = spmatrix([], [], [], (5*N-11,M))
    G = sparse([[H*O,H*O,-dir*D1,D2,-D2,D3,-D3],[-IM,OM,B],[OM,-IM, B]])
    h1 = spmatrix([], [], [], (2*M+N-1,1))
    h2 = matrix(1.0, (2*N-4,1)) * 10 # acceleration constraint
    h3 = matrix(1.0, (2*N-6,1)) * 10 # jerk constraint
    h = sparse([h1,h2,h3])
    
    return Q, p, H, G, h, N, M, D1,D2,D3



def _blocdiag(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2]
    n indicates the total number of blocks (horizontally)
    """
    if not isinstance(X, spmatrix):
        X = sparse(X)
    a,b = X.size
    if n==b:
        return X
    else:
        mat = []
        for i in range(n-b+1):
            row = spmatrix([],[],[],(1,n))
            row[i:i+b]=matrix(X,(b,1))
            mat.append(row)
        return sparse(mat)







    
if __name__ == '__main__': 
    print("not implemented")


        
    
    
    
    
    
    