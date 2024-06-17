import matplotlib.pyplot as plt
import scipy.io as scio
import math
import numpy as np
import cmath
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import csiread
from csilib import getcsi


# 下面是原始的savgol_filter函数代码，包括savgol_coeffs和其他相关函数...
# 请在此处插入原始函数的其余部分

def process(csi_data1, csi_data2, samp_rate, loc_ini,tx,rx):
    lambda_ = 3e8 / 5.18e9
    time_length = csi_data1.shape[1]
    time = np.zeros(time_length)
    for i in range(0, time_length):
        time[i] = 1 / samp_rate * i
    initpoint = loc_ini
    skip_silence = 1
    # get Doppler frequency shift (in Hz)
    speed1, score1, slope, duslope, mrange, ne1 = mimo2speed(csi_data1, samp_rate)
    speed2, score2, slope, duslope, mrange, ne2 = mimo2speed(csi_data2, samp_rate)
    #plotPolarColor(ne1,'CSI Quotient Slope RX 1')
    #plotPolarColor(ne2,'CSI Quotient Slope RX 2')
    # [speed3, score3] = mimo2speed(pc3, samp_rate)
    # convert to speed (in m/s)
    movespeed1 = speed1 * lambda_
    movespeed2 = speed2 * lambda_
    new_length = min(movespeed2.size, movespeed1.size)
    # 调整数组长度
    movespeed2 = np.resize(movespeed2, new_length)
    movespeed1 = np.resize(movespeed1, new_length)
    new_length = min(score2.size, score1.size)
    # 调整数组长度
    score2 = np.resize(score2, new_length)
    score1 = np.resize(score1, new_length)
    # movespeed3 = speed3 * lambda_
    speed = np.vstack((movespeed1, movespeed2)).T
    score = np.vstack((score1, score2)).T
    # for data plot purpose
    index = np.argmin(score, axis=1)
    len_ = score.shape[0]
    sel1 = np.zeros((len_, 1), dtype=bool)
    sel2 = np.zeros((len_, 1), dtype=bool)
    sel3 = np.zeros((len_, 1), dtype=bool)
    for i in range(len_):
        if np.ceil(index[i]) == 1:
            sel1[i] = True
        if np.ceil(index[i]) == 2:
            sel2[i] = True
        if np.ceil(index[i]) == 3:
            sel3[i] = True
    # calculate skip
    skip = math.floor(skip_silence * samp_rate)
    if skip < 1:
        skip = 1
    # slice speed and score arrays
    speed = speed[1:-1, :]
    score = score[1:-1, :]
    # print(speed)
    # calculate trajectory
    loc = trajectory_by_doppler_v3(speed, score, np.diff(time), initpoint, tx, rx, 10)

    return loc

    # overlap groundtruth
    # plt.plot(np.real(groundtruth), np.imag(groundtruth), '--', color='red')


def mimo2speed(rx, samp_rate):
    # get CSI ratio from raw CSI input
    csiq = getAntMIMO(rx, 1, 2)

    speed_high, score_high, agree_high, slope, duslope, mrange, ne = windowd_speed(csiq, samp_rate, 14)
    speed_low, score_low, agree_low = windowd_speed(csiq, samp_rate, 14)[:3]
    speed = np.vstack((speed_high, speed_low))
    score_in = -np.vstack((agree_high, agree_low))

    score, idx = np.max(score_in, axis=0), np.argmax(score_in, axis=0)
    dopplerspeed = np.zeros_like(speed_high)

    for i in range(len(score)):
        dopplerspeed[i] = speed[idx[i], i]

    return dopplerspeed, score, slope, duslope, mrange, ne



def getAntMIMO(rx, ant1, ant2):
    length = rx.shape[1]
    rx1 = rx[(1 - 1) * 30:1 * 30, 0:length]
    rx2 = rx[(2 - 1) * 30:2 * 30, 0:length]
    rx3 = rx[(3 - 1) * 30:3 * 30, 0:length]
    # avoid inf and nan
    r1 = m_removecomplexzero(rx1)
    r2 = m_removecomplexzero(rx2)
    r3 = m_removecomplexzero(rx3)
    r1avg = np.mean(np.abs(r1))
    r2avg = np.mean(np.abs(r2))
    r3avg = np.mean(np.abs(r3))
    level = [r1avg, r2avg, r3avg]


    # r1avg = np.nanmean(np.abs(r1))
    # r2avg = np.nanmean(np.abs(r2))
    # r3avg = np.nanmean(np.abs(r3))
    # level = [r1avg, r2avg, r3avg]

    if level[ant1 - 1] < level[ant2 - 1]:
        maxIndex = ant1
        minIndex = ant2
        ret = np.divide(r1 , r2)
    else:
        maxIndex = ant2
        minIndex = ant1
        ret = np.divide(r2 , r1)

    print('level = ', level)
    print('r', minIndex, '/r', maxIndex, ', r', minIndex, '/r', maxIndex)
    #if minIndex == 1:
    #    ret = rx1 / rx3
    #else:
    #    ret = rx3 / rx1
    #denominator = maxIndex
    
    
    # fill nan position in the stream

    # append nan endings with the last non-nan data

    return ret


def m_removecomplexzero(r):
    """
    Remove NaN and Inf in data input. Replace them with interpolated data.

    Parameters:
    r (np.ndarray): Input data

    Returns:
    np.ndarray: Interpolated data with NaN and Inf replaced by interpolated data
    """
    length = r.shape[1]

    mask = (r == 0 + 0j)
    r[mask] = 0.1 + 0.1j
    return r

def interpolateComplexInf(inputArray):
    # 确保输入是一个列向量
    if not np.isscalar(inputArray) and len(inputArray.shape) > 1:
        raise ValueError('Input must be a vector.')
    
    # 分别找到实部和虚部中 Inf 或 -Inf 的索引
    infIndicesReal = np.where(np.isinf(inputArray.real))[0]
    infIndicesImag = np.where(np.isinf(inputArray.imag))[0]
    
    # 获取数组的有效索引（非 Inf 和非 NaN），分别对实部和虚部
    validIndicesReal = np.where(~np.isinf(inputArray.real) & ~np.isnan(inputArray.real))[0]
    validIndicesImag = np.where(~np.isinf(inputArray.imag) & ~np.isnan(inputArray.imag))[0]
    
    # 检查是否有足够的非 Inf/NaN 点进行插值
    if len(validIndicesReal) < 2 or len(validIndicesImag) < 2:
        raise ValueError('Not enough non-Inf and non-NaN points in either the real or imaginary parts to perform interpolation.')
    
    # 创建输出数组，初始时等于输入数组
    output = np.copy(inputArray)
    
    # 分别对实部和虚部的 Inf 位置进行插值
    if infIndicesReal.size > 0:
        interpFuncReal = interp1d(validIndicesReal, inputArray.real[validIndicesReal], kind='linear', fill_value='extrapolate')
        interpolatedValuesReal = interpFuncReal(infIndicesReal)
        output[infIndicesReal] = interpolatedValuesReal + 1j * output[infIndicesReal].imag
    
    if infIndicesImag.size > 0:
        interpFuncImag = interp1d(validIndicesImag, inputArray.imag[validIndicesImag], kind='linear', fill_value='extrapolate')
        interpolatedValuesImag = interpFuncImag(infIndicesImag)
        output[infIndicesImag] = output[infIndicesImag].real + 1j * interpolatedValuesImag
    
    return output
def windowd_speed(csiq, samp_rate, window_size):
    timeslot = 6 / window_size  # 0.05 before modification
    wsize = int(np.floor(timeslot * samp_rate))

    window_size = window_size * samp_rate / 400

    ncsi = np.mean(csiq, axis=0)
    ncsi = interpolateComplexInf(ncsi)
    len_ncsi = len(ncsi)

    for i in range(3):
        ncsi = m_denoise_w(ncsi, samp_rate / window_size)

    skip = 2  # for smoother doppler speed extraction
    ns = np.hstack((ncsi[0:skip], ncsi))
    ne = np.hstack((ncsi, ncsi[len_ncsi - skip:len_ncsi]))

    csislope = ne - ns  # get tangent on the circle
    csislope = csislope[skip:skip + len_ncsi]  # restore the same amount of samples
    slope = np.angle(csislope)  # calculate phase of tangent
    slope[0] = slope[1]
    slope[-1] = slope[-2]

    uslope = np.unwrap(slope)  # unwarp phase by 2pi
    uslope = uslope - uslope[0]  # phase change start from 0

    # mark unsure segmentss'lo
    duslope = np.diff(uslope, axis=0)  # doppler speed
    duslope = np.hstack((duslope[0], duslope))
    mslope_min = movmin(duslope, wsize)
    mslope_max = movmax(duslope, wsize)
    mslope_range = mslope_max - mslope_min

    du = movmean(duslope, wsize)

    agree = np.sign(du)

    speed_dir = np.sign(agree)

    agree = np.abs(agree)

    # min duslope variation
    min_score = np.nanargmin(mslope_range, axis=0)
    mrange = np.nanmin(mslope_range, axis=0)
    dopplerspeed = np.zeros(len_ncsi)
    for i in range(len_ncsi):
        dopplerspeed[i] = -np.abs(duslope[i]) * speed_dir[i]

    xnan = np.isnan(dopplerspeed)
    x = np.arange(len_ncsi)
    f = interp1d(x[~xnan], dopplerspeed[~xnan], kind='linear', fill_value='extrapolate')
    dopplerspeed = f(x)
    dopplerspeed = dopplerspeed.T
    duslope = dopplerspeed
    slope = -np.cumsum(dopplerspeed, axis=0)
    dopplerspeed = dopplerspeed * samp_rate / (2 * np.pi)
    dopplerspeed = movmean(dopplerspeed, wsize / 2)

    score = -mrange / window_size

    agree = agree / score

    return dopplerspeed.flatten(), score.flatten(), agree, slope, duslope, mrange, ne


def movmin(datas, k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k / 2))

    for i in range(len(datas)):
        if i < int(np.ceil(k / 2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k / 2)):
            end_pt = len(datas)
        result[i] = np.min(datas[start_pt:end_pt])
        start_pt += 1
        end_pt += 1
    return result


def movmax(datas, k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k / 2))

    for i in range(len(datas)):
        if i < int(np.ceil(k / 2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k / 2)):
            end_pt = len(datas)
        result[i] = np.max(datas[start_pt:end_pt])
        start_pt += 1
        end_pt += 1
    return result


def movmean(datas, k):
    result = np.empty_like(datas)
    start_pt = 0
    end_pt = int(np.ceil(k / 2))

    for i in range(len(datas)):
        if i < int(np.ceil(k / 2)):
            start_pt = 0
        if i > len(datas) - int(np.ceil(k / 2)):
            end_pt = len(datas)
        result[i] = np.mean(datas[start_pt:end_pt])
        start_pt += 1
        end_pt += 1
    return result


def m_denoise_w(x, winsize):
    """
    Signal denoise using Savitzky-Golay smoothing.
    
    Parameters:
    x : array_like
        The input signal.
    winsize : int
        The length of the window, should be a positive odd integer.
    
    Returns:
    ret : ndarray
        The denoised signal.
    """
    
    # Check the length of the input signal
    len_x = len(x)
    if len_x < 30:
        return x
    
    # Order of polynomial fit
    N = 2
    
    # Window size, make sure it's an odd integer and does not exceed the signal length
    F = int(np.floor(winsize))
    if F > len_x:
        F = len_x - 2
    if F % 2 == 0:
        F += 1  # Make it odd
    
    # Apply Savitzky-Golay filter
    ret = savgol_filter(x, F, N,mode='mirror')
    
    return ret
def trajectory_by_doppler_v3(move_speed, score, deltaT, initPos, Tx, Rx, window):
    # get trajectory from Doppler speeds
    # input: move_speed Doppler speeds of Rx
    #        score score of Rx
    #        deltaT sample interval time
    #        initPos initial position of person
    #        Tx position of Tx (in complex value)
    #        Rx position of Rx (in complex vector)
    #        window input/output sample ratio
    # output: Loc location sequence (in complex vector)
    #         LocT time sequence
    #         v human motion speed (in complex vector)
    length = len(move_speed)
    seg = int(length / window)

    v = np.zeros((seg, 1), dtype=np.complex128)
    Loc = np.zeros((seg, 1), dtype=np.complex128)
    LocT = np.zeros((seg, 1), dtype=np.complex128)
    #deltaT = np.hstack(([0], deltaT))
    curPos = initPos
    min_score = np.argmin(score, axis=1)
    loc_list=[]
    for i in range(seg):
        s = window * i + 1
        e = s + window - 1
        drop_antenna = np.round(np.mean(min_score[s:e]))
        ant = []
        for j in range(3):
            if j != drop_antenna:
                ant.append(j)
        ant = [0, 1]
        ref_ang = np.angle(curPos - Tx)
        normal_dir1 = getnormaldir(Tx, Rx[ant[0]], curPos)
        normal_dir2 = getnormaldir(Tx, Rx[ant[1]], curPos)

        dopplermove_step = [np.sum(move_speed[s:e, ant[0]] * deltaT[s:e]),
                            np.sum(move_speed[s:e, ant[1]] * deltaT[s:e])]

        proj = [np.cos(normal_dir1 - ref_ang), np.cos(normal_dir2 - ref_ang)]

        normal_step = np.zeros(2)
        normal_step[0] = dopplermove_step[0] / proj[0] / 2
        normal_step[1] = dopplermove_step[1] / proj[1] / 2

        newPos = gethumanspeed(curPos, [normal_dir1, normal_dir2], normal_step)
        #         if (abs(newPos) != abs(1+1j)):
        #             print('stop here')
        Loc[i] = curPos
        loc_list.append(curPos)
        v[i] = (newPos - curPos) / deltaT[i]
        #print(f"The velocity is{v[i]/window}")
        if v[i]<100:
            curPos = (newPos-curPos)*1+curPos
        LocT[i] = np.sum(deltaT[s:e])
    LocT = np.cumsum(LocT)
    return loc_list

def getnormaldir(tx, rx, pos):
    # this function return the normal vector based on the location of tx, rx,
    # and human target
    p1 = pos - tx
    p2 = pos - rx
    ang1 = np.angle(p1)
    ang2 = np.angle(p2)
    normal_dir = (ang1 + ang2) / 2
    pos_x = pos + 0.01 * np.exp(1j * normal_dir)
    # deal with pi/-pi ambiguity
    if abs(tx - pos_x) + abs(rx - pos_x) < abs(p1) + abs(p2):
        normal_dir = normal_dir + np.pi
    return normal_dir

def gethumanspeed(pos, doppler_dir, doppler_move):
    if sum(abs(doppler_move)) == 0:
        return pos
    pos_x = pos + doppler_move[0] * cmath.exp(1j * doppler_dir[0])
    pos_y = pos + doppler_move[1] * cmath.exp(1j * doppler_dir[1])

    t1 = -1 / cmath.tan(doppler_dir[0])
    t2 = -1 / cmath.tan(doppler_dir[1])

    x = (pos_x.imag - pos_y.imag + t2 * pos_y.real - t1 * pos_x.real) / (t2 - t1)
    y = t1 * x + pos_x.imag - t1 * pos_x.real
    newpos = x + y * 1j
    return newpos


def matrixreshape(original_array):
    #change the csi shape to 90*time
    length = original_array.shape[0]
    reshaped_array = np.zeros((length, 90), dtype=np.complex128)
    for i in range(3):
        start_idx = i * 30
        end_idx = (i + 1) * 30
        reshaped_array[:, start_idx:end_idx] = original_array[ :,i, :]
    return reshaped_array.T
if __name__ == '__main__':


# # Linux 802.11n CSI Tool
#     csifile = "line-1.dat"
#     # csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=1)
#     # csidata.read()
#     csi = getcsi(csifile)  
#     print(csi.shape)
#     data1 = matrixreshape(np.squeeze(csi))

#     csifile = "line-2.dat"
#     # csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=1)
#     # csidata.read()
#     csi = getcsi(csifile)   
#     data2 = matrixreshape(np.squeeze(csi))


#     # dataFile1 = 's-1'
#     # dataread1 = scio.loadmat(dataFile1)
#     # data1 = dataread1['csi_data'].T

#     # dataFile2 = 'shun2-2'
#     # dataread2 = scio.loadmat(dataFile2)
#     # data2 = dataread2['csi_data'].T
#     # print(data1.shape[1])

#     # #data1 = data1[:,:2000]
#     # #data2 = data2[:, :2000]
#     # print(data1.shape)
#     tx = 0 + 0j
#     r1 = 0 - 5.6j
#     r2 = 2.7 + 0j
#     rx = [r1, r2]
#     loc_ini = 2.2 -6.25j
  

#     loc=process(data1,data2,1000,loc_ini,tx,rx)
#     #loc = savgol_filter(loc,10,2)


#     n = len(loc)
#     fig, ax = plt.subplots()
#     for i in range(1,n):
#         ax.plot(np.real(loc[i]), np.imag(loc[i]), '.')
    

#     ax.set_aspect("equal")
#     plt.show()

    csifile = "test-1.dat"
    csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=1)
    csidata.read()
    csi = csidata.get_scaled_csi()
    #csi = getcsi(csifile)  
    print(csi.shape)
    data1 = matrixreshape(np.transpose(np.squeeze(csi), (0, 2, 1)))

    csifile = "test-2.dat"
    csidata = csiread.Intel(csifile, nrxnum=3, ntxnum=1)
    csidata.read()
    csi = csidata.get_scaled_csi()
    #csi = getcsi(csifile)     
    data2 = matrixreshape(np.transpose(np.squeeze(csi), (0, 2, 1)))


    # dataFile1 = 's-1'
    # dataread1 = scio.loadmat(dataFile1)
    # data1 = dataread1['csi_data'].T

    # dataFile2 = 'shun2-2'
    # dataread2 = scio.loadmat(dataFile2)
    # data2 = dataread2['csi_data'].T
    # print(data1.shape[1])

    # #data1 = data1[:,:2000]
    # #data2 = data2[:, :2000]
    # print(data1.shape)
    tx = 3.2 + 0j
    r1 = 0 - 0j
    r2 = 3.2 + 2.7j
    rx = [r1, r2]
    loc_ini = 1.6 + 1.35j
  

    loc=process(data1,data2,1000,loc_ini,tx,rx)
    #loc = savgol_filter(loc,10,2)


    n = len(loc)
    fig, ax = plt.subplots()
    for i in range(1,n):
        ax.plot(np.real(loc[i]), np.imag(loc[i]), '.')
    

    ax.set_aspect("equal")
    plt.show()
    


    

