import matplotlib.pyplot as plt
import scipy.io as scio
import math
import numpy as np
import cmath
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def process(csi_data1, csi_data2, samp_rate, loc_ini):
    lambda_ = 3e8 / 5.24e9

    # tx=0+0i
    tx = 1 + 2j

    time_length = csi_data1.shape[1]
    time = np.zeros(time_length)
    for i in range(0, time_length):
        time[i] = 1 / samp_rate * i

    # r1=2.8+0i
    r1 = 0 + 0j

    # r2=3+0i
    r2 = 3 + 0j

    # initpoint=0.5-0.5i
    initpoint = loc_ini

    rx = [r1, r2]  # three RXs

    # read CSI data

    # cut three CSI data to the same length
    # [pc1, pc2, pc3, time] = align_time3(time1, time2, time3, pc1, pc2, pc3)

    samp_rate = samp_rate

    pc1 = csi_data1
    pc2 = csi_data2

    skip_silence = 1.4

    # get Doppler frequency shift (in Hz)
    speed1, score1, slope, duslope, mrange, ne1 = mimo2speed(pc1, samp_rate)
    speed2, score2, slope, duslope, mrange, ne2 = mimo2speed(pc2, samp_rate)
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
    speed = speed[skip:-150, :]
    score = score[skip:-150, :]
    # print(speed)



    # calculate trajectory
    loc = trajectory_by_doppler_v3(speed, score, np.diff(time), initpoint, tx, rx, 10)




    return loc,ne1,ne2

    # overlap groundtruth
    # plt.plot(np.real(groundtruth), np.imag(groundtruth), '--', color='red')


def mimo2speed(rx, samp_rate):
    # get CSI ratio from raw CSI input
    csiq = getAntMIMO(rx, 2, 3)

    speed_high, score_high, agree_high, slope, duslope, mrange, ne = windowd_speed(csiq, samp_rate, 40)
    speed_low, score_low, agree_low = windowd_speed(csiq, samp_rate, 25)[:3]
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

    x = np.arange(1, len(r1) + 1)

    r1avg = np.nanmean(np.abs(r1))
    r2avg = np.nanmean(np.abs(r2))
    r3avg = np.nanmean(np.abs(r3))
    level = [r1avg, r2avg, r3avg]

    if level[ant1 - 1] > level[ant2 - 1]:
        maxIndex = ant1
        minIndex = ant2
    else:
        maxIndex = ant2
        minIndex = ant1

    # print('level = ', level)
    # print('r', minIndex, '/r', maxIndex, ', r', minIndex, '/r', maxIndex)
    if minIndex == 1:
        ret = rx1 / rx3
    else:
        ret = rx3 / rx1
    denominator = maxIndex

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


def windowd_speed(csiq, samp_rate, window_size):
    timeslot = 6 / window_size  # 0.05 before modification
    wsize = int(np.floor(timeslot * samp_rate))

    window_size = window_size * samp_rate / 400

    ncsi = np.mean(csiq, axis=0)
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
    Signal denoising using Savitzky-Golay smoothing.
    Args:
        x (ndarray): Input signal.
        winsize (float): Window size for smoothing.
    Returns:
        ndarray: Denoised signal.
    """
    # Check input length
    len_x = len(x)
    if len_x < 30:
        return x

    # Set polynomial order and window length
    n = 2
    f = int(winsize)  # window length (about 1 period)

    # Adjust window length
    if f > len_x:
        f = len_x - 2
    if f % 2 == 0:
        f += 1  # Make it odd

    # Apply Savitzky-Golay filter
    ret = savgol_filter(x, f, n)
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

        v[i] = (newPos - curPos) / deltaT[i]
        if ~np.isnan(newPos):
            curPos = newPos
        LocT[i] = np.sum(deltaT[s:e])
    LocT = np.cumsum(LocT)
    return Loc


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


def get_scaled_csi(csi_st):
    # Pull out CSI
    csi = csi_st['csi']

    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = csi * np.conj(csi)
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csi_st))
    # Scale CSI -> Signal power : rssi_pwr / (mean of csi_pwr)
    scale = rssi_pwr / (csi_pwr / 30)

    # Thermal noise might be undefined if the trace was
    # captured in monitor mode.
    # ... If so, set it to -92
    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)

    # Quantization error: the coefficients in the matrices are
    # 8-bit signed numbers, max 127/-128 to min 0/1. Given that Intel
    # only uses a 6-bit ADC, I expect every entry to be off by about
    # +/- 1 (total across real & complex parts) per entry.
    #
    # The total power is then 1^2 = 1 per entry, and there are
    # Nrx*Ntx entries per carrier. We only want one carrier's worth of
    # error, since we only computed one carrier's worth of signal above.
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])

    # Total noise and error power
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    # Ret now has units of sqrt(SNR) just like H in textbooks
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * np.sqrt(2)
    elif csi_st['Ntx'] == 3:
        # Note: this should be sqrt(3)~ 4.77 dB. But, 4.5 dB is how
        # Intel (and some other chip makers) approximate a factor of 3
        #
        # You may need to change this if your card does the right thing.
        ret = ret * np.sqrt(dbinv(4.5))
    return ret


def get_total_rss(csi_st):
    # Careful here: rssis could be zero
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag += dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag += dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag += dbinv(csi_st['rssi_c'])

    ret = np.log10(rssi_mag) * 10 - 44 - csi_st['agc']
    return ret


def m_normTime(ts):
    dt = np.diff(ts) / 1000000
    mm = np.logical_or(dt > 1, dt < 0)
    if np.sum(mm) > 0:
        dt[mm] = np.mean(dt[~mm])
    t = np.concatenate(([0], np.cumsum(dt)))
    mm = np.concatenate(([0], (dt == 0)))
    if np.sum(mm) > 0:
        idx = np.arange(len(ts))
        t = np.interp(idx, idx[~mm], t[~mm])

    return t


def dbinv(x):
    return np.power(10, x / 10)


import matplotlib.pyplot as plt
import numpy as np


def plotPolarColor(x, caption):
    # 绘制标题
    plt.title(caption)

    # 计算数据点数量和颜色映射
    n = len(x)
    colors = plt.cm.jet(np.linspace(0, 1, n))
    print("Traj Number",n)

    # 绘制第一个数据点

    # 绘制剩余的数据点
    for i in range(1, n):
        plt.plot(np.real(x[i]), np.imag(x[i]), '.', color=colors[i])
        
    

    # 设置坐标轴范围
    plt.xlim((-2, 4))
    plt.ylim((0, 6))
    plt.show()





if __name__ == '__main__':
    dataFile1 = 'shun2-1'
    dataread1 = scio.loadmat(dataFile1)
    data1 = dataread1['csi_data'].T

    dataFile2 = 'shun2-2'
    dataread2 = scio.loadmat(dataFile2)
    data2 = dataread2['csi_data'].T
    print(data1.shape[1])

    #data1 = data1[:,:2000]
    #data2 = data2[:, :2000]
    print(data1.shape)


    process(data1, data2,400,loc_ini=0.25+1j)
