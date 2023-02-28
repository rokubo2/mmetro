from AppNotes_WinLinux.Class import TinyRad
import time as time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.constants import speed_of_light

print("start visualizing")

try:
    duration = 30 # Visualization duration

    # Most constants are commented in collect_data.py
    maxRange = 10
    chirpDur = 125
    c0 = speed_of_light
    switchPer = 625e-6
    modF = 1 / (2 * switchPer)
    tDelay = 0.05

    # Setup Connection
    Brd = TinyRad.TinyRad('Usb', '127.0.0.1')
    Brd.BrdRst()

    # Software Version
    Brd.BrdDispSwVers()

    # Configure Receiver
    Brd.RfRxEna()

    # Configure Transmitter (Antenna 0 - 2, Pwr 0 - 100)
    TxPwr = 100
    Brd.RfTxEna(2, TxPwr)

    CalDat = Brd.BrdGetCalDat()['Dat']

    # Configure Measurements
    Cfg = dict()
    Cfg['fStrt'] = 24.00e9
    Cfg['fStop'] = 24.25e9
    Cfg['TRampUp'] = chirpDur * 1e-6
    Cfg['Perd'] = Cfg['TRampUp'] + 100e-6
    Cfg['N'] = 100
    Cfg['Seq'] = [1]
    Cfg['CycSiz'] = 2
    Cfg['FrmSiz'] = 256
    Cfg['FrmMeasSiz'] = 256
    frmMeasSiz = Cfg['FrmMeasSiz']

    Brd.RfMeas(Cfg)
    time.sleep(1)

    # Read actual configuration
    N = int(Brd.Get('N'))
    NrChn = int(Brd.Get('NrChn'))
    fs = Brd.Get('fs')
    Perd = Cfg['Perd']

    # I have no idea why there are duplicate lines in the original library
    N = int(Brd.Get('N'))
    TRampUp = Brd.Get('TRampUp')

    # Processing of range profile
    Win2D = Brd.hanning(N, int(frmMeasSiz))
    ScaWin = sum(Win2D[:, 0])
    NFFT = 2**10
    NFFTVel = 2**8
    kf = (Cfg['fStop'] - Cfg['fStrt']) / Cfg['TRampUp']
    vRange = np.arange(NFFT) / NFFT * fs * c0 / (2*kf)
    fc = (Cfg['fStop'] + Cfg['fStrt']) / 2

    RMin = 0.2
    RMax = min(maxRange, (N / TRampUp) * c0 / (4 * 250e6 / TRampUp))
    RMinIdx = np.argmin(np.abs(vRange - RMin))
    RMaxIdx = np.argmin(np.abs(vRange - RMax))
    vRangeExt = vRange[RMinIdx:RMaxIdx]

    WinVel2D = Brd.hanning(int(frmMeasSiz), len(vRangeExt))
    ScaWinVel = sum(WinVel2D[:, 0])
    WinVel2D = WinVel2D.transpose()

    vFreqVel = np.arange(-NFFTVel//2, NFFTVel//2)/NFFTVel*(1/Cfg['Perd'])
    vVel = vFreqVel*c0/(2*fc)

    # Visualization settings
    fig, axes = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.4)

    vRangeExtVisIdx = np.arange(0, vRangeExt.shape[0], 20)
    vRangeExtVis = list(map(lambda x: "%.2f" % x, vRangeExt[vRangeExtVisIdx]))
    vVelVisIdx = np.arange(0, vVel.shape[0], 50)
    vVelVis = list(map(lambda x: "%.2f" % x, vVel[vVelVisIdx]))
    chirpVisIdx = np.arange(0, Cfg["FrmMeasSiz"], 50)

    # Measure and calculate Range Doppler Map
    # frameSize = int(duration / (frmMeasSiz * Perd))
    # Instead of using a fixed frameSize, use time.time() compared to duration
    print("[Rad24GHz] Started Visualizing")
    startTime = time.time()
    while True:
        if time.time() - startTime > duration:
            break

        DataFrame = Brd.BrdGetData()

        # Range FFT and Doppler FFT
        Data = np.reshape(DataFrame[:, 0], (N, frmMeasSiz), order='F')
        RP = 2 * fft(np.multiply(Data, Win2D), n=NFFT, axis=0) / ScaWin * Brd.FuSca
        RPExt = RP[RMinIdx:RMaxIdx, :]
        RD = fft(np.multiply(RPExt, WinVel2D), n=NFFTVel, axis=1) / ScaWinVel

        axes[0].imshow(np.abs(RPExt), aspect='auto')
        axes[0].set_title("Range Profile")
        axes[0].set_xticks(chirpVisIdx)
        axes[0].set_xlabel("Chirps")
        axes[0].set_yticks(vRangeExtVisIdx, labels=vRangeExtVis)
        axes[0].set_ylabel("Range (m)")
        # axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[1].imshow(np.abs(RD), aspect='auto')
        axes[1].set_title("Doppler Profile")
        axes[1].set_xticks(vVelVisIdx, labels=vVelVis)
        axes[1].set_xlabel("Vel Bins")
        axes[1].set_yticks(vRangeExtVisIdx, labels=vRangeExtVis)
        axes[1].set_ylabel("Range (m)")

        fig.savefig("_visualize_.jpg") # File name _*_.jpg for .gitignore

        time.sleep(tDelay)

    print('[Rad24GHz] Finished visualizing after {0} seconds'.format(time.time() - startTime))

finally:
    del Brd