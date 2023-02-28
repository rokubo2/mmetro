from AppNotes_WinLinux.Class import TinyRad
import time as time
import numpy as np
from scipy.io import savemat
from scipy.constants import speed_of_light

try:
    # Change these two parameters for data collection
    description = "2nd-order-tag-15s-3cm"
    duration = 30 # in seconds

    filePrefix = "../Data/"
    filename = time.strftime(filePrefix + "%y%m%d-%H%M%S-" + description + "-24GHz.mat", time.localtime())

    chirpDur = 125              # Chirp duration, in microseconds
    c0 = speed_of_light         # 2.998e8
    switchPer = 625e-6          # Tag switching rate (i.e. tag period / 2)
    modF = 1 / (2 * switchPer)  # Tag modulation frequency

    # Setup Connection
    Brd = TinyRad.TinyRad('Usb', '127.0.0.1')
    Brd.BrdRst()

    # Software Version  
    Brd.BrdDispSwVers()

    # Configure Receiver
    Brd.RfRxEna()
    TxPwr = 100

    # Configure Transmitter (Antenna 0 - 2, Pwr 0 - 100)
    Brd.RfTxEna(2, TxPwr)

    CalDat = Brd.BrdGetCalDat()['Dat']

    # Configure Measurements
    Cfg = dict()
    Cfg['fStrt'] = 24.00e9                  # Start frequency
    Cfg['fStop'] = 24.25e9                  # Stop frequency
    Cfg['TRampUp'] = chirpDur * 1e-6        # Chirp duration (in seconds)
    Cfg['Perd'] = Cfg['TRampUp'] + 100e-6   # Period, chirp duration + between-chirp wait
    Cfg['N'] = 100                          # Samples per chirp
    Cfg['Seq'] = [1]                        # Tx sequence, unused
    Cfg['CycSiz'] = 2                       # ?
    Cfg['FrmSiz'] = 256                     # Chirps per frame, don't use
    Cfg['FrmMeasSiz'] = 256                 # Sampled chirps per frame, use this instead
    frmMeasSiz = Cfg['FrmMeasSiz']

    Brd.RfMeas(Cfg)
    time.sleep(1)

    # Read actual configuration
    N = int(Brd.Get('N'))
    NrChn = int(Brd.Get('NrChn'))
    fs = Brd.Get('fs')
    Perd = Cfg['Perd']

    # I have no idea why there are duplicate lines
    N = int(Brd.Get('N'))

    # Container for raw radar data and timestamps
    DataAll = None
    Timestamps = np.array([])

    # Measure and calculate Range Doppler Map
    # frameSize = int(duration / (frmMeasSiz * Perd))
    # Instead of using a fixed frameSize, use time.time() compared to duration
    print("[Rad24GHz] Started measuring")
    startTime = time.time()
    while True:
        if time.time() - startTime > duration:
            break

        DataFrame = Brd.BrdGetData()

        curr = time.time()
        strcurr = time.strftime("%Y-%m-%d %H:%M:%S:", time.localtime(curr)) + str(int(curr % 1 * 1000))
        Timestamps = np.append(Timestamps, [strcurr])
        reshaped = DataFrame.reshape(DataFrame.shape[0], 1, DataFrame.shape[1])
        DataAll = np.array(reshaped, dtype='float64') if DataAll is None else np.append(DataAll, reshaped, axis=1)

    print('[Rad24GHz] Finished measuring after {0} seconds'.format(time.time() - startTime))
    print('[Rad24GHz] Collected data in shape {0}'.format(DataAll.shape))

    save = {
        "Data": DataAll,
        "dtime": Timestamps,
        # "Brd": Brd, # Brd not used in processing code, but in collectData
        "Cfg": Cfg,
        "N": N,
        "NrChn": NrChn,
        "fs": fs,
        "measRounds": DataAll.shape[1],
        "CalDat": CalDat,
        "switchPer": switchPer
    }
    # print(save['Data'].dtype)
    savemat(filename, save)

finally:
    del Brd