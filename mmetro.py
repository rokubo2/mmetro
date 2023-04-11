import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
import pickle
import pandas as pd
import glob
import os
from pprint import pprint
from datetime import datetime
from progressbar import ProgressBar, Bar, Percentage
from ipywidgets import IntProgress
from IPython.display import display
from scipy import signal

class Mmetro_Utils():
    def __init__(self,
                 maxRange = 30,
                 chirpDur = 125,
                 switchPer = 2500e-6,
                 modDuty = 50,
                 fs = 1.0e6,
                 RMin = 1) -> None:
        
        self.maxRange = maxRange
        self.chirpDur = chirpDur
        self.c0 = speed_of_light
        self.swichDur = switchPer
        self.modDuty = modDuty
        self.fs = fs
        self.modF = 1 / (2 * switchPer)
        self.RMin = RMin

        return
    
    def hanning(self, M, *varargin):
        m = np.linspace(-(M-1)/2,(M-1)/2,M)  
        Win = 0.5 + 0.5*np.cos(2*np.pi*m/M)
        if len(varargin) > 0:
            N = varargin[0]
            Win = np.broadcast_to(Win,(N,M)).T
        return Win
    
    def get_RP_RD(self, data_snapshot, cfg, nFFT, nFFT_vel):
        '''========================= setting config information ============================='''
        frmMeasSiz = cfg['FrmMeasSiz']
        N = cfg['N']
        TRampUp = cfg['TRampUp']   
        kf = (cfg['fStop'] - cfg['fStrt']) / cfg['TRampUp']
        vRange = np.arange(nFFT) / nFFT * self.fs * self.c0 / (2*kf)
        fc = (cfg['fStop'] + cfg['fStrt']) / 2
        
        RMin = self.RMin
        RMax = min(self.maxRange, (N / TRampUp) * self.c0 / (4 * 250e6 / TRampUp))
        RMinIdx = np.argmin(np.abs(vRange - RMin))
        RMaxIdx = np.argmin(np.abs(vRange - RMax))
        vRangeExt = vRange[RMinIdx:RMaxIdx]
        
        # reshape the data
        data = np.reshape(data_snapshot, (N, frmMeasSiz), order='F')
        # background substraction
        data = data - data[:,0][:,None]

        
        '''================================ getting RP ====================================='''
        # hanning window used for getting RP
        Win2D = self.hanning(N, int(frmMeasSiz))
        ScaWin = sum(Win2D[:, 0])
        
        RP = 2 * np.fft.fft(np.multiply(data, Win2D), n=nFFT, axis=0) / ScaWin * 0.498 / 65536
        
        '''================================= getting RD ====================================='''
        # hanning window used for getting RD
        WinVel2D = self.hanning(int(frmMeasSiz), len(vRangeExt))
        ScaWinVel = sum(WinVel2D[:, 0])
        WinVel2D = WinVel2D.transpose()
        
        # getting RD
        RPExt = RP[RMinIdx:RMaxIdx, :]
        RD = np.fft.fft(np.multiply(RPExt, WinVel2D), n=nFFT_vel, axis=1) / ScaWinVel
        
        return RPExt, RD, vRangeExt
    
    def plot_doppler(self, file_name, frame_idx, save_fig = False, save_RD_norm = False, save_dir = 'Data/', cut=[30,30], threshold = 0.3):
        # load data from file name
        file = open(file_name, 'rb')
        data_raw = pickle.load(file)
        file.close()
        
        # get singe channel raw data and its config
        data_single_chn = data_raw['Data'][:,:,1] # select one channel (antena)
        data_snapshot = data_single_chn[:,frame_idx]
        cfg = data_raw['Cfg']
    
        # get RP and RD
        nFFT = 2**10
        nFFT_vel = 2**8
        RPExt, RD, vRangeExt = self.get_RP_RD(data_snapshot, cfg, nFFT, nFFT_vel)
        
        # normalize RD for plotting
        RD = RD[:,cut[0]:-cut[1]]
        RD_diff = np.abs(RD) - np.min(np.abs(RD))
        RD_norm = RD_diff / np.max(np.abs(RD))
        RD_norm[RD_norm < threshold] = 0
        
        # Get information for plotting
        fc = (cfg['fStop'] + cfg['fStrt']) / 2
        vFreqVel = np.arange(-nFFT_vel//2, nFFT_vel//2)/nFFT_vel*(1/cfg['Perd'])
        vVel = (vFreqVel*self.c0/(2*fc))[cut[0]:-cut[1]]
        
        vRangeExtVisIdx = np.arange(0, vRangeExt.shape[0], 20)
        vRangeExtVis = list(map(lambda x: "%.2f" % x, vRangeExt[vRangeExtVisIdx]))
        vVelVisIdx = np.arange(0, vVel.shape[0], 50)
        vVelVis = list(map(lambda x: "%.2f" % x, vVel[vVelVisIdx]))
        chirpVisIdx = np.arange(0, cfg["FrmMeasSiz"], 50)
        
        # plotting RP and RD
        fig, axes = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0.4)
        axes[0].imshow(np.abs(RPExt), aspect='auto')
        axes[0].set_title("Range Profile")
        axes[0].set_xticks(chirpVisIdx)
        axes[0].set_xlabel("Chirps")
        axes[0].set_yticks(vRangeExtVisIdx, labels=vRangeExtVis)
        axes[0].set_ylabel("Range (m)")

        axes[1].imshow(np.abs(RD_norm), aspect='auto')
        axes[1].set_title("Doppler Profile")
        # axes[1].set_xticks(vVelVisIdx, labels=vVelVis)
        axes[1].set_xlabel("Vel Bins")
        axes[1].set_yticks(vRangeExtVisIdx, labels=vRangeExtVis)
        axes[1].set_ylabel("Range (m)")
        
        # save figure/data setting
        if save_fig:
            path = save_dir + file_name[28:-7] + '/image'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(path + '/frame_'+str(frame_idx)+'.png')
        if save_RD_norm:
            path = save_dir + file_name[28:-7] + '/RD_norm'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path+'/frame_' + str(frame_idx) + '.npy', RD_norm)
            
        plt.show()

    def sincTemplate(self, cfg, modF, modDuty, NFFTVel, vel):
        fc = (cfg['fStop'] + cfg['fStrt']) / 2
        c0 = 2.998e8
        t = np.arange(0, cfg['Perd']*200, cfg['Perd'])
        doppShift = np.fft.fft(np.cos(2*np.pi*(2*vel*fc/c0*t)), n=NFFTVel, axis=0)
        undopp_square = np.fft.fft(sp.signal.square(2*np.pi*modF*t, modDuty/100), n=NFFTVel, axis=0)
        if vel > 0.01:
            doppShift[int(NFFTVel/2):] = 0
            sq_wav = np.convolve(doppShift, undopp_square)
            to_return = np.abs(sq_wav[:NFFTVel])
        elif vel < -0.01:
            doppShift[int(NFFTVel/2):] = 0
            doppShift = np.flip(doppShift)
            sq_wav = np.convolve(doppShift, undopp_square)
            to_return = np.abs(sq_wav[NFFTVel-1:])
        else:
            to_return = np.abs(undopp_square)
        to_return = to_return / np.max(to_return)
        to_return[to_return<0.1] = 0
        return to_return

    def create_template(self, vel_list, cfg, NFFTVel):
        templateAll = np.zeros((len(vel_list), NFFTVel))
        for vIdx in range(len(vel_list)):
            template = self.sincTemplate(cfg, self.modF, self.modDuty, NFFTVel, vel_list[vIdx])
            templateAll[vIdx] = template
        return templateAll
        
    def get_position(self, RD_norm, templateAll, vRangeExt):
        max_max_corr = 0
        max_idx = 0
        for i in range(len(templateAll)):
            template = templateAll[i]
            corr = np.abs(RD_norm) * np.repeat(np.abs(template[:, np.newaxis]), RD_norm.shape[0], axis = 1).T
            corr_sum = np.sum(np.abs(corr), axis=1)
            max_corr = np.max(corr_sum)
            if max_corr > max_max_corr:
                vel_i = i
                max_max_corr = max_corr
                max_idx = np.argmax(corr_sum)
                                    
        range_idx = max_idx
        to_return = vRangeExt[range_idx]
        return to_return, vel_i, range_idx, max_max_corr

    def get_timestamps(self, data_raw):
        date_time_obj = []
        first_timestamp = True
        for i in range(len(data_raw['dtime'])):
            date = data_raw['dtime'][i]
            while(len(date) < 23):
                pre = date[:20]
                post = date[20:]
                date = pre+'0'+post
                
            if first_timestamp:
                time_init = datetime.strptime(date, '%Y-%m-%d %H:%M:%S:%f')
                date_time_obj.append(0)
                first_timestamp = False
            else:
                delta = datetime.strptime(date, '%Y-%m-%d %H:%M:%S:%f') - time_init
                date_time_obj.append(delta.total_seconds()) 
        return date_time_obj
    
    def process_plot(self, names, save=True, save_dir = 'Data/', file_name = None, cut=[30,30], threshold=0.3):
        for names_i in range(len(names)):
            file = open(names[names_i], 'rb')
            data_raw = pickle.load(file)
            file.close()
            data_single_chn = data_raw['Data'][:,:,0]
            cfg = data_raw['Cfg']
            n_frame = data_single_chn.shape[1]
            position = np.zeros(n_frame)
            vel = np.zeros(n_frame)
            nFFT = 2**10
            nFFT_vel = 2**8
            
            pprint("start processing " + names[names_i])

            f = IntProgress(min=0, max=n_frame) # instantiate the bar
            display(f)
            for i in range(n_frame):
                data_snapshot = data_single_chn[:,i]
                RPExt, RD, vRangeExt = self.get_RP_RD(data_snapshot, cfg, nFFT, nFFT_vel)
                
                RD = RD[:,cut[0]:-cut[1]]
                RD_diff = np.abs(RD) - np.min(np.abs(RD))
                RD_norm = RD_diff / np.max(np.abs(RD))
                RD_norm[RD_norm < threshold] = 0
                
                velArr = np.arange(-10,1,0.1)
                templateAll = self.create_template(velArr, cfg, nFFT_vel)
                templateAll = templateAll[:, cut[0]:-cut[1]]

                position[i], vel_i, _, _ = self.get_position(RD_norm, templateAll, vRangeExt)
                vel[i] = velArr[vel_i]
                f.value += 1
                
                
            timestamps = self.get_timestamps(data_raw)

            plt.scatter(timestamps, position, s=5)
            plt.xlabel('time (ms)')
            plt.ylabel('range (m)')
            
            plt.figure()
            plt.scatter(timestamps, vel, s=5)
            
            
            if save:
                to_save = [timestamps, position]
                to_save_v = [timestamps, vel]
                path = save_dir + names[names_i][28:-7]
                if not os.path.exists(path):
                    os.makedirs(path)
                if file_name:
                    np.save(path+ '/' + file_name + '.npy', to_save)
                    plt.savefig(path + '/' + file_name + '_plot.png')
                    np.save(path+ '/' + file_name + '_vel.npy', to_save_v)
                else:
                    np.save(path+ '/processed_range_time.npy', to_save)
                    plt.savefig(path + '/range_time_plot.png')
                    np.save(path + '/processed_velocity.npy', to_save_v)
                
            plt.show()

class Kalman:
    def __init__(self,
                 p = 1,
                 p_v = 0.1,
                 q = 0.1,
                 r = 0.5,
                 threshold = 2
                 ):
        
        self.p_init = p
        self.p_v = p_v
        self.q = q
        self.r_default = r
        self.threshold = threshold

        return
    
    def _predict(self, x, p, v, dt):
        x = x + dt*v                    # State Transition Equation (Dynamic Model or Prediction Model)
        p = p + (dt**2 * self.p_v) + self.q
        return x, p
    
    def _update(self, x, p, z, velocity):
        err = False
        if abs(z - x) > abs(self.threshold):
            r = 100
            err = True
        else:
            r = 0.5
        k = p / ( p + r)                # Kalman Gain
        x = x + k * (z - x)             # State Update
        p = (1 - k) * p                 # Covariance Update
        return x, p, err
    
    def run(self, filename, save=True, plot=True, start = 0, end = -1):
        # load distance and velocity file
        distance_file = 'Data/' + filename[28:-7]+'/processed_range_time.npy'
        velocity_file = 'Data/' + filename[28:-7]+'/processed_velocity.npy'

        # parse npy file to retrieve timestamp, range, and velocity data
        data_range = np.load(distance_file)
        data_velocity = np.load(velocity_file)
        timestamp = data_range[0]
        distance = data_range[1]
        velocity = data_velocity[1]

        # initial set up
        n = len(timestamp)
        dt = (timestamp[-1] - timestamp[0]) / n
        x = distance[0]
        p = self.p_init

        # set up estimated range array to be returned
        x_to_return = []
        x_to_return.append(x)
        
        # iterate thru each frame and get state updated
        num_err = 0
        for i in range(1, n):
            x_, p_ = self._predict(x, p, velocity[i], dt)
            z = distance[i]
            x, p, err = self._update(x_, p_, z, velocity[i])
            x_to_return.append(x)
            if err:
                num_err+=1

        # save processed data
        if save:
            range_kalman = [timestamp, x_to_return]
            speed_est = (distance[start] - distance[end]) / (timestamp[end] - timestamp[start])
            to_save = {'data_raw':distance,
                       'data':range_kalman, 
                       'timestamp':timestamp,
                       'num_err':num_err,
                       'n_frame':n,
                       'speed_est':speed_est
                       }
            path = 'Data/' + filename[28:-7] + '/processed_data.pickle'
            with open(path, 'wb') as handle:
                pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #plot data
        if plot:
            plt.scatter(timestamp, distance, s=5, c='g')
            plt.plot(timestamp, x_to_return, alpha=1, c='r')
            if start != 0:
                plt.axvline(timestamp[start], c = 'b', alpha=0.5)
            if end != -1:
                plt.axvline(timestamp[end], c = 'b', alpha=0.5)
            plt.xlabel('time (s)')
            plt.ylabel('distance (m)')
            plt.legend(['raw', 'filtered'])
            plt.show()

        return
