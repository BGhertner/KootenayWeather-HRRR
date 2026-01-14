

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from herbie import FastHerbie
import pandas as pd

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = True
mpl.rcParams["font.size"] = 11


class forecast:


    def __init__(self, lat_lon, features, name):
        
        self.__FEATURES = features
        self.__NAME     = name

        print(f'Loading Forecast Data for {name} ...')
        print('This may take several minutes ...')

        #Current time in UTC
        recent = pd.Timestamp("now", tz='UTC').floor("1h") - pd.Timedelta("1h", tz='UTC')
        recent = [str(recent).split('+')[0]]

        #Load snow fall forecast
        fxx = list(range(0,18))
        FH_snow_fcst = FastHerbie(recent, model='hrrr', product='sfc', fxx=fxx)
        ds_snow_fcst = FH_snow_fcst.xarray(':ASNOW:')

        #Find lat-lon coordinate
        lat_vals = ds_snow_fcst['latitude'].values
        lon_vals = ds_snow_fcst['longitude'].values

        shape = lat_vals.shape
        idx = np.unravel_index(
                    np.argmin((lon_vals-lat_lon[1])**2 + (lat_vals-lat_lon[0])**2), 
                    shape)

        self.__LAT_LON  = (lat_vals[idx], lon_vals[idx])

        #Save snow fall forecast at grid point
        ds_snow_fcst = ds_snow_fcst.isel(x=idx[1], y=idx[0])
        self.__snow_fcst = ds_snow_fcst['unknown']*100
        
        #Load thermodynamic forecast
        FH_thermo = FastHerbie(recent, model='hrrr', product='prs', fxx=fxx)
        FH_thermo.download(":(?:TMP|RH|HGT|UGRD|VGRD):1*[5-9,0][0,2,5,7][0,5] mb")
        ds_thermo = FH_thermo.xarray(":(?:TMP|RH|HGT|UGRD|VGRD):1*[5-9,0][0,2,5,7][0,5] mb").isel(x=idx[1], y=idx[0])

        steps = ds_thermo['step'].to_numpy().astype(float)/3600/1e9
        prs   = ds_thermo['isobaricInhPa'].to_numpy()

        #Save data used in plotting
        _, self.__time_grid = np.meshgrid(prs, steps)

        self.__ZZ = ds_thermo['gh'].to_numpy()
        self.__TT = ds_thermo['t'].to_numpy()-273.15
        self.__RH = ds_thermo['r'].to_numpy()
        self.__UU = ds_thermo['u'].to_numpy()
        self.__VV = ds_thermo['v'].to_numpy()

        #Load past snow fall (1hr accumilation)
        process_past = lambda x: str( pd.Timestamp("now", tz='UTC').floor("1h") - pd.Timedelta(f"{1+x}h", tz='UTC') ).split('+')[0]
        FH_snow_past = FastHerbie([process_past(x) for x in range(1,19)], fxx=[1], model='hrrr', product='sfc')

        #Save past snow fall
        ds_snow_past = FH_snow_past.xarray(":ASNOW:.*:(?:0-1|[1-9]\d*-\d+) hour").isel(x=idx[1], y=idx[0])
        self.__snow_past = ds_snow_past['unknown'].values*100

        print(f'Finished Loading Forecast Data for {name}.')


    def __time_labels(self, nsteps):
        recent = str(pd.Timestamp("now").floor("1h") - pd.Timedelta("1h"))
        recent_int = int(recent.split(' ')[1].split(':')[0])
        times = np.arange(nsteps)+recent_int
        times = np.remainder(times, 24)
        def time_fmt(time):
            if   time == 0:  return '12:00 am'
            elif time < 12:  return f'{time}:00 am'
            elif time == 12: return '12:00 pm'
            else:            return f'{time-12}:00 pm'
        return [time_fmt(time) for time in times]


    def __time_labels_past(self, nsteps):
        recent = str(pd.Timestamp("now").floor("1h") - pd.Timedelta("1h"))
        recent_int = int(recent.split(' ')[1].split(':')[0])
        times = (-np.arange(1,1+nsteps) + recent_int + 24*np.ceil(nsteps/24)).astype('int')
        times = np.remainder(times, 24)
        def time_fmt(time):
            if   time == 0:  return '12:00 am'
            elif time < 12:  return f'{time}:00 am'
            elif time == 12: return '12:00 pm'
            else:            return f'{time-12}:00 pm'
        return [time_fmt(time) for time in times]


    def temperature_plot(self, zrange=(1000,4000)):

        ZZ   = self.__ZZ
        TT   = self.__TT
        time = self.__time_grid

        T_max   = int(np.ceil(np.max(np.abs(TT))))
        T_max_5 = int(np.ceil(T_max/5)*5)

        steps = np.arange(0,time.shape[0])

        fig, ax = plt.subplots(layout='constrained')
        fig.set_size_inches(8,4)
        fig.set_dpi(200)
        ax.set(ylim=zrange)

        contf = ax.contourf(time, ZZ, TT, levels=np.linspace(-T_max,T_max,num=2*T_max+1), cmap='PuOr_r')
        cbar = fig.colorbar(contf)
        cbar.set_ticks(np.arange(-np.floor(T_max/10)*10, np.floor(T_max/10)*10+1, 10))

        cont = ax.contour(time, ZZ, TT, levels=np.arange(-T_max_5, T_max_5+1, 5), linewidths=0.5, colors='k', linestyles='solid')
        ax.clabel(cont, cont.levels)

        ax.set(title=r'Temperature [$^\circ$C] at ' + f'{self.__NAME} ({self.__LAT_LON[0]:.2f}, {self.__LAT_LON[1]-360:.2f})',
            ylabel=r'Elevation [m]')

        ax.set_xticks(steps, self.__time_labels(steps.size), rotation=65, ha='right', rotation_mode='anchor')

        for label, val in self.__FEATURES.items():
            ax.axhline(val, linewidth=0.5, color='darkcyan', linestyle='--')
            ax.annotate(label, (0.05, val+20), fontsize=8, color='darkcyan')

        return fig, ax


    def humidity_plot(self, zrange=(1000,4000)):

        ZZ   = self.__ZZ
        RH   = self.__RH
        time = self.__time_grid

        steps = np.arange(0,time.shape[0])

        fig, ax = plt.subplots(layout='constrained')
        fig.set_size_inches(8,4)
        fig.set_dpi(200)
        ax.set(ylim=zrange)

        contf = ax.contourf(time, ZZ, RH, levels=np.linspace(0,100,num=101), cmap='BuPu')
        cbar = fig.colorbar(contf)
        cbar.set_ticks(np.arange(0, 101, 10))

        cont = ax.contour(time, ZZ, RH, levels=np.arange(0, 101, 5), linewidths=0.5, colors='k', linestyles='solid')
        ax.clabel(cont, cont.levels)

        ax.set(title=r'Relative Humidity [\%] at ' + f'{self.__NAME} ({self.__LAT_LON[0]:.2f}, {self.__LAT_LON[1]-360:.2f})',
            ylabel=r'Elevation [m]')

        ax.set_xticks(steps, self.__time_labels(steps.size), rotation=65, ha='right', rotation_mode='anchor')

        for label, val in self.__FEATURES.items():
            ax.axhline(val, linewidth=0.5, color='darkcyan', linestyle='--')
            ax.annotate(label, (0.05, val+20), fontsize=8, color='darkcyan')

        return fig, ax
    
    def wind_plot(self, zrange=(1000,4000)):

        ZZ    = self.__ZZ
        UU    = self.__UU
        VV    = self.__VV
        speed = np.sqrt(UU**2 + VV**2)
        time  = self.__time_grid

        steps = np.arange(0,time.shape[0])

        fig, ax = plt.subplots(layout='constrained')
        fig.set_size_inches(8,4)
        fig.set_dpi(200)
        ax.set(ylim=zrange)

        max_speed = np.ceil(np.max(speed))

        contf = ax.contourf(time, ZZ, speed, levels=np.arange(0, max_speed+1, 1), cmap='YlOrBr')
        cbar = fig.colorbar(contf)
        cbar.set_ticks(np.arange(0, max_speed, 10))

        cont = ax.contour(time, ZZ, speed, levels=np.arange(0, max_speed+5, 5), linewidths=0.5, colors='k', linestyles='solid')
        ax.clabel(cont, cont.levels)

        ax.quiver(time, ZZ, UU/speed, VV/speed, pivot='mid', width=0.002)

        ax.set(title=r'Wind Speed [m/s] at ' + f'{self.__NAME} ({self.__LAT_LON[0]:.2f}, {self.__LAT_LON[1]-360:.2f})',
            ylabel=r'Elevation [m]')

        ax.set_xticks(steps, self.__time_labels(steps.size), rotation=65, ha='right', rotation_mode='anchor')

        for label, val in self.__FEATURES.items():
            ax.axhline(val, linewidth=0.5, color='darkcyan', linestyle='--')
            ax.annotate(label, (0.05, val+20), fontsize=8, color='darkcyan')

        return fig, ax

    def snow_plot(self):

        snow_fcst = self.__snow_fcst
        snow_past = self.__snow_past

        snow_acc = [-np.sum(snow_past[-idx:]) for idx in range(0,snow_past.size+1)]
        snow_acc[0] = 0

        steps = np.arange(0,snow_fcst.size)

        ntimes_past = snow_past.size
        steps_past  = -np.arange(0,1+ntimes_past)

        fig, ax = plt.subplots(layout='constrained')
        fig.set_size_inches(8,3)
        fig.set_dpi(200)

        ax.plot(steps_past, snow_acc, '--k', linewidth=1)
        ax.plot(steps, snow_fcst, 'k', linewidth=1)

        ax.fill_between(steps_past, snow_acc, color='grey', alpha=0.3)
        ax.fill_between(steps, snow_fcst, color='grey', alpha=0.3)

        ax.grid()
        ax.set(xlim = (steps_past[-1], steps[-1]),
            ylim = (np.minimum(-3, snow_acc[-1]*1.1), np.maximum(3, snow_fcst[-1]*1.1)),
            title=r'Snow Fall at ' + f'{self.__NAME} ({self.__LAT_LON[0]:.2f}, {self.__LAT_LON[1]-360:.2f})',
            ylabel='Forcasted Snow Fall [cm]\n(Negative for Accumulated)')
        ax.axvline(0, color='brown', linestyle='dashed', linewidth=1)

        ax.set_xticks((list(steps_past[:0:-1]) + list(steps))[::2], (self.__time_labels_past(ntimes_past)[::-1] + self.__time_labels(steps.size))[::2], rotation=65, ha='right', rotation_mode='anchor')

        return fig, ax