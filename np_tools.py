import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem, zscore
from scipy.ndimage import gaussian_filter1d
import erp_util
import matplotlib as mpl
def plot_raster(fig,ax,dat,offset=1,t_ar=[-1,1]):
    # dat: tr x time
    t = np.linspace(t_ar[0],t_ar[1],dat.shape[1])
    for n in range(dat.shape[0]):
        spks = t[np.where(dat[n] > 0)[0]]
        ys = np.zeros_like(spks) + n*offset
        ax.scatter(spks,ys,clip_on=False,color='k',alpha=0.8)

    t_ar = np.array(t_ar)
    t_ar[0] -= 0.2
    t_ar[1] += 0.2
    ax.set(ylim=[0,dat.shape[0]*offset],xlim=t_ar)
    #sns.despine(ax=ax,left=5,bottom=5)
    #ax.spines[['left','bottom']].set_visible(False)
    return(ax)


def get_psth_from_spikes(spikes,win_size=0.1,target_fs=100,gauss_smooth=-1,max_time=2000):

    half_size = win_size/2
    t_ar = np.arange(0,max_time,1/target_fs)
    firing_rates = []
    cur_times = spikes.copy()
    for t in t_ar:
        firing_rates.append(np.sum( (cur_times >= t-half_size) & (cur_times <= t+half_size) ) )

    if(gauss_smooth != -1):
        firing_rates = gaussian_filter1d(np.array(firing_rates) / win_size,gauss_smooth,axis=0)
    else:
        firing_rates = np.array(firing_rates) / win_size
    
    return(firing_rates,t_ar)    



def get_psth(spk_dictionary,win_size=0.1,target_fs=100,gauss_smooth=-1):
    # dat: time x neurons
    # win_size: smoothing window in seconds 
    
    max_time = np.max(spk_dictionary['mua']) + 60
    half_size = win_size/2
    t_ar = np.arange(0,max_time,1/target_fs)
    firing_rates,neuron_ids = [],[]

    for neuron in spk_dictionary['sua'].keys():
        neuron_ids.append(neuron)
        firing_rate = []
        cur_times = spk_dictionary['sua'][neuron][0]
        for t in t_ar:
            firing_rate.append(np.sum( (cur_times >= t-half_size) & (cur_times <= t+half_size) ) )
        firing_rates.append(firing_rate)

    spk_dictionary['psth'] = {}
    spk_dictionary['psth']['ids'] = neuron_ids
    spk_dictionary['psth']['time'] = t_ar
    if(gauss_smooth != -1):
        spk_dictionary['psth']['psth'] = gaussian_filter1d(np.array(firing_rates).T / win_size,gauss_smooth,axis=0)
    else:
        spk_dictionary['psth']['psth'] = np.array(firing_rates).T / win_size
    spk_dictionary['psth']['fs'] = target_fs
    spk_dictionary['psth']['win_size'] = win_size
    
    return(spk_dictionary)

def plot_raster_from_times(quality,times,neuron_id,offset=1,t_ar=[-1,1],add_psth=True,psth_fs=100):
    #spk_dictionary
    # dat: tr x time
    fig, axs = plt.subplots(2, 1, sharex=True,height_ratios=[1,5])
    ax = axs[1]
    cur_times = quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
        elg_times -= tr
        if(len(elg_times) == 0):
            continue
        y_ar = np.zeros_like(elg_times) + i*offset
        ax.scatter(elg_times,y_ar,color='k',alpha=0.3,clip_on=False,marker='|')
        ax.set(yticks=[])
        ax.axvline(0,linestyle='--',color='k')

    if add_psth: 
        ax = axs[0]
        # helper function needs pad before and after as positive
        t_ar[0] = t_ar[0]*-1
        # aligned = erp_util.align(spk_dictionary['psth']['psth'],times,
        #                t_ar,fs=int(spk_dictionary['psth']['fs'])) 

        aligned = erp_util.align(np.array(quality['psth'].to_list()).T,times,
                       t_ar,fs=int(psth_fs)) 
        
        # psth_sem = sem(aligned,axis=0)[:,neuron_id] 
        # psth_avg = np.mean(aligned,axis=0)[:,neuron_id]
        psth_sem = sem(aligned,axis=0)[:,quality.cluster_id == neuron_id].squeeze() 
        psth_avg = np.mean(aligned,axis=0)[:,quality.cluster_id == neuron_id].squeeze()
       # print(psth_avg)
        t_span = np.linspace(-1*t_ar[0],t_ar[1],psth_avg.shape[0])


        ax.plot(t_span,psth_avg,color='b',alpha=0.8)
        ax.fill_between(t_span, psth_avg - psth_sem, psth_avg + psth_sem,color='b',alpha=0.2)
        ax.set(yticks=[],ylabel='Firing rate \n (Hz)')
        return(fig,axs)










def plot_raster_from_times_to_ax(ax,quality,times,neuron_id,offset=1,t_ar=[-1,1],add_psth=True,psth_fs=100):
    #spk_dictionary
    # dat: tr x time
    cur_times = quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
        elg_times -= tr
        if(len(elg_times) == 0):
            continue
        y_ar = np.zeros_like(elg_times) + i*offset
        ax.scatter(elg_times,y_ar,color='k',alpha=0.3,clip_on=False,marker='|')
        ax.set(yticks=[])
        #ax.axvline(0,linestyle='--',color='k',linewidth=0.25,alpha=0.2)
        
    if add_psth: 
        # helper function needs pad before and after as positive
        t_ar[0] = t_ar[0]*-1
        # aligned = erp_util.align(spk_dictionary['psth']['psth'],times,
        #                t_ar,fs=int(spk_dictionary['psth']['fs'])) 

        aligned = erp_util.align(np.array(quality['psth'].to_list()).T,times,
                       t_ar,fs=int(psth_fs)) 
        

        psth_sem = sem(aligned,axis=0)[:,quality.cluster_id == neuron_id].squeeze() 
        psth_avg = np.mean(aligned,axis=0)[:,quality.cluster_id == neuron_id].squeeze()
       # print(psth_avg)
        t_span = np.linspace(-1*t_ar[0],t_ar[1],psth_avg.shape[0])


        ax.plot(t_span,psth_avg,color='b',alpha=0.8)
        ax.fill_between(t_span, psth_avg - psth_sem, psth_avg + psth_sem,color='b',alpha=0.2)
        ax.set(yticks=[],ylabel='Firing rate \n (Hz)')
    return(ax)



def return_spk_raster(spks,times,t_ar=[-1,1]):

    cur_times = spks.copy()#quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    cnt = -1
    all_elg_trial = []
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
        elg_times -= tr
        all_elg_trial.append(elg_times)

    return(all_elg_trial)


def return_spk_amp_raster(spks,amps,times,t_ar=[-1,1]):

    cur_times = spks.copy()#quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    cur_amps = amps.copy()
    cnt = -1
    all_elg_trial = []
    all_elg_amps = []
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ].squeeze()
        elg_amps =  amps[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ].squeeze()
        elg_times -= tr
        all_elg_trial.append(elg_times)
        all_elg_amps.append(elg_amps)
    return(all_elg_trial,all_elg_amps)





def plot_raster_to_ax(ax,spks,times,psth=None,offset=1,t_ar=[-1,1],add_psth=True,psth_fs=100,only_spk_trs=True,scale_psth=1,spk_col='k'):
    #spk_dictionary
    # dat: tr x time
    cur_times = spks.copy()#quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    cnt = -1
    elg_trial = []
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
        elg_times -= tr
        cnt +=1 
        if(len(elg_times) == 0):
            cnt -= 1
            elg_trial.append(False)
            continue
        elg_trial.append(True)
        if(only_spk_trs):
            y_ar = np.zeros_like(elg_times) + cnt*offset
        else:
            y_ar = np.zeros_like(elg_times) + i*offset
        ax.scatter(elg_times,y_ar,color=spk_col,alpha=0.3,clip_on=False,marker='|')
        ax.set(yticks=[])
        #ax.axvline(0,linestyle='--',color='k',linewidth=0.25,alpha=0.2)
        
    if add_psth: 
        # helper function needs pad before and after as positive
        t_ar[0] = t_ar[0]*-1
        # aligned = erp_util.align(spk_dictionary['psth']['psth'],times,
        #                t_ar,fs=int(spk_dictionary['psth']['fs'])) 

        aligned = erp_util.align(psth.reshape((-1,1)),times[elg_trial],
                       t_ar,fs=int(psth_fs)) 
        

        psth_sem = sem(aligned,axis=0).squeeze()*scale_psth
        psth_avg = np.mean(aligned,axis=0).squeeze()*scale_psth
       # print(psth_avg)
        t_span = np.linspace(-1*t_ar[0],t_ar[1],psth_avg.shape[0])


        ax.plot(t_span,psth_avg,color='r',alpha=0.8)
        ax.fill_between(t_span, psth_avg - psth_sem, psth_avg + psth_sem,color='r',alpha=0.1)
        ax.set(yticks=[],ylabel='Firing rate \n (Hz)')
    return(ax)



def plot_raster_to_ax_with_amps(ax,spks,times,amps,psth=None,offset=1,t_ar=[-1,1],add_psth=True,psth_fs=100,only_spk_trs=True,scale_psth=1,cmap='Reds'):
    cmap = mpl.colormaps[cmap]#mpl.colors.Colormap('reds')
    #spk_dictionary
    # dat: tr x time
    cur_times = spks.copy()#quality.sua[quality.cluster_id == neuron_id].values[0]#spk_dictionary['sua'][neuron_id][0]
    cnt = -1
    elg_trial = []
    for i,tr in enumerate(times):
        elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
        elg_times -= tr
        cur_amps = amps[(cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) )]
        cnt +=1 
        if(len(elg_times) == 0):
            cnt -= 1
            elg_trial.append(False)
            continue
        elg_trial.append(True)
        if(only_spk_trs):
            y_ar = np.zeros_like(elg_times) + cnt*offset
        else:
            y_ar = np.zeros_like(elg_times) + i*offset

        cur_amps -= np.min(cur_amps)
        cur_amps /= np.max(cur_amps)
        cur_cols = [cmap(a) for a in cur_amps]    
        ax.scatter(elg_times,y_ar,clip_on=False,marker='|',color=cur_cols)
        ax.set(yticks=[])
        #ax.axvline(0,linestyle='--',color='k',linewidth=0.25,alpha=0.2)
        
    if add_psth: 
        # helper function needs pad before and after as positive
        t_ar[0] = t_ar[0]*-1
        # aligned = erp_util.align(spk_dictionary['psth']['psth'],times,
        #                t_ar,fs=int(spk_dictionary['psth']['fs'])) 

        aligned = erp_util.align(psth.reshape((-1,1)),times[elg_trial],
                       t_ar,fs=int(psth_fs)) 
        

        psth_sem = sem(aligned,axis=0).squeeze()*scale_psth
        psth_avg = np.mean(aligned,axis=0).squeeze()*scale_psth
       # print(psth_avg)
        t_span = np.linspace(-1*t_ar[0],t_ar[1],psth_avg.shape[0])


        ax.plot(t_span,psth_avg,color='r',alpha=0.8)
        ax.fill_between(t_span, psth_avg - psth_sem, psth_avg + psth_sem,color='r',alpha=0.1)
        ax.set(yticks=[],ylabel='Firing rate \n (Hz)')
    return(ax)






def plot_raster_conditions_to_ax(ax,spks,time_set,colors,psth=None,offset=1,t_ar=[-1,1],add_psth=True,psth_fs=100,only_spk_trs=True):
    #spk_dictionary
    # dat: tr x time
    cur_times = spks.copy()
    cnt = -1
    all_elg_trial = []
    for ii,times in enumerate(time_set):
        elg_trial = []
        for i,tr in enumerate(times):
            elg_times = cur_times[ (cur_times >= (tr + t_ar[0]) ) & (cur_times <= (tr + t_ar[1]) ) ]
            elg_times -= tr
            cnt +=1 
            if(len(elg_times) == 0):
                cnt -= 1
                elg_trial.append(False)
                continue
            elg_trial.append(True)
            if(only_spk_trs):
                y_ar = np.zeros_like(elg_times) + cnt*offset
            else:
                y_ar = np.zeros_like(elg_times) + i*offset
        
            ax.scatter(elg_times,y_ar,alpha=0.3,clip_on=False,marker='|',color=colors[ii])
            ax.set(yticks=[])
        all_elg_trial.append(elg_trial)
    if add_psth: 
        # helper function needs pad before and after as positive
        t_ar[0] = t_ar[0]*-1
        for ii,times in enumerate(time_set):
            aligned = erp_util.align(psth.reshape((-1,1)),times[all_elg_trial[ii]],
                           t_ar,fs=int(psth_fs)) 
            

            psth_sem = sem(aligned,axis=0).squeeze()
            psth_avg = np.mean(aligned,axis=0).squeeze()
            t_span = np.linspace(-1*t_ar[0],t_ar[1],psth_avg.shape[0])
    
    
            ax.plot(t_span,psth_avg,color=colors[ii],alpha=0.8)
            ax.fill_between(t_span, psth_avg - psth_sem, psth_avg + psth_sem,color=colors[ii],alpha=0.2)
            ax.set(yticks=[],ylabel='Firing rate \n (Hz)')
    return(ax)








def makeMemMapRaw(binFullPath,nChan=383):
    #nChan = int(meta['nSavedChans'])
    test = np.fromfile(binFullPath,dtype='int16')
    nFileSamp = int(test.shape[0] / 383)
    print("nChan: %d, nFileSamp: %d" % (nChan, nFileSamp))
    rawData = np.memmap(binFullPath, dtype='int16', mode='r',
                        shape=(nChan, nFileSamp), offset=0, order='F')
    return(rawData)


