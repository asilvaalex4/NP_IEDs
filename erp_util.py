import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib as mpl
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import sem
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 8})#, 'font.sans-serif': 'Arial'})


def plot_single_erp(dat,ax=None,t_ar=None,alpha=0.3,color='b',label=None):
    if(ax is None):
        fig,ax = plt.subplots()
    if(label is None):
        label = '_hide'
    
    sig = dat.mean(0)
    err = sem(dat,axis=0)
    if(t_ar is None): 
        t_ar = range(sig.shape[0])
    ax.plot(t_ar,sig,color=color,label=label)
    ax.fill_between(t_ar,sig+err,sig-err,color=color,alpha=alpha)
    return(ax,sig,err)




def extract_interval(dat,times,pads,fs=400,floor=False):
    # INPUTS:
        # dat: T x N matrix
        # times: array of 2 timepoints (not samples)
        # pads: [t_before,t_after]
        # fs: sampling frequency   
    times[0] = times[0] - pads[0]
    times[1] = times[1] + pads[1]
    if(floor):
        samples = np.round(times*fs).astype(int)
    else:
        samples = (times*fs).astype(int)
    return(dat[samples[0]:samples[1],:])


import matplotlib.pyplot as plt

def align(dat,times,pads,fs=400):
    # INPUTS:
        # dat: T x N matrix
        # times: array of timepoints (not samples)
        # pads: [t_before,t_after]
        # fs: sampling frequency   
    samples = (times*fs).astype(int)
    aligned = np.zeros((times.shape[0],int(sum(pads*fs)),dat.shape[1]))
    for ind,start in enumerate(samples):
        #aligned[ind,:,:] = dat[start-int(pads[0]*fs):start+int(pads[1]*fs),:]
        aligned[ind,:,:] = dat[start-int(pads[0]*fs):start-int(pads[0]*fs)+aligned.shape[1],:]
    return(aligned)


import matplotlib.pyplot as plt

def align_tr_df(tr_df,pads,target_col,data_col='neural'):
    aligned_dat = []
    for i,row in tr_df.iterrows():
        aligned_dat.append(row[data_col][(row[target_col] - pads[0]):(row[target_col] + pads[1]),:])
    return(np.array(aligned_dat))

def extract_tr_df(tr_df,start_phase,end_phase):
    aligned_dat = []
    for i,row in tr_df.iterrows():
        aligned_dat.append(row.neural[row[start_phase]:row[end_phase],:])
    return(aligned_dat)   


def align_tr_df_chars(tr_df,pads):
    aligned_dat = []
    aligned_chars = []
    for i,row in tr_df.iterrows():
        for c,t in zip(row.chars,row.char_times):
            aligned_dat.append(row.neural[(t - pads[0]):(t + pads[1]),:])
            aligned_chars.append(c)
    
    
    return(np.array(aligned_dat),np.array(aligned_chars))


def align_tr_df_chars_in_sents(tr_df,pads):
    aligned_dat = []
    aligned_chars = []
    for i,row in tr_df.iterrows():
        for w,w_t in zip(row.chars,row.char_onsets):
            for c,t in zip(w,w_t):
                aligned_dat.append(row.neural[(t - pads[0]):(t + pads[1]),:])
                aligned_chars.append(c)
    
    
    return(np.array(aligned_dat),np.array(aligned_chars))

def align_tr_df_chars_pencil(tr_df,pads):
    aligned_dat = []
    aligned_chars = []
    for i,row in tr_df.iterrows():
        for c,t in zip(row.chars,row.char_times):
            t -= row.action_onset
            t /= 4.
            t = int(t)
            cur_aligned = row.cont_feats[(t - pads[0]):(t + pads[1]),:]
            extra_samps = pads[1] - cur_aligned.shape[0]
            if(extra_samps != 0):
                cur_aligned = np.concatenate((cur_aligned,np.zeros((extra_samps,9))),axis=0)
            aligned_dat.append(cur_aligned)
            aligned_chars.append(c)
    
    
    return(np.array(aligned_dat),np.array(aligned_chars))



def align_tr_df_words_in_sent(tr_df,pads):
    aligned_dat = []
    aligned_chars = []
    for i,row in tr_df.iterrows():
        for c,t in zip(row.words,row.word_onsets):
            aligned_dat.append(row.neural[(t - pads[0]):(t + pads[1]),:])
            aligned_chars.append(c)
    return(np.array(aligned_dat),np.array(aligned_chars))

def get_cond_erps(aligned,ids,des_ids):
    means = []
    sems = []
    for cur_id in des_ids:
        means.append(np.mean(aligned[ids==cur_id],axis=0))
        sems.append(stats.sem(aligned[ids==cur_id],axis=0))
    
    return(means,sems)
        

def plot_erp(aligned,grid,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),anat=None,col_mapper=None,reg_alpha=0.2,line_color='k',metric=np.mean):
    aligned = metric(aligned,axis=0)
    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax[r,c].plot(aligned[:,grid[r,c]],color=line_color)
            ax[r,c].set_title(str(grid[r,c]+1))
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            ax[r,c].axhline(0)
            ax[r,c].axvline(fs*t_lims[0])
            ax[r,c].set_ylim(ylim)
            if(anat is not None):
                try:
                    ax[r,c].patch.set_facecolor(col_mapper[anat[grid[r,c]]])
                    ax[r,c].patch.set_alpha(reg_alpha)
                except:
                    ax[r,c].patch.set_facecolor('#808080')
                    ax[r,c].patch.set_alpha(reg_alpha)
            
    return(fig,ax) 

def plot_mult_erps(aligned_conditions,grid,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),cur_cols=sns.color_palette(),metric=np.mean,anat=None,col_mapper=None,reg_alpha=0.2):
    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for i,aligned in enumerate(aligned_conditions):  
                if(len(aligned.shape)>2):
                    aligned = metric(aligned,axis=0)
                ax[r,c].plot(aligned[:,grid[r,c]],color=cur_cols[i])
                
            ax[r,c].set_title(str(grid[r,c]+1))
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            ax[r,c].axhline(0)
            ax[r,c].axvline(fs*t_lims[0])
            ax[r,c].set_ylim(ylim)
            if(anat is not None):
                try:
                    ax[r,c].patch.set_facecolor(col_mapper[anat[grid[r,c]]])
                    ax[r,c].patch.set_alpha(reg_alpha)
                except:
                    ax[r,c].patch.set_facecolor('#808080')
                    ax[r,c].patch.set_alpha(reg_alpha)

    return(fig,ax)



def plot_enc_maps(aligned,grid,rs,t_lims=[0,0.6],fs=400,figsize=(15,15),anat=None,col_mapper=None,reg_alpha=0.2,cmap='coolwarm',clim=[-1,1]):

    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            #print(aligned[grid[r,c],:].sum())
            ax[r,c].pcolormesh(aligned[grid[r,c]],cmap=cmap)#,vmin=clim[0],vmax=clim[1])
            title = str(grid[r,c]+1)+ ': ' + str(np.round(rs[grid[r,c]],2))
            ax[r,c].set_title(title,fontsize=8,pad=-1)
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            #ax[r,c].axhline(0)
            #ax[r,c].axvline(fs*t_lims[0])
            #ax[r,c].set_ylim(ylim)
            # if(anat is not None):
            #     try:
            #         ax[r,c].patch.set_facecolor(col_mapper[anat[grid[r,c]]])
            #         ax[r,c].patch.set_alpha(reg_alpha)
            #     except:
            #         ax[r,c].patch.set_facecolor('#808080')
            #         ax[r,c].patch.set_alpha(reg_alpha)
            
    return(fig,ax) 





def plot_elec_all_phases(tr_df,elec):
    stim_erp = np.mean(align_tr_df(tr_df,[400,400],'stim_cue'),axis=0)
    delay_erp = np.mean(align_tr_df(tr_df,[200,400],'delay_cue'),axis=0)
    go_erp = np.mean(align_tr_df(tr_df,[200,400],'go_cue'),axis=0)
    action_erp = np.mean(align_tr_df(tr_df,[200,1200],'action_onset'),axis=0)
    
    from scipy.stats import sem
    stim_sem = sem(align_tr_df(tr_df,[400,400],'stim_cue'),axis=0)
    delay_sem = sem(align_tr_df(tr_df,[200,400],'delay_cue'),axis=0)
    go_sem = sem(align_tr_df(tr_df,[200,400],'go_cue'),axis=0)
    action_sem = sem(align_tr_df(tr_df,[200,1200],'action_onset'),axis=0)    
    
    
    erp_col = sns.color_palette()[0]
    fig, axs = plt.subplots(1,4,gridspec_kw={'width_ratios': [8,6,8,16]})
    plt.subplots_adjust(wspace=0.4)
    overall_min = np.min([np.min(l) for l in [stim_erp[:,elec],delay_erp[:,elec],go_erp[:,elec],action_erp[:,elec]]])-0.1
    overall_max = np.max([np.max(l) for l in [stim_erp[:,elec],delay_erp[:,elec],go_erp[:,elec],action_erp[:,elec]]])+0.1
    ax = axs[0]
    ax.plot(np.linspace(-1,1,stim_erp.shape[0]),stim_erp[:,elec],color=erp_col)
    ax.fill_between(np.linspace(-1,1,stim_erp.shape[0]),
                        stim_erp[:,elec] + stim_sem[:,elec],stim_erp[:,elec] - stim_sem[:,elec],alpha=0.1,color=erp_col)

                    
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Auditory stimulus')
    ax = axs[1]
    ax.plot(np.linspace(-0.5,1,delay_erp.shape[0]),delay_erp[:,elec],color=erp_col)
    ax.fill_between(np.linspace(-0.5,1,delay_erp.shape[0]),
                        delay_erp[:,elec] + delay_sem[:,elec],delay_erp[:,elec] - delay_sem[:,elec],alpha=0.1,color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Delay period')
    
    ax = axs[2]
    ax.plot(np.linspace(-0.5,1,go_erp.shape[0]),go_erp[:,elec],color=erp_col)
    ax.fill_between(np.linspace(-0.5,1,go_erp.shape[0]),
                        go_erp[:,elec] + go_sem[:,elec],go_erp[:,elec] - go_sem[:,elec],alpha=0.1,color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Go cue')
    
    ax = axs[3]
    ax.plot(np.linspace(-0.5,3,action_erp.shape[0]),action_erp[:,elec],color=erp_col)
    ax.fill_between(np.linspace(-0.5,3,action_erp.shape[0]),
                        action_erp[:,elec] + action_sem[:,elec],action_erp[:,elec] - action_sem[:,elec],alpha=0.1,color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Writing onset')
    
    return(fig,axs)


def plot_speech_writing_erps(tr_df_write,tr_df_speak,elec):
    aud_erp = np.mean(align_tr_df(tr_df_write,[400,400],'stim_cue'),axis=0)
    write_erp = np.mean(align_tr_df(tr_df_write,[400,1200],'action_onset'),axis=0)
    speak_erp = np.mean(align_tr_df(tr_df_speak,[400,400],'action_onset'),axis=0)
    read_erp = np.mean(align_tr_df(tr_df_speak,[400,400],'stim_cue'),axis=0)
    erp_col = sns.color_palette()[0]
    fig, axs = plt.subplots(1,4,gridspec_kw={'width_ratios': [8,16,8,8]})
    plt.subplots_adjust(wspace=0.4)
    overall_min = np.min([np.min(l) for l in [aud_erp[:,elec],write_erp[:,elec],speak_erp[:,elec],read_erp[:,elec]]])-0.1
    overall_max = np.max([np.max(l) for l in [aud_erp[:,elec],write_erp[:,elec],speak_erp[:,elec],read_erp[:,elec]]])+0.1
    ax = axs[0]
    ax.plot(np.linspace(-1,1,aud_erp.shape[0]),aud_erp[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Auditory stimulus')
    ax = axs[1]
    ax.plot(np.linspace(-1,3,write_erp.shape[0]),write_erp[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Writing')
    
    ax = axs[2]
    ax.plot(np.linspace(-1,1,read_erp.shape[0]),read_erp[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Reading stimulus')
    
    ax = axs[3]
    ax.plot(np.linspace(-1,1,speak_erp.shape[0]),speak_erp[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Speech')
    
    return(fig,axs)



def plot_2_erps(tr_df_write,tr_df_speak,key1,key2,elec,t_range=[400,800]):
    write_erp = np.mean(align_tr_df(tr_df_write,t_range,key1),axis=0)
    speak_erp = np.mean(align_tr_df(tr_df_speak,t_range,key2),axis=0)
    erp_col = sns.color_palette()[0]
    fig, axs = plt.subplots(1,2)
    plt.subplots_adjust(wspace=0.4)
    overall_min = np.min([np.min(l) for l in [write_erp[:,elec],speak_erp[:,elec]]])-0.1
    overall_max = np.max([np.max(l) for l in [write_erp[:,elec],speak_erp[:,elec]]])+0.1
    ax = axs[0]
    ax.plot(np.linspace(-t_range[0]/400.,t_range[1]/400.,speak_erp.shape[0]),speak_erp[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    
    ax.set_ylim([overall_min,overall_max])
    ax.set_title(f'{key1}')
    ax = axs[1]
    ax.plot(np.linspace(-t_range[0]/400.,t_range[1]/400.,write_erp.shape[0]),write_erp[:,elec],color=erp_col)  
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title(f'{key2}')
    

    
    return(fig,axs)





def plot_onset_and_vowels(tr_df_write,elec,before_after=[400,800]):
    write_erp_0 = np.mean(align_tr_df(tr_df_write,before_after,'action_onset'),axis=0)
    write_erp_1 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset'),axis=0)
    write_erp_2 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset_1'),axis=0)
    write_erp_3 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset_2'),axis=0)

    erp_col = sns.color_palette()[0]
    fig, axs = plt.subplots(1,4)
    plt.subplots_adjust(wspace=0.4)
    overall_min = np.min([np.min(l) for l in [write_erp_0[:,elec],write_erp_1[:,elec],write_erp_2[:,elec],write_erp_3[:,elec]]])-0.1
    overall_max = np.max([np.max(l) for l in [write_erp_0[:,elec],write_erp_1[:,elec],write_erp_2[:,elec],write_erp_3[:,elec]]])+0.1
    ax = axs[0]
    t = np.linspace(-before_after[0]/400.,before_after[1]/400.,
                        write_erp_0.shape[0])
    ax.plot(t,write_erp_0[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 0')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')    
    
    ax = axs[1]
    t = np.linspace(-before_after[0]/400.,before_after[1]/400.,
                        write_erp_1.shape[0])
    ax.plot(t,write_erp_1[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('(Vowel) Char 1')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')
    ax = axs[2]
    ax.plot(t,write_erp_2[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 2')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')

    ax = axs[3]
    ax.plot(t,write_erp_3[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 3')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')
    return(fig,axs)





def plot_vowel_aligns(tr_df_write,elec,before_after=[400,800]):
    write_erp_1 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset'),axis=0)
    write_erp_2 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset_1'),axis=0)
    write_erp_3 = np.mean(align_tr_df(tr_df_write,before_after,'vowel_onset_2'),axis=0)

    erp_col = sns.color_palette()[0]
    fig, axs = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.4)
    overall_min = np.min([np.min(l) for l in [write_erp_1[:,elec],write_erp_2[:,elec],write_erp_3[:,elec]]])-0.1
    overall_max = np.max([np.max(l) for l in [write_erp_1[:,elec],write_erp_2[:,elec],write_erp_3[:,elec]]])+0.1
    ax = axs[0]
    t = np.linspace(-before_after[0]/400.,before_after[1]/400.,
                        write_erp_1.shape[0])
    ax.plot(t,write_erp_1[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 1')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')
    ax = axs[1]
    ax.plot(t,write_erp_2[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 2')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')

    ax = axs[2]
    ax.plot(t,write_erp_3[:,elec],color=erp_col)
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Char 3')
    ax.set_ylabel('HGA (Z)')
    ax.set_xlabel('Time (s)')
    return(fig,axs)


def plot_mult_erps_one_elec(ax,aligned_conditions,elec,sems=(),t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),cur_cols=sns.color_palette()):
    for i,aligned in enumerate(aligned_conditions):
        if(len(aligned.shape)>2):
            aligned = np.mean(aligned,axis=0)
        ax.plot(np.linspace(t_lims[0],t_lims[1],aligned[:,elec].shape[0]),aligned[:,elec],color=cur_cols[i])
        if(len(sems)>0):
            ax.fill_between(np.linspace(t_lims[0],t_lims[1],aligned[:,elec].shape[0]),
                            aligned[:,elec] + sems[i][:,elec],aligned[:,elec] - sems[i][:,elec],alpha=0.1,label='_nolegend_')
    ax.set_title(str(elec+1))
    ax.set_xticks([t_lims[0],(t_lims[1]+t_lims[0])/2,t_lims[1]])
    ax.axhline(0,color='k',linestyle='--')
    #ax.axvline(fs*-1*t_lims[0],color='k',linestyle='--')
    ax.set_ylim(ylim)
    ax.set_xlim(t_lims)
    ax.set_yticks([ylim[0],0,1,ylim[1]])
    ax.axvline(0,color='k',linestyle='--')
    return(ax)

def plot_single_erp_mult_elec(ax,aligned,elecs,sems=(),t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),cur_cols=sns.color_palette()):
    if(len(aligned.shape)>2):
        aligned = np.mean(aligned,axis=0)
    for i,e in enumerate(elecs):
        ax.plot(np.linspace(t_lims[0],t_lims[1],aligned[:,e].shape[0]),aligned[:,e],color=cur_cols[i])
        if(len(sems)>0):
            ax.fill_between(np.linspace(t_lims[0],t_lims[1],aligned[:,e].shape[0]),
                            aligned[:,e] + sems[:,e],aligned[:,e] - sems[:,e],
                            alpha=0.1,label='_nolegend_')
    #ax.set_title(str(elec+1))
    ax.set_xticks([t_lims[0],(t_lims[1]+t_lims[0])/2,t_lims[1]])
    ax.axhline(0,color='k',linestyle='--')
    #ax.axvline(fs*-1*t_lims[0],color='k',linestyle='--')
    ax.set_ylim(ylim)
    ax.set_xlim(t_lims)
    ax.set_yticks([ylim[0],0,1,ylim[1]])
    ax.axvline(0,color='k',linestyle='--')
    return(ax)


def plot_mult_erps_one_elec(ax,aligned_conditions,elec,sems=(),t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),cur_cols=sns.color_palette()):
    for i,aligned in enumerate(aligned_conditions):
        if(len(aligned.shape)>2):
            aligned = np.mean(aligned,axis=0)
        ax.plot(np.linspace(t_lims[0],t_lims[1],aligned[:,elec].shape[0]),aligned[:,elec],color=cur_cols[i])
        if(len(sems)>0):
            ax.fill_between(np.linspace(t_lims[0],t_lims[1],aligned[:,elec].shape[0]),
                            aligned[:,elec] + sems[i][:,elec],aligned[:,elec] - sems[i][:,elec],alpha=0.1,label='_nolegend_')
    ax.set_title(str(elec+1))
    ax.set_xticks([t_lims[0],(t_lims[1]+t_lims[0])/2,t_lims[1]])
    ax.axhline(0,color='k',linestyle='--')
    #ax.axvline(fs*-1*t_lims[0],color='k',linestyle='--')
    ax.set_ylim(ylim)
    ax.set_xlim(t_lims)
    ax.set_yticks([ylim[0],0,1,ylim[1]])
    ax.axvline(0,color='k',linestyle='--')
    return(ax)












from scipy import stats
def plot_mult_erps_vowel_aligns(fig,axs,tr_df_write,elec,ids=[0,1,2],use_correct=True,before_after=np.array([400,800])):
    means = []
    sems = []
    if(use_correct):
        tr_df_write = tr_df_write[tr_df_write.correct]
    utt_ids = tr_df_write.utt_id.to_numpy()
    for cur_key in ['vowel_onset','vowel_onset_1','vowel_onset_2']:
        aligned = align_tr_df(tr_df_write,before_after,cur_key)
        aligned_conds = []
        for cur_id in ids:
             aligned_conds.append(aligned[utt_ids == cur_id])        
        means.append([np.mean(aligned_cond,axis=0) for aligned_cond in aligned_conds])
        sems.append([stats.sem(aligned_cond,axis=0) for aligned_cond in aligned_conds])
    
    plt.subplots_adjust(wspace=0.4)
    #cur_ylim = [np.min(np.array(means[0]
    before_after[0] = -1*before_after[0]
    axis_pad = 0.5
    cur_ylim = [np.min(np.array(means[0])[:,:,elec])-axis_pad,np.max(np.array(means[0])[:,:,elec])+axis_pad]
    cur_ylim = [round(y,1) for y in cur_ylim]
    plot_mult_erps_one_elec(axs[0],means[0],elec,sems=sems[0],t_lims=before_after/400,ylim=cur_ylim)

    cur_ylim = [np.min(np.array(means[1])[:,:,elec])-axis_pad,np.max(np.array(means[1])[:,:,elec])+axis_pad]
    cur_ylim = [round(y,1) for y in cur_ylim]
    plot_mult_erps_one_elec(axs[1],means[1],elec,sems=sems[1],t_lims=before_after/400,ylim=cur_ylim)

    cur_ylim = [np.min(np.array(means[2])[:,:,elec])-axis_pad,np.max(np.array(means[2])[:,:,elec])+axis_pad]
    cur_ylim = [round(y,1) for y in cur_ylim]
    plot_mult_erps_one_elec(axs[2],means[2],elec,sems=sems[2],t_lims=before_after/400,ylim=cur_ylim)
    return(fig,axs)



def plot_elec_all_phases_certain_trials(tr_df,elec,action_align='action_onset',trs=[0,1,2]):
    
    stim_erp = align_tr_df(tr_df,[400,400],'stim_cue')
    delay_erp = align_tr_df(tr_df,[200,400],'delay_cue')
    go_erp = align_tr_df(tr_df,[400,400],'go_cue')
    action_erp = align_tr_df(tr_df,[400,1200],action_align)
    y = tr_df.utt_id.values
    stim_erp = [np.mean(stim_erp[y==t],axis=0) for t in trs]
    delay_erp = [np.mean(delay_erp[y==t],axis=0) for t in trs]
    go_erp = [np.mean(go_erp[y==t],axis=0) for t in trs]
    action_erp = [np.mean(action_erp[y==t],axis=0) for t in trs]
   
    
    erp_col = sns.color_palette()
    fig, axs = plt.subplots(1,4,gridspec_kw={'width_ratios': [8,6,8,16]})
    plt.subplots_adjust(wspace=0.4)
    overall_mins = []
    overall_maxs = []
    for i in range(0,len(action_erp)):
        overall_mins.append(np.min([np.min(l) for l in [stim_erp[i][:,elec],delay_erp[i][:,elec],go_erp[i][:,elec],action_erp[i][:,elec]]])-0.1)
        overall_maxs.append(np.max([np.max(l) for l in [stim_erp[i][:,elec],delay_erp[i][:,elec],go_erp[i][:,elec],action_erp[i][:,elec]]])+0.1)
    overall_min = np.min(overall_mins)
    overall_max = np.max(overall_maxs)
    
    ax = axs[0]
    for i in range(len(action_erp)):
        ax.plot(np.linspace(-1,1,stim_erp[i].shape[0]),stim_erp[i][:,elec],color=erp_col[i])
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Auditory stimulus')
    ax = axs[1]
    for i in range(len(action_erp)):
        ax.plot(np.linspace(-0.5,1,delay_erp[i].shape[0]),delay_erp[i][:,elec],color=erp_col[i])
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Delay period')
    
    ax = axs[2]
    for i in range(len(action_erp)):
        ax.plot(np.linspace(-1,1,go_erp[i].shape[0]),go_erp[i][:,elec],color=erp_col[i])
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Go cue')
    
    ax = axs[3]
    for i in range(len(action_erp)):
        ax.plot(np.linspace(-1,3,action_erp[i].shape[0]),action_erp[i][:,elec],color=erp_col[i])    
    
    ax.axvline(0,color='k',linestyle='--')
    ax.axhline(0,color='k',linestyle='--')
    ax.set_ylim([overall_min,overall_max])
    ax.set_title('Writing onset')
    
    return(fig,axs)









def single_trial_raster(aligned,grid,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15)):
    #aligned = np.mean(aligned,axis=0)
    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax[r,c].imshow(aligned[:,:,grid[r,c]],vmin=ylim[0],vmax=ylim[1],cmap='binary')
            ax[r,c].set_title(str(grid[r,c]+1))
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])


def single_elec_trial_raster(aligned,elec,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),aspect='equal'):
    #aligned = np.mean(aligned,axis=0)
    fig,ax = plt.subplots(figsize=figsize)

    im = ax.imshow(aligned[:,:,elec],vmin=ylim[0],vmax=ylim[1],cmap='binary',aspect=aspect)
    ax.set_title(elec+1)
    ax.set_xticks([])
    ax.set_yticks([])
    return(fig,ax,im)

def single_elec_erp(aligned,elec,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(15,15),aspect='equal'):
    #aligned = np.mean(aligned,axis=0)
    from scipy.stats import sem
    fig,ax = plt.subplots(figsize=figsize)
    erp = np.mean(aligned[:,:,elec],axis=0)
    sem_erp = sem(aligned[:,:,elec],axis=0)
    t = np.linspace(t_lims[0],t_lims[1],erp.shape[0])
    ax.plot(t,erp,color='k')
    ax.fill_between(t,
                    erp + sem_erp,erp - sem_erp,alpha=0.1,color='k')
    ax.set_title(elec+1)
    return(fig,ax)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_erp_on_cont_condition(aligned,grid,cond_var,cutoff,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(20,20)):
    aligned_under = np.mean(aligned[cond_var<cutoff,:,:],axis=0)
    aligned_over = np.mean(aligned[cond_var>=cutoff,:,:],axis=0)
    
    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax[r,c].plot(aligned_under[:,grid[r,c]],color='b')
            ax[r,c].plot(aligned_over[:,grid[r,c]],color='r')
            
            ax[r,c].set_title(str(grid[r,c]+1))
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            ax[r,c].axhline(0)
            ax[r,c].axvline(fs*t_lims[0])
            ax[r,c].set_ylim(ylim)
            
            
def plot_erp_on_condition(aligned,grid,labels,targets,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(20,20)):
    erps = []
    colors = ['b','r','m','k','g']
    for target in targets:
        erps.append(np.mean(aligned[labels==target,:,:],axis=0))
    
    fig,ax = plt.subplots(grid.shape[0],grid.shape[1],figsize=figsize)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            for i,erp in enumerate(erps):
                ax[r,c].plot(erp[:,grid[r,c]],color=colors[i])
            
            ax[r,c].set_title(str(grid[r,c]+1))
            ax[r,c].set_xticks([])
            ax[r,c].set_yticks([])
            ax[r,c].axhline(0)
            ax[r,c].axvline(fs*t_lims[0])
            ax[r,c].set_ylim(ylim)
            
            
            
def plot_erp_on_condition_single_elec(aligned,grid,labels,targets,elec,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(10,10)):
    erps = []
    colors = ['b','r','m','k','g']
    t_ar = np.linspace(-1*t_lims[0],t_lims[1],aligned.shape[1])
    for target in targets:
        erps.append(np.mean(aligned[labels==target,:,:],axis=0))
    
    f = plt.figure(figsize=figsize)
    for i,erp in enumerate(erps):
        plt.plot(t_ar,erp[:,elec],color=colors[i])
    
            
    plt.gca().set_title(str(elec+1))
    plt.axhline(0)
    plt.axvline(0)
    plt.gca().set_ylim(ylim)
    
    
    
    
def plot_single_trials_on_condition_single_elec(aligned,grid,labels,target,elec,t_lims=[2,4],fs=400,ylim=[-.50,2],figsize=(10,10),smooth=-1):
    erps = []
    colors = ['b','r','m','k','g']
    t_ar = np.linspace(-1*t_lims[0],t_lims[1],aligned.shape[1])
    trials = aligned[labels==target,:,elec]
    f = plt.figure(figsize=figsize)
    for i in range(0,trials.shape[0]):
        if(smooth != -1):
            trials[i,:] = gaussian_filter1d(trials[i,:],smooth)
        plt.plot(t_ar,trials[i,:] + i*4)
        plt.axhline(i*4)

            
    plt.gca().set_title(str(elec+1))
    plt.axvline(0)
    #plt.gca().set_ylim(ylim)
    
    
def align_to_task_phase(all_hg,all_metadat,phase,t_align=[1,1],fs=400):
    all_aligned_hg = []
    all_labs = []
    for hg,meta in zip(all_hg,all_metadat):

        all_aligned_hg.append(align(hg,meta[:,phase],t_align,fs=fs))
        all_labs.append(meta[:,-1])

    all_aligned_hg = np.concatenate(all_aligned_hg,axis=0)
    all_labs = np.concatenate(all_labs,axis=0)
    return(all_aligned_hg,all_labs)