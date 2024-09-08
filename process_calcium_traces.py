#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from IPython.display import display, HTML
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter, freqz
from scipy import signal
import math
from scipy.stats import pearsonr
import scipy.integrate as integrate


# In[2]:


def filter_traces(df, order, critical_freq):
    filt_dfs = []
    for index, rows in df.iterrows():
        df_np = np.asarray(rows.dropna())

        # Create an order 3 lowpass butterworth filter:  
        b, a = signal.butter(order, critical_freq)   #  <=====HERE YOU CAN CHANGE THE FILTER FEATURE

        # Apply the filter to 1 trace of deltaF. Use lfilter_zi to choose the initial condition of the filter: 
        zi = signal.lfilter_zi(b, a)

        z, _ = signal.lfilter(b, a, df_np, zi=zi*df_np[0])

        #Apply the filter again, to have a result filtered at an order the same as filtfilt:
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

        #Use filtfilt to apply the filter to the entire array of traces:
        filt_dfs.append(signal.filtfilt(b, a, df_np))
    filt_df = pd.DataFrame(filt_dfs)
    filt_df.index = df.index
    
    return filt_df 


# In[3]:


def calculate_deltaF(ROI_df, fps, start, end):
    y = []
    smooth = lambda F, n=10: pd.Series(F).rolling(window=n, min_periods=1, center=True)
    for index, rows in ROI_df.iterrows():
        s = int(round(start*fps,0))
        e = int(round(end*fps,0))
        F0 = np.median(rows[s-1:e-1])
        y.append(F0)

    F0s = smooth(y, 20).median() # apply rolling window smothing 
    F0s = F0s.values.reshape(-1,1)
    deltaF = (ROI_df - F0s)/F0s
    deltaF_df = pd.DataFrame(deltaF)
    deltaF_df.index = ROI_df.index
    
    return deltaF_df


# In[4]:


def detect_events(data, ul, ll, min_width=6,):
    '''returns a list of pandas Series (indexed by the flattened data)
       containing the captured events. Events are captured using a
       histeresis threshold, composed from the MAD of the data.
       Events must be longer than min_width in order to be counted
    '''
    tmp = np.zeros_like(data, )

    tmp[data > ul] = 1

    edge = np.diff(np.r_[0, (data > ll), 0]).nonzero()[0]

    events = [gp for gp in np.array_split(data, edge)
                        if (gp.values > ul).sum() >= min_width]
    return events


# In[5]:


def get_ROI_event_metrics(ROI_deltaF, fps, dropped_ROIs, thresh, min_samples):
    df = ROI_deltaF.T

    ## First calculate threshold for each ROI ##
    std = np.asarray(df.iloc[int(round(1*fps,0)):int(round(2*fps,0))]).std()
    std_thresh = (std*thresh)

    # Pandas series of each event (with deltaF/indexes/column names from "all_ROIs") will be put in a list of series ##
    events = [detect_events(trace, std_thresh, std_thresh, min_width=min_samples) # use function detect events to find events 
                       for _, trace in df.items()]
    events_flat = []
    for element in events:
        if len(element) == 0:
            None
        elif len(element) == 1:
            events_flat.append(element[0])
        else:
            for el in element:
                events_flat.append(el)

    # Now place relevant event/behavior metrics from each pandas series into a dataframe of all the events for this ROI ## 
    ROIs_df = []
    for e in events_flat:
        ev_df = pd.DataFrame([list(e.name)+[e.max(), e.index[0]/fps, e.idxmax()/fps, e.index[-1]/fps, e.shape[0]/fps, integrate.simpson(e)]],
                              columns=list(ROI_deltaF.index.names)+['peak', 'ev_onset', 'peak_time', 'ev_offset', 'ev_duration', 'integral'])
        ev_df = ev_df.loc[ev_df['ev_onset'] > 0]
        ROIs_df.append(ev_df)  # append all ROIs 
    ROIs_df = pd.concat(ROIs_df) if len(ROIs_df) > 0 else pd.DataFrame([])
    if ROIs_df.shape[0] > 0:
        ROIs_df = ROIs_df[~ROIs_df['Unique_ROI'].isin(dropped_ROIs)]
    else:
        None
        
    return ROIs_df


# In[6]:


def calculate_ROI_correlations(deltaF, rthresh):
    cross_correlations = []
    self_correlations = []
    single_sessions = deltaF.groupby(deltaF.index.get_level_values('Unique session'))
    session_pairs = {}
    for session in deltaF.index.get_level_values('Unique session').unique():
        session_df = single_sessions.get_group(session)

        flattened_ROIs = []
        ROIs = session_df.groupby(session_df.index.get_level_values('Unique_ROI'))
        for ROI in session_df.index.get_level_values('Unique_ROI').unique():
            ROI_df = ROIs.get_group(ROI)
            ROI_flat = ROI_df.to_numpy().flatten()
            flattened_df = pd.DataFrame(ROI_flat, index=range(len(ROI_flat)), columns=[ROI]).T
            flattened_ROIs.append(flattened_df)
        flattened_ROIs = pd.concat(flattened_ROIs)
        df = flattened_ROIs.T.corr()
        rho = df.corr()
        pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
        p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
        corr_df = rho.round(2).astype(str) + p
        
        ROI_pairs = {}
        ROI_groups = corr_df.groupby(corr_df.index)
        for ROI in corr_df.index.unique():
            ROI_df = ROI_groups.get_group(ROI)
            for col_name, col in ROI_df.iteritems():
                c = col.values[0]
#                 if '*' in c:
                r = float(c.split('*')[0])
                if abs(r) > rthresh:
                    if ROI != col_name:
                        if ROI.split('_')[-2] != col_name.split('_')[-2]:
                            if ROI not in ROI_pairs.keys():
                                ROI_pairs[ROI] = {}
                                ROI_pairs[ROI][col_name] = r
                            else:
                                ROI_pairs[ROI][col_name] = r   
        session_pairs[session] = ROI_pairs

        for row1 in ['Red', 'Green']:
            for row2 in ['Red', 'Green']:
                filt1 = corr_df[corr_df.index.str.contains(row1)]
                filt2 = filt1.T[filt1.T.index.str.contains(row2)]

                for col_name, col in filt2.items():
                    sig_boutons = [float(x.split('*')[0]) for x in col.values]#in [x for x in col.values if '*' in x]]
                    nboutons = len(sig_boutons)

                    if nboutons > 0 :
                        proportion = nboutons / filt2.shape[0]
                        avg_r = sum(sig_boutons)/nboutons
                    else:
                        proportion = 0
                        avg_r = 0
                    metric_df = pd.DataFrame([avg_r, proportion], index=['avg_r', 'proportion'], columns=[col_name]).T
                    if row1 == row2:
                        self_correlations.append(metric_df)
                    else:
                        cross_correlations.append(metric_df)
    self_correlations = pd.concat(self_correlations)
    cross_correlations = pd.concat(cross_correlations)
    return {'intra-compartment':self_correlations, 'inter-compartment':cross_correlations, 'cross-pairs':session_pairs}


# In[ ]:


def multi_index_all_channels(aligned_data):
    all_F = pd.DataFrame([])
    all_pupil = pd.DataFrame([])
    all_licks = pd.DataFrame([])
    for protocol, protocol_dict in aligned_data.items():
        for date, date_dict in protocol_dict.items():
            for FOV, FOV_dict in date_dict.items():
                for channel, channel_dict in FOV_dict.items():
                    F_df = pd.DataFrame(channel_dict['F'])
                    pupil_df = pd.DataFrame(channel_dict['pupil'])
                    lick_df = pd.DataFrame([channel_dict['Trial licks'], channel_dict['Post-trial licks']], index=['Trial licks', 'Post-trial licks']).T
                    
                    for datatype, df in zip(['F', 'Pupil', 'Lick'],[F_df, pupil_df, lick_df]):
                        print(protocol, date, FOV, channel, datatype)
                        print('starting multiindexing...')
                        for i, (index, rows) in enumerate(df.iterrows()):
                            if datatype == 'Lick':
                                trial_df = rows.to_frame()
                            else:
                                trial_df = rows.item()
                                
                            if type(trial_df) == float:
                                if datatype == 'Pupil':
                                    trial_df = pd.DataFrame([np.nan]*360, index=list(range(1,361)), columns=[i]).T
                            else:
                                trial_df = trial_df
                                
                            nROIs = trial_df.shape[0]
                                
                            session = ('_').join([protocol, date, FOV])
                            levels = [[session]*nROIs, [protocol]*nROIs, [date]*nROIs, [FOV]*nROIs, [channel]*nROIs , [index]*nROIs, list(trial_df.index),
                                      #[channel_dict['pupil'][index]]*nROIs, 
                                      [channel_dict['Lick_time'][index]]*nROIs, #[channel_dict['Trial licks'][index]]*nROIs, [channel_dict['Post-trial licks'][index]]*nROIs,
                                      [channel_dict['pre_count'][index]]*nROIs, [channel_dict['ant_count'][index]]*nROIs, [channel_dict['rew_count'][index]]*nROIs,
                                      [channel_dict['Water'][index]]*nROIs, [channel_dict['Feedback_time'][index]]*nROIs, [channel_dict['Feedback_end'][index]]*nROIs,
                                      [channel_dict['stimtype'][index]]*nROIs, [channel_dict['outcome'][index]]*nROIs, [channel_dict['trialType'][index]]*nROIs, 
                                      [('_').join([session, str(x)]) for x in [index]*nROIs], [('_').join([session,channel, str(x)]) for x in list(trial_df.index)]]
    # #                     print(levels)
                            midx = pd.MultiIndex.from_arrays(levels, names=('Unique session','Protocol', 'Date', 'FOV', 'Channel', 'Trial', 'ROI',
                                                                            #'Pupil', 
                                                                            'Response latency', #'Trial licks', 'Post-trial licks',
                                                                            'Prestim licks', 'Anticipatory licks', 'Reward licks',
                                                                            'Water', 'Water_time', 'Timeout_end',
                                                                            'Stimulus', 'Outcome', 'Trial type',
                                                                            'Unique_trial','Unique_ROI'))
                            multi_df = trial_df.set_index(midx)
                                        
                            if datatype == 'F':
                                all_F = all_F.append(multi_df)
                            elif datatype == 'Pupil':
                                all_pupil = all_pupil.append(multi_df)
                            else:
                                all_licks = all_licks.append(multi_df.T)
                        print('multiindexing done')
                        print()
    return {'all_F':all_F, 'all_pupil':all_pupil, 'all_licks':all_licks}

