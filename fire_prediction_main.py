"""Use Machine Learning (ML) methods to surrogate traditional climate models, and make predictions.
Author = Fa Li

.. MIT License
..
.. Copyright (c) 2019 Fa Li
"""
from netCDF4 import Dataset
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
import os
import time
import math
from sklearn.metrics import explained_variance_score
import scipy.stats as stats
import torch
from torch.autograd import Variable
from torch import optim, nn
from sklearn import preprocessing
from scipy.stats import pearsonr
from sklearn.metrics import explained_variance_score
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import explained_variance_score
# from networks import IMVFullLSTM, IMVTensorLSTM
import warnings
warnings.filterwarnings('ignore')
from neural_network_module import MultiLayerFeedForwardNN
from fire_module import Fire_model
def data_loading(data_dir,file_variables):#
    """**Description:** this method is used to load the required climate variables

        Parameters
        ----------
        data_dir: str,default=""
            the directory of all files corresponding to required climate variables
        file_names: {}, default={}
            a dictionary mapping each file name to its corresponding variable name
        Return
        ----------
        output: list
            a list containing all variables used for further trainging and test through machine learning
        Examples
        --------
        data_dir='temp/'
        file_names={'file_name1.nc': 'temperature'}
        output=data_loading(data_dir,file_names)#outpiut is a list of the required variables (np.ndarray)
    """
    assert len(file_variables)>0 and type(file_variables)==dict
    output=[]
    for file in file_variables:
        file_path=data_dir+file
        nc_file = Dataset(file_path, 'r')
        var_names=file_variables[file]
        for var_name in var_names:
            var_value =nc_file[var_name][:]
            print(var_name,var_value.shape)
            if len(var_value.shape)==3:
                var_value=var_value[:,np.newaxis,:,:]
            output.append(np.array(var_value))
        nc_file.close()
    return output
def train(vars,target_idx,time_dim,lat_dim,lon_dim,repeat_types,regions,surrogate_var_idx,time_lead,teleconnection=False,include_target_history=False,para_path='region_para_path',save=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            vars: list,default=[]
                the list of all required climate variables
            target_idx: list, default=[]
                the list of indexes corresponding to predicted variable
            time_lag: int, default=1
                time delay of climate variables
            time_dim: int
                the corresponding dimension index of 'time' in climate variables
            lat_dim: int
                the corresponding dimension index of 'latitude' in climate variables
            lon_dim: int
                the corresponding dimension index of 'longitude' in climate variables
            para_path: str
                the directory for saving parameters of models during training
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            None: The return value will be designed for next processing
            Examples
            --------
            root_dir = ''
            file_variables = {'Temperature_2.nc': 'temperature'}
            time_lag = 2
            time_dim, lat_dim, lon_dim = 0, 2, 3
            vars = data_loading(root_dir, file_variables)
            output=data_loading(data_dir,file_names)
            target_idx = [0]
            train(vars, target_idx, time_lag, time_dim, lat_dim, lon_dim, para_path)

        """
    if not os.path.exists(para_path):
        os.makedirs(para_path)
    dims = vars[0].shape
    time_length, rows, cols = dims[time_dim] - time_lead, dims[lat_dim], dims[lon_dim]
    repeat_keys=repeat_types.keys()
    input_fea_num = 0
    for var_idx in range(len(vars)):
        input_fea_num += vars[var_idx].shape[1]
    if not include_target_history:
        input_fea_num-=1
    if teleconnection:
        tele_data=pd.read_csv(r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\teleconnection\tele_index.csv').values[:,1:]
        input_fea_num+=tele_data.shape[1]
        tele_data=tele_data.reshape(tele_data.shape[0],tele_data.shape[1],1,1)
        tele_data=np.tile(tele_data,(1,1,rows,cols))

    fea_data=np.full((time_length,input_fea_num,96,144),np.nan)
    target_data=np.full((time_length,time_lead,96,144),np.nan)
    fea_idx=0
    for var_idx in range(len(vars)):
        var = vars[var_idx]
        if var_idx==surrogate_var_idx:
            if var_idx==1:###yearly population density data:
                temp_var=np.nanmean(var,axis=0,keepdims=True)
                temp_var=np.repeat(temp_var,var.shape[0],axis=0)
            else:
                temp_var =var.reshape(-1,12,var.shape[1],var.shape[2],var.shape[3])
                year_num=temp_var.shape[0]
                temp_var=np.nanmean(temp_var,axis=0)
                temp_var=np.tile(temp_var,(year_num,1,1,1))
            var=temp_var
        if var_idx not in repeat_keys:
            if var_idx not in target_idx:
                for level in range(var.shape[1]):
                    fea_data[:,fea_idx]=var[(-time_length-time_lead):(-time_lead),level]
                    fea_idx+=1
            if (var_idx in target_idx) and include_target_history:
                for level in range(var.shape[1]):
                    fea_data[:, fea_idx] = var[(-time_length - time_lead):(-time_lead), level]
                    fea_idx += 1
            if var_idx in target_idx:
                for temp_lead in range(1,time_lead+1):
                    target_data[:,temp_lead-1] = var[(-time_length - time_lead+temp_lead):(-time_lead+temp_lead), 0] if temp_lead<time_lead else var[(-time_length):, 0]
        else:
            repeat_type = repeat_types[var_idx]
            if repeat_type == 'tile':
                for level in range(var.shape[1]):
                    var_temp=var[:,level]
                    var_temp=np.tile(var_temp,(16,1,1))
                    fea_data[:, fea_idx] = var_temp[(-time_length - time_lead):(-time_lead)]
                    fea_idx += 1
            elif repeat_type == 'repeat':
                for level in range(var.shape[1]):
                    var_temp=var[:,level]
                    var_temp=np.repeat(var_temp,12,axis=0)
                    fea_data[:, fea_idx] = var_temp[(-time_length - time_lead):(-time_lead)]
                    fea_idx += 1
    if teleconnection:
        fea_data[:, fea_idx:] = tele_data[(-time_length - time_lead):(-time_lead)]
    sequence_length=12
    cut_out_idx=fea_data.shape[0]%sequence_length
    fea_data=fea_data[cut_out_idx:]
    target_data=target_data[cut_out_idx:]
    input_idxs=np.array([idx for idx in range(sequence_length,fea_data.shape[0])])
    output_idxs=np.array([input_idxs[idx] for idx in range(sequence_length-1,len(input_idxs),sequence_length)])
    fea_data_final=np.zeros((fea_data.shape[0],sequence_length,fea_data.shape[1],fea_data.shape[2],fea_data.shape[3]))
    target_data_final=np.zeros(target_data.shape)
    for idx in range(sequence_length):
        input_idxs_temp=input_idxs-idx
        output_idxs_temp=output_idxs-idx
        input_temp=fea_data[input_idxs_temp]
        input_temp=input_temp.reshape((-1,sequence_length,fea_data.shape[1],fea_data.shape[2],fea_data.shape[3]))
        fea_data_final[output_idxs_temp]=input_temp[:]
        output_temp=target_data[output_idxs_temp]
        target_data_final[output_idxs_temp]=output_temp[:]
    fea_data_final=fea_data_final[sequence_length:]#time x seq_len x fea_num x lat x lon
    target_data_final=target_data_final[sequence_length:]#time x lat x lon
    print(target_data_final.shape)
    Y_predict=ML_mdoel_LeaveOneOut_Regions(X=fea_data_final,Y=target_data_final,model='lstm',region_mask=regions,save=True,time_lead=time_lead)
    Y_predict=ML_mdoel_CESM_future(X=fea_data_final,Y=target_data_final,model='lstm',region_mask=regions,save=True,time_lead=time_lead,surrogate_var_idx=surrogate_var_idx)
    Y_predict=ML_mdoel_LeaveOneOut_Regions_for_attention(X=fea_data_final,Y=target_data_final,model='lstm',region_mask=regions,save=True,time_lead=time_lead,teleconnection=teleconnection)
    # nc_save(Y_predict)
    # ML_mdoel_AttentionWeights(X=fea_data_final,Y=target_data_final,model='lstm',region_mask=regions,save=True,time_lead=time_lead,teleconnection=teleconnection)
    # ML_mdoel_Surrogate_driver(X=fea_data_final, Y=target_data_final, model='lstm', region_mask=regions, save=True,
    #                           time_lead=time_lead, teleconnection=teleconnection)

from netCDF4 import num2date, date2num
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
def nc_save(predicted_results,gfed_result_path):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    # root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\results\gfed-attn\0731"
    # nc_fid2 = Dataset(root_dir + '/gfed_attn.nc', 'w', format="NETCDF4")
    nc_fid2 = Dataset(gfed_result_path, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', predicted_results.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    burntArea = nc_fid2.createVariable('burntArea', "f8", ("time", "lat", "lon",))
    time_v.units = "days since 1993-08-01 00:00:00.0"
    time_v.calendar = "gregorian"
    dates = [datetime(1997, 1, 1) + relativedelta(months=+n) for n in range(predicted_results.shape[0])]
    time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    burntArea[:] = predicted_results[:]
    nc_fid2.close()
def nc_save_projection(predicted_results,gfed_result_path):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    # root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\results\gfed-attn\0731"
    # nc_fid2 = Dataset(root_dir + '/gfed_attn.nc', 'w', format="NETCDF4")
    nc_fid2 = Dataset(gfed_result_path, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', predicted_results.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    burntArea = nc_fid2.createVariable('burntArea', "f8", ("time", "lat", "lon",))
    time_v.units = "days since 1993-08-01 00:00:00.0"
    time_v.calendar = "gregorian"
    dates = [datetime(2015, 1, 1) + relativedelta(months=+n) for n in range(predicted_results.shape[0])]
    time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    burntArea[:] = predicted_results[:]
    nc_fid2.close()
def attn_weights_save_projection(nc_temporal_wts,nc_var_wts,gfed_result_path):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features"
    nc_fid2 = Dataset(gfed_result_path, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', 12)
    nc_fid2.createDimension('fea', nc_temporal_wts.shape[2])
    nc_fid2.createDimension('step', nc_temporal_wts.shape[1])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    fea_v = nc_fid2.createVariable("fea", "f8", ("fea",))
    step_v = nc_fid2.createVariable("step", "f8", ("step",))
    temporal_w = nc_fid2.createVariable('temporal', "f8", ("time","step","fea", "lat", "lon",))
    var_w = nc_fid2.createVariable('var', "f8", ("time","fea", "lat", "lon",))
    # time_v.units = "days since 1993-08-01 00:00:00.0"
    # time_v.calendar = "gregorian"
    # dates = [datetime(1997, 1, 1) + relativedelta(months=+n) for n in range(nc_temporal_wts.shape[0])]
    # time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    time_v[:]=range(1,13)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    fea_v[:]=range(nc_temporal_wts.shape[2])
    step_v[:]=range(nc_temporal_wts.shape[1],0,-1)
    nc_temporal_wts_anual=nc_temporal_wts.reshape(-1,12,nc_temporal_wts.shape[1],nc_temporal_wts.shape[2],nc_temporal_wts.shape[3],nc_temporal_wts.shape[4])
    nc_temporal_wts_anual=np.nanmean(nc_temporal_wts_anual,axis=0)

    nc_var_wts_anual=nc_var_wts.reshape(-1,12,nc_var_wts.shape[1],nc_var_wts.shape[2],nc_var_wts.shape[3])
    nc_var_wts_anual=np.nanmean(nc_var_wts_anual,axis=0)
    temporal_w[:] = nc_temporal_wts_anual[:]
    var_w[:]=nc_var_wts_anual[:]
    nc_fid2.close()
def attn_weights_save(nc_temporal_wts,nc_var_wts,gfed_result_path):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features"
    nc_fid2 = Dataset(gfed_result_path, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', 12)
    nc_fid2.createDimension('fea', nc_temporal_wts.shape[2])
    nc_fid2.createDimension('step', nc_temporal_wts.shape[1])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    fea_v = nc_fid2.createVariable("fea", "f8", ("fea",))
    step_v = nc_fid2.createVariable("step", "f8", ("step",))
    temporal_w = nc_fid2.createVariable('temporal', "f8", ("time","step","fea", "lat", "lon",))
    var_w = nc_fid2.createVariable('var', "f8", ("time","fea", "lat", "lon",))
    # time_v.units = "days since 1993-08-01 00:00:00.0"
    # time_v.calendar = "gregorian"
    # dates = [datetime(1997, 1, 1) + relativedelta(months=+n) for n in range(nc_temporal_wts.shape[0])]
    # time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    time_v[:]=range(1,13)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    fea_v[:]=range(nc_temporal_wts.shape[2])
    step_v[:]=range(nc_temporal_wts.shape[1],0,-1)
    nc_temporal_wts_anual=nc_temporal_wts.reshape(-1,12,nc_temporal_wts.shape[1],nc_temporal_wts.shape[2],nc_temporal_wts.shape[3],nc_temporal_wts.shape[4])
    nc_temporal_wts_anual=np.nanmean(nc_temporal_wts_anual,axis=0)

    nc_var_wts_anual=nc_var_wts.reshape(-1,12,nc_var_wts.shape[1],nc_var_wts.shape[2],nc_var_wts.shape[3])
    nc_var_wts_anual=np.nanmean(nc_var_wts_anual,axis=0)
    temporal_w[:] = nc_temporal_wts_anual[:]
    var_w[:]=nc_var_wts_anual[:]
    nc_fid2.close()

def attn_prediction_save(nc_predict,nc_predict_surrogate,gfed_result_path_pred):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features"
    nc_fid2 = Dataset(gfed_result_path_pred, 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', nc_predict.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    predict_raw = nc_fid2.createVariable('predict_raw', "f8", ("time","lat", "lon",))
    predict_surrogate = nc_fid2.createVariable('predict_surrogate', "f8", ("time","lat", "lon",))
    time_v.units = "days since 1993-08-01 00:00:00.0"
    time_v.calendar = "gregorian"
    dates = [datetime(2010, 12, 1) + relativedelta(months=-n) for n in range(nc_predict.shape[0])]
    dates=dates[::-1]
    time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    predict_raw[:]=nc_predict[:]
    predict_surrogate[:]=nc_predict_surrogate[:]
    nc_fid2.close()


def train_lstm(model, loss, optimizer, x_val, y_val,location_input,device):

    y = Variable(y_val, requires_grad=False).to(device)
    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx, alphas, betas = model(x_val)
    fx = fx.squeeze(1)
    output = loss(fx, y)
    output.backward()

    # Update parameters
    optimizer.step()

    return output.detach().cpu().item(),fx.detach().cpu().data.numpy(),y.detach().cpu().data.numpy()


def predict_lstm(model, x_val,device):
    x_val = x_val.to(device=device)

    output, alphas, betas = model.forward(x_val)
    # output = output.squeeze(1)
    return output.cpu().data.numpy(), alphas, betas#.argmax(axis=1)
def ML_mdoel_Surrogate_driver(X,Y,model,region_mask,time_lead=1,para_path='region_para_path',save=False,teleconnection=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    # train: https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/
    delta_time = 12
    input_dim = X.shape[2]
    # time_lead=Y.shape[1]
    sequence_len = 12
    epochs = 12
    batch_size = 32
    hidden_dim = 12  # 16
    output_dim = 1
    # Y=Y[:,-1]
    # Y=Y[:,np.newaxis,:,:]
    """
       batch_size, hidden_dim, emd_dim
    5: 64, 12,1
    4: 64, 12,1
    8:64,16,1
    9:64,16,1
    12:64,10,1 #0.91 0.79
    14: 64,16,1
    """
    target_region = 8
    # target_regions=[idx for idx in range(1,15)]#15
    target_regions = [9]  # 8,9,
    # target_regions = [5,14]
    # ['BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS','AUST']

    if model == 'lstm':
        for model_idx in target_regions:  # range(target_region,(1+target_region)):

            print('start training')

            all_indxs = [idx for idx in range(X.shape[0])]

            region_data = X[:, :, :, region_mask == model_idx]
            region_data = region_data.transpose((0, 1, 3, 2))
            region_data = region_data.reshape((-1, region_data.shape[3]))
            region_data_means = np.nanmean(region_data, axis=0)
            region_data_std = np.nanstd(region_data, axis=0)
            region_y = Y[:, :, region_mask == model_idx]  # .reshape(-1)
            region_y = region_y.transpose((2, 0, 1))
            region_y = region_y.reshape(-1, region_y.shape[2])
            region_y_mean = np.nanmean(region_y, axis=0)
            region_y_std = np.nanstd(region_y, axis=0)
            # region_y_mean = region_data_means[0]
            # region_y_std = region_data_std[0]

            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/gfed_attn_{time_lead}_tele.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_tele_{time_lead}_weights_attn_8.nc"#v1 for paper
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_attn_{time_lead}_tele.nc"
            gfed_result_path_pred = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_{time_lead}_{model_idx}_noRh.nc"
            if os.path.exists(gfed_result_path_pred) == False:
                if teleconnection:
                    nc_temporal_wts = np.full((X.shape[0],X.shape[1], X.shape[2]-3, X.shape[3], X.shape[4]), np.nan)
                    nc_var_wts = np.full((X.shape[0], X.shape[2]-3, X.shape[3], X.shape[4]), np.nan)
                else:
                    # nc_temporal_wts = np.full((X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]), np.nan)
                    # nc_var_wts = np.full((X.shape[0], X.shape[2], X.shape[3], X.shape[4]), np.nan)
                    nc_predict=np.full((X.shape[0],X.shape[3], X.shape[4]), np.nan)
                    nc_predict_surrogate=np.full((X.shape[0], X.shape[3], X.shape[4]), np.nan)
            else:
                data = Dataset(gfed_result_path_pred, 'r')
                # nc_var_wts = data.variables['temporal'][:]
                # nc_temporal_wts = data.variables['var'][:]
                nc_predict = data.variables['predict_raw'][:]
                nc_predict_surrogate =data.variables['predict_surrogate'][:]
                data.close()
            # gfed_result_path = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\gfed_attn_weights.nc"


            for year in range(1,2):#1,15: 2010-1997  #start from X.shape[0]-delta_time to get well-trained trees

                test_idxs = all_indxs

                # para_path = 'K:\lifa\lbnl\wildfire\code\wildfire_prediction/results\gfed-attn/0405\gfed_attention/' + str(model_idx) + '_' + str(2010-year+1) + '.para'
                para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vector.para'
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                device = "cuda"
                if os.path.exists(para_path) == False:
                    # regressor = Attention_LSTMNet(input_dim , hidden_dim, output_dim, emd_input_dim,
                    #                               emd_out_dim, sequence_len)#+ emd_out_dim
                    # regressor = IMVFullLSTM(input_dim, output_dim, hidden_dim).cuda()
                    regressor = IMV_fire_model(input_dim, time_lead, time_lead, hidden_dim,teleconnection=teleconnection).cuda()  #
                    # regressor =IMV_fire_model(input_dim, 1, hidden_dim, include_burn_history=False,tele=True,tele_bottleneck=True).cuda()
                else:
                    # regressor = Attention_LSTMNet(input_dim , hidden_dim, output_dim, emd_input_dim,
                    #                               emd_out_dim, sequence_len)#+ emd_out_dim
                    # regressor = IMVFullLSTM(input_dim, output_dim, hidden_dim).cuda()
                    regressor = IMV_fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=teleconnection).cuda()  #
                    regressor.load_state_dict(torch.load(para_path))
                regressor.to(device)
                model = regressor

                ###################################################################################
                #tain process
                #####################################################################################
                #test

                model.load_state_dict(torch.load(para_path))

                with torch.no_grad():
                    # weights_result=np.ones((1,sequence_len,input_dim + emd_out_dim))
                    model.eval()
                    for test_idx in range(2):
                        y_true = []
                        y_region_predict = []
                        for lat in range(X.shape[3]):
                            for lon in range(X.shape[4]):
                                if region_mask[lat,lon]==model_idx:
                                    sample=X[test_idxs,:,:,lat,lon]
                                    sample = np.array((sample - region_data_means) / (region_data_std+pow(10,-8)))
                                    if test_idx ==1:
                                        sample[:,:,10]=np.random.randn(sample.shape[0],sample.shape[1])#9 rain 10 rh 0 ltn
                                    sample = torch.from_numpy(sample).float()
                                    predicted_y, alphas, betas = predict_lstm(model, sample, device)
                                    if len(predicted_y.shape) == 2:
                                        predicted_y = predicted_y[:, -1]
                                    predicted_y = predicted_y * region_y_std[-1] + region_y_mean[-1]
                                    predicted_y[predicted_y < 0] = 0
                                    if test_idx ==1:
                                        nc_predict_surrogate[test_idxs, lat, lon]=predicted_y
                                    else:
                                        nc_predict[test_idxs, lat, lon] = predicted_y
                                        # nc_temporal_wts[test_idxs,:,:, lat, lon] = alphas[...,-1].detach().cpu().data.numpy()
                                        # nc_var_wts[test_idxs, :,  lat, lon] = betas[..., -1].detach().cpu().data.numpy()
                                    y_region_predict.extend(predicted_y)
                                    y_true.extend(Y[test_idxs,-1, lat, lon])
                        print("test:",model_idx,2011-year,explained_variance_score(y_true, y_region_predict))
        # attn_weights_save(nc_temporal_wts,nc_var_wts,gfed_result_path)
        attn_prediction_save(nc_predict,nc_predict_surrogate,gfed_result_path_pred)
def ML_mdoel_AttentionWeights(X,Y,model,region_mask,time_lead=1,para_path='region_para_path',save=False,teleconnection=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    # train: https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/
    delta_time = 12
    input_dim = X.shape[2]
    # time_lead=Y.shape[1]
    sequence_len = 12
    epochs = 12
    batch_size = 32
    hidden_dim = 12  # 16
    output_dim = 1
    # Y=Y[:,-1]
    # Y=Y[:,np.newaxis,:,:]
    """
       batch_size, hidden_dim, emd_dim
    5: 64, 12,1
    4: 64, 12,1
    8:64,16,1
    9:64,16,1
    12:64,10,1 #0.91 0.79
    14: 64,16,1
    """
    target_region = 8
    # target_regions=[idx for idx in range(1,15)]#15
    target_regions = [5,8,9]  # 8,9,
    # target_regions = [5,14]
    # ['BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS','AUST']

    if model == 'lstm':
        for model_idx in target_regions:  # range(target_region,(1+target_region)):

            print('start training')

            all_indxs = [idx for idx in range(X.shape[0])]

            region_data = X[:, :, :, region_mask == model_idx]
            region_data = region_data.transpose((0, 1, 3, 2))
            region_data = region_data.reshape((-1, region_data.shape[3]))
            region_data_means = np.nanmean(region_data, axis=0)
            region_data_std = np.nanstd(region_data, axis=0)
            region_y = Y[:, :, region_mask == model_idx]  # .reshape(-1)
            region_y = region_y.transpose((2, 0, 1))
            region_y = region_y.reshape(-1, region_y.shape[2])
            region_y_mean = np.nanmean(region_y, axis=0)
            region_y_std = np.nanstd(region_y, axis=0)
            # region_y_mean = region_data_means[0]
            # region_y_std = region_data_std[0]
            region_data_means = np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_means_x.npy')
            region_data_std = np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_stds_x.npy')
            region_y_mean = np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_means_y.npy')
            region_y_std = np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_stds_y.npy')

            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/gfed_attn_{time_lead}_tele.nc"
            gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_tele_{time_lead}_weights_attn_8.nc"#rh for paper
            gfed_result_path = f"D:\lbnl\phd\dataset\CESM_585\T62/gfed_imv_tele_{time_lead}_weights_attn_{model_idx}_vpd_replace_rh_noLTN.nc"# vpd replace rh
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_attn_{time_lead}_tele.nc"
            if os.path.exists(gfed_result_path) == False:
                if teleconnection:
                    nc_temporal_wts = np.full((X.shape[0],X.shape[1], X.shape[2]-3, X.shape[3], X.shape[4]), np.nan)
                    nc_var_wts = np.full((X.shape[0], X.shape[2]-3, X.shape[3], X.shape[4]), np.nan)
                else:
                    nc_temporal_wts = np.full((X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]), np.nan)
                    nc_var_wts = np.full((X.shape[0], X.shape[2], X.shape[3], X.shape[4]), np.nan)
            else:
                data = Dataset(gfed_result_path, 'r')
                nc_var_wts = data.variables['temporal'][:]
                nc_temporal_wts = data.variables['var'][:]
                data.close()
            # gfed_result_path = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\gfed_attn_weights.nc"


            for year in range(1,2):#1,15: 2010-1997  #start from X.shape[0]-delta_time to get well-trained trees

                test_idxs = all_indxs

                # para_path = 'K:\lifa\lbnl\wildfire\code\wildfire_prediction/results\gfed-attn/0405\gfed_attention/' + str(model_idx) + '_' + str(2010-year+1) + '.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vector.para'#rh for paper
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vpd_replace_rh.para'  # rh for paper
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vpd_replace_rh_noLTN.para'
                para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vpd_replace_rh_noLTN.para'
                device = "cuda"
                if os.path.exists(para_path) == False:
                    # regressor = Attention_LSTMNet(input_dim , hidden_dim, output_dim, emd_input_dim,
                    #                               emd_out_dim, sequence_len)#+ emd_out_dim
                    # regressor = IMVFullLSTM(input_dim, output_dim, hidden_dim).cuda()
                    regressor = IMV_fire_model(input_dim, time_lead, time_lead, hidden_dim,teleconnection=teleconnection).cuda()  #rh for paper
                    # regressor =IMV_fire_model(input_dim, 1, hidden_dim, include_burn_history=False,tele=True,tele_bottleneck=True).cuda()
                else:
                    # regressor = Attention_LSTMNet(input_dim , hidden_dim, output_dim, emd_input_dim,
                    #                               emd_out_dim, sequence_len)#+ emd_out_dim
                    # regressor = IMVFullLSTM(input_dim, output_dim, hidden_dim).cuda()
                    regressor = IMV_fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=teleconnection).cuda()  # rh for paper
                    regressor.load_state_dict(torch.load(para_path))
                regressor.to(device)
                model = regressor

                ###################################################################################
                #tain process
                #####################################################################################
                #test

                model.load_state_dict(torch.load(para_path))

                with torch.no_grad():
                    # weights_result=np.ones((1,sequence_len,input_dim + emd_out_dim))
                    model.eval()
                    # y_true = []
                    # y_region_predict = []
                    for lat in range(X.shape[3]):
                        for lon in range(X.shape[4]):
                            if region_mask[lat,lon]==model_idx:
                                sample=X[test_idxs,:,:,lat,lon]
                                sample = np.array((sample - region_data_means) / (region_data_std+pow(10,-8)))
                                sample = torch.from_numpy(sample).float()
                                predicted_y, alphas, betas = predict_lstm(model, sample, device)
                                if len(predicted_y.shape) == 2:
                                    predicted_y = predicted_y[:, -1]
                                predicted_y = predicted_y * region_y_std[-1] + region_y_mean[-1]
                                predicted_y[predicted_y < 0] = 0
                                nc_temporal_wts[test_idxs,:,:, lat, lon] = alphas[...,-1].detach().cpu().data.numpy()
                                nc_var_wts[test_idxs, :,  lat, lon] = betas[..., -1].detach().cpu().data.numpy()
                    #             y_region_predict.extend(predicted_y)
                    #             y_true.extend(Y[test_idxs,-1, lat, lon])
                    # print("test:",model_idx,2011-year,explained_variance_score(y_true, y_region_predict))
            attn_weights_save_projection(nc_temporal_wts,nc_var_wts,gfed_result_path)
def ML_mdoel_CESM_future(X,Y,model,region_mask,time_lead=1,surrogate_var_idx=1,para_path='region_para_path',save=False):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    #train: https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/
    delta_time=12
    input_dim=X.shape[2]
    output_dim=Y.shape[1]
    sequence_len=12
    epochs = 12 # nhaf and shaf epochs=12; 5: 4epoches for attention weight
    batch_size = 32
    hidden_dim = 12# 16
    """
       batch_size, hidden_dim, emd_dim
    5: 64, 12,1
    4: 64, 12,1
    8:64,16,1
    9:64,16,1
    12:64,10,1 #0.91 0.79
    14: 64,16,1
    """
    target_region=8
    # target_regions=[idx for idx in range(1,15)]#15
    target_regions=[8,9,5]#8,9,
    # target_regions = [5,14]
    #['BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS','AUST']
    if model=='lstm':
        for model_idx in target_regions:#range(target_region,(1+target_region)):

            print('start training')

            all_indxs=[idx for idx in range(X.shape[0])]

            region_data = X[:, :,:, region_mask == model_idx]
            region_data = region_data.transpose((0,1,3, 2))
            region_data = region_data.reshape((-1, region_data.shape[3]))
            region_data_means = np.nanmean(region_data, axis=0)
            region_data_std = np.nanstd(region_data, axis=0)
            region_y=Y[:,:,region_mask == model_idx]#.reshape(-1)
            region_y = region_y.transpose((2, 0, 1))
            region_y = region_y.reshape(-1,region_y.shape[2])
            region_y_mean=np.nanmean(region_y, axis=0)
            region_y_std=np.nanstd(region_y, axis=0)

            region_data_means=np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_means_x.npy')
            region_data_std=np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_stds_x.npy')
            region_y_mean=np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_means_y.npy')
            region_y_std=np.load(f'D:\lbnl\phd\dataset\CESM_585\T62/region_stds_y.npy')
            # region_y_mean = region_data_means[0]
            # region_y_std = region_data_std[0]

            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/gfed_attn_{time_lead}_tele.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_{time_lead}_notvector.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_attn_{time_lead}_tele.nc"
            gfed_result_path = f"D:\lbnl\phd\dataset\CESM_585\T62/surrogate_climatology/gfed_imv_{time_lead}_2015-2100_varidx{surrogate_var_idx}.nc"
            if os.path.exists(gfed_result_path) == False:
                nc_y = np.full((X.shape[0], X.shape[3], X.shape[4]), np.nan)
            else:
                data = Dataset(gfed_result_path, 'r')
                nc_y = data.variables['burntArea'][:]
                data.close()
            target_years=range(1,15)#[14,1,2,5] #15
            target_years = [1]
            for year in target_years:# range(1,15) 1,15: 2010-1997  #start from X.shape[0]-delta_time to get well-trained trees
                start_date = X.shape[0] - year * delta_time
                end_date = start_date + delta_time
                test_idxs=all_indxs

                para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vpd_replace_rh_noLTN.para'
                device = "cuda"#"cpu"
                if os.path.exists(para_path) == False:
                    regressor = Fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=False).cuda()  #

                else:
                    regressor = Fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=False).cuda()  #

                    regressor.load_state_dict(torch.load(para_path))
                # regressor.to(device)
                model = regressor

                ###################################################################################
                #test process
                ########################

                model.load_state_dict(torch.load(para_path))
                model.eval()
                with torch.no_grad():
                    # weights_result=np.ones((1,sequence_len,input_dim + emd_out_dim))
                    y_true = []
                    y_region_predict = []
                    for lat in range(X.shape[3]):
                        for lon in range(X.shape[4]):
                            if region_mask[lat,lon]==model_idx:
                                sample=X[test_idxs,:,:,lat,lon]
                                sample = np.array((sample - region_data_means) / (region_data_std+ pow(10, -8)))
                                sample = torch.from_numpy(sample).float()
                                predicted_y,  alphas, betas = predict_lstm(model, sample, device)
                                # if len(predicted_y.shape)==2:
                                predicted_y=predicted_y[:,-1]
                                predicted_y = predicted_y * region_y_std[-1] + region_y_mean[-1]
                                predicted_y[predicted_y<0]=0
                                nc_y[test_idxs, lat, lon] = predicted_y
                    #             y_region_predict.extend(predicted_y)
                    #             y_true.extend(Y[test_idxs,-1, lat, lon])
                    # print("test:",model_idx,2011-year,explained_variance_score(y_true, y_region_predict))
                    nc_save_projection(nc_y,gfed_result_path)
                    print('done!')
                    # plt.subplot(121)
                    # plt.plot(range(len(y_region_predict)),y_true,label='y_true')
                    # plt.plot(range(len(y_region_predict)), y_region_predict,label='y_region_predict')
                    # plt.legend()
                    #
                    # plt.subplot(122)
                    # plt.scatter(y_true,y_region_predict)
                    # plt.show()


    return nc_y
def ML_mdoel_LeaveOneOut_Regions(X,Y,model,region_mask,time_lead=1,para_path='region_para_path',save=False):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    #train: https://blog.ailemon.me/2019/02/26/solution-to-loss-doesnt-drop-in-nn-train/
    delta_time=12
    input_dim=X.shape[2]
    output_dim=Y.shape[1]
    sequence_len=12
    epochs = 12 # nhaf and shaf epochs=12; 5: 4epoches for attention weight
    batch_size = 32
    hidden_dim = 12# 16
    """
       batch_size, hidden_dim, emd_dim
    5: 64, 12,1
    4: 64, 12,1
    8:64,16,1
    9:64,16,1
    12:64,10,1 #0.91 0.79
    14: 64,16,1
    """
    target_region=8
    # target_regions=[idx for idx in range(1,15)]#15
    target_regions=[5,8,9]#8,9,
    # target_regions = [5,14]
    #['BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS','AUST']
    if model=='lstm':
        for model_idx in target_regions:#range(target_region,(1+target_region)):

            print('start training')

            all_indxs=[idx for idx in range(X.shape[0])]

            region_data = X[:, :,:, region_mask == model_idx]
            region_data = region_data.transpose((0,1,3, 2))
            region_data = region_data.reshape((-1, region_data.shape[3]))
            region_data_means = np.nanmean(region_data, axis=0)
            region_data_std = np.nanstd(region_data, axis=0)
            region_y=Y[:,:,region_mask == model_idx]#.reshape(-1)
            region_y = region_y.transpose((2, 0, 1))
            region_y = region_y.reshape(-1,region_y.shape[2])
            region_y_mean=np.nanmean(region_y, axis=0)
            region_y_std=np.nanstd(region_y, axis=0)

            np.save(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_means_x.npy',region_data_means)
            np.save(f'D:\lbnl\phd\dataset\CESM_585\T62/region_data_stds_x.npy', region_data_std)
            np.save(f'D:\lbnl\phd\dataset\CESM_585\T62/region_means_y.npy', region_y_mean)
            np.save(f'D:\lbnl\phd\dataset\CESM_585\T62/region_stds_y.npy', region_y_std)
            # region_y_mean = region_data_means[0]
            # region_y_std = region_data_std[0]

            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/gfed_attn_{time_lead}_tele.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_{time_lead}_notvector.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_attn_{time_lead}_tele.nc"
            gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_{time_lead}_vpd_replace_rh_noLTN.nc"
            if os.path.exists(gfed_result_path) == False:
                nc_y = np.full((X.shape[0], X.shape[3], X.shape[4]), np.nan)
            else:
                data = Dataset(gfed_result_path, 'r')
                nc_y = data.variables['burntArea'][:]
                data.close()
            target_years=range(1,15)#[14,1,2,5] #15
            target_years = [1]
            for year in target_years:# range(1,15) 1,15: 2010-1997  #start from X.shape[0]-delta_time to get well-trained trees
                start_date = X.shape[0] - year * delta_time
                end_date = start_date + delta_time
                test_idxs = [idx for idx in range(start_date, end_date)]
                validate_start=start_date - 1 * delta_time
                validate_end=end_date+1 * delta_time
                if validate_start<0:
                    validate_start=end_date
                    validate_idxs = [idx for idx in range(validate_start,validate_start + 2 * delta_time)]
                if validate_end>X.shape[0]:
                    validate_start = start_date - 2 * delta_time
                    validate_idxs = [idx for idx in range(validate_start, validate_start + 2 * delta_time)]
                if validate_start>=0 and validate_end<=X.shape[0]:
                    validate_idxs_part1=[idx for idx in range(validate_start, validate_start + 1 * delta_time)]
                    validate_idxs_part2 = [idx for idx in range(validate_end - 1 * delta_time,validate_end )]
                    validate_idxs=list(set(validate_idxs_part1).union(set(validate_idxs_part2)))
                train_validate_idxs = list(set(all_indxs).difference(set(test_idxs)))
                train_idxs = list(set(train_validate_idxs).difference(set(validate_idxs)))
                # train_idxs=all_indxs#attn weights in 9 region, but not used in 5,8, region as overfitting

                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/{model_idx}_{2010-year+1}_lead{time_lead}_tele.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_tele.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_tele.para'
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vpd_replace_rh_noLTN.para'
                device = "cuda"#"cpu"
                if os.path.exists(para_path) == False:
                    regressor = Fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=False).cuda()  #

                else:
                    regressor = Fire_model(input_dim, time_lead, time_lead, hidden_dim,
                                               teleconnection=False).cuda()  #

                    regressor.load_state_dict(torch.load(para_path))
                # regressor.to(device)
                model = regressor

                ###################################################################################
                X_merge_input=X[train_idxs]
                X_merge_input=X_merge_input[:,:,:,region_mask==model_idx]
                X_merge_input=X_merge_input.transpose((3,0,1, 2))
                X_merge_input = X_merge_input.reshape((-1,X_merge_input.shape[2],X_merge_input.shape[3]))
                X_merge_input=(X_merge_input-region_data_means)/(region_data_std+pow(10,-8))
                print(np.sum(np.isnan(X_merge_input.reshape(-1))))
                Y_merge_input=Y[train_idxs]
                Y_merge_input=Y_merge_input[:,:,region_mask==model_idx]
                Y_merge_input = Y_merge_input.transpose((2, 0,1))
                Y_merge_input =Y_merge_input.reshape(-1,Y_merge_input.shape[2])
                Y_merge_input=(Y_merge_input-region_y_mean)/(region_y_std+pow(10,-8))
                print('start training of region:',model_idx)

                loss = torch.nn.MSELoss(reduction='elementwise_mean')
                lr = 0.01
                # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=pow(10, -3))#for attention weight 5: pow(10,-4), 9: pow(10,-4),8: 8*pow(10,-4)
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
                X_merge_input = torch.from_numpy(X_merge_input).float()
                Y_merge_input = torch.from_numpy(Y_merge_input).float()
                dataset = TensorDataset(X_merge_input, Y_merge_input)
                loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
                batch_num=X_merge_input.shape[0]//batch_size

                batch_cnt=0

                delta_lr=int(math.floor(batch_num*epochs/15))
                with torch.no_grad():
                    model.eval()
                    explained_vars,mse,r=evaluate(model,X,validate_idxs,region_mask,model_idx,sequence_len,region_data_means,region_data_std,region_y_std,region_y_mean,device,Y,loss)
                    print('eva,mse', explained_vars, mse)
                    torch.save(model.state_dict(), para_path)
                for epo_idx in range(epochs):
                    costs = []
                    for batch_idx, (X_batch_input, Y_batch_input) in enumerate(loader):
                        model.train()
                        X_batch_input=X_batch_input.to(device=device)
                        ##############################################
                        #start train
                        y = Variable(Y_batch_input, requires_grad=False).to(device)
                        # Reset gradient
                        optimizer.zero_grad()

                        # Forward
                        fx, alphas, betas = model(X_batch_input)
                        # fx = fx.squeeze(1)
                        output = loss(fx, y)
                        # output.detach().cpu().item()
                        output.backward()

                        # Update parameters
                        optimizer.step()

                        cost, y_train_pred, y_train_true =output.detach().cpu().item(),fx.detach().cpu().data.numpy(),y.detach().cpu().data.numpy()
                        # print(cost)
                        costs.append(cost)
                        # # cost,y_train_pred,y_train_true=train_lstm(model, loss, optimizer, X_batch_input, Y_batch_input,location_input,device)#X_merge_input: sequence_len x batch_size x feature_num
                        batch_cnt+=1
                        if batch_cnt%delta_lr==0:
                            # lr*=0.8
                            # print('new lr:',lr)
                            scheduler.step()

                    if (epo_idx+1)%1==0:
                        model.eval()
                        print(np.mean(costs))
                        with torch.no_grad():
                            temp_explained_vars,temp_mse,temp_r=evaluate(model,X,validate_idxs,region_mask,model_idx,sequence_len,region_data_means,region_data_std,region_y_std,region_y_mean,device,Y,loss)
                            print('eva,mse', temp_explained_vars,temp_mse)
                            if temp_explained_vars>explained_vars and temp_mse<mse:
                                explained_vars=temp_explained_vars
                                mse=temp_mse
                                torch.save(model.state_dict(), para_path)
                        model.train()
                        # plt.plot(range(len(costs)),costs)
                        # plt.show()
                print('evaluate:',explained_vars)
                # tain process
                #####################################################################################

                model.load_state_dict(torch.load(para_path))
                model.eval()
                with torch.no_grad():
                    # weights_result=np.ones((1,sequence_len,input_dim + emd_out_dim))
                    y_true = []
                    y_region_predict = []
                    for lat in range(X.shape[3]):
                        for lon in range(X.shape[4]):
                            if region_mask[lat,lon]==model_idx:
                                sample=X[test_idxs,:,:,lat,lon]
                                sample = np.array((sample - region_data_means) / (region_data_std+ pow(10, -8)))
                                sample = torch.from_numpy(sample).float()
                                predicted_y,  alphas, betas = predict_lstm(model, sample, device)
                                # if len(predicted_y.shape)==2:
                                predicted_y=predicted_y[:,-1]
                                predicted_y = predicted_y * region_y_std[-1] + region_y_mean[-1]
                                predicted_y[predicted_y<0]=0
                                nc_y[test_idxs, lat, lon] = predicted_y
                                y_region_predict.extend(predicted_y)
                                y_true.extend(Y[test_idxs,-1, lat, lon])
                    print("test:",model_idx,2011-year,explained_variance_score(y_true, y_region_predict))
                    nc_save(nc_y,gfed_result_path)
                    # plt.subplot(121)
                    # plt.plot(range(len(y_region_predict)),y_true,label='y_true')
                    # plt.plot(range(len(y_region_predict)), y_region_predict,label='y_region_predict')
                    # plt.legend()
                    #
                    # plt.subplot(122)
                    # plt.scatter(y_true,y_region_predict)
                    # plt.show()


    return nc_y
def weighted_mse_loss(input,target,weights):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    out = weights *(input-target)**2
    loss = out.mean()
    return loss
def ML_mdoel_LeaveOneOut_Regions_for_attention(X,Y,model,region_mask,time_lead=1,para_path='region_para_path',save=False,teleconnection=True):
    """**Description:** this method is used to load the required climate variables

            Parameters
            ----------
            X: ndarray
                input values to train the model
            Y: ndarray
                output values corresponding to X
            model: str,default='decision tree'
                model name used for training, and more models will be added in future work
            X_test: ndarray
                input values to test the model
            para_path: str
                the directory for saving parameters of models during training
            file_name:
                the file name of the saved parameter file
            save: bool, default=True
                whether to save parameters of models during training

            Return
            ----------
            output: ndarray
                the predicted values corresponding to X_test
            Examples
            --------

        """
    delta_time = 12
    input_dim = X.shape[2]
    # time_lead=Y.shape[1]
    sequence_len = 12
    epochs = 12
    batch_size = 32
    hidden_dim = 8 # 16
    output_dim = 3
    # Y=Y[:,-1]
    # Y=Y[:,np.newaxis,:,:]
    """
       batch_size, hidden_dim, emd_dim
    5: 64, 12,1
    4: 64, 12,1
    8:64,16,1
    9:64,16,1
    12:64,10,1 #0.91 0.79
    14: 64,16,1
    """
    target_region = 8
    # target_regions=[idx for idx in range(1,15)]#15
    target_regions = [8]  # 8,9,
    # target_regions = [5,14]
    # ['BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS','AUST']
    if model == 'lstm':
        for model_idx in target_regions:  # range(target_region,(1+target_region)):

            print('start training')

            all_indxs = [idx for idx in range(X.shape[0])]

            region_data = X[:, :, :, region_mask == model_idx]
            region_data = region_data.transpose((0, 1, 3, 2))
            region_data = region_data.reshape((-1, region_data.shape[3]))
            region_data_means = np.nanmean(region_data, axis=0)
            region_data_std = np.nanstd(region_data, axis=0)
            region_y = Y[:, :, region_mask == model_idx]  # .reshape(-1)
            region_y = region_y.transpose((2, 0, 1))
            region_y = region_y.reshape(-1, region_y.shape[2])
            region_y_mean = np.nanmean(region_y, axis=0)
            region_y_std = np.nanstd(region_y, axis=0)
            # region_y_mean = region_data_means[0]
            # region_y_std = region_data_std[0]

            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/gfed_attn_{time_lead}_tele.nc"
            gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_imv_tele_{time_lead}_attn.nc"
            # gfed_result_path = f"K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/gfed_attn_{time_lead}_tele.nc"
            if os.path.exists(gfed_result_path) == False:
                nc_y = np.full((X.shape[0], Y.shape[1], X.shape[3], X.shape[4]), np.nan)
            else:
                data = Dataset(gfed_result_path, 'r')
                nc_y = data.variables['burntArea'][:]
                data.close()
            # target_years = range(1, 15)  # [14,1,2,5] #15
            target_years = [1]
            for year in target_years:  # range(1,15) 1,15: 2010-1997  #start from X.shape[0]-delta_time to get well-trained trees
                start_date = X.shape[0] - year * delta_time
                end_date = start_date + delta_time
                test_idxs = [idx for idx in range(start_date, end_date)]
                validate_start = start_date - 1 * delta_time
                validate_end = end_date + 1 * delta_time
                if validate_start < 0:
                    validate_start = end_date
                    validate_idxs = [idx for idx in range(validate_start, validate_start + 2 * delta_time)]
                if validate_end > X.shape[0]:
                    validate_start = start_date - 2 * delta_time
                    validate_idxs = [idx for idx in range(validate_start, validate_start + 2 * delta_time)]
                if validate_start >= 0 and validate_end <= X.shape[0]:
                    validate_idxs_part1 = [idx for idx in range(validate_start, validate_start + 1 * delta_time)]
                    validate_idxs_part2 = [idx for idx in range(validate_end - 1 * delta_time, validate_end)]
                    validate_idxs = list(set(validate_idxs_part1).union(set(validate_idxs_part2)))
                train_validate_idxs = list(set(all_indxs).difference(set(test_idxs)))
                train_idxs = list(set(train_validate_idxs).difference(set(validate_idxs)))
                train_idxs = train_validate_idxs
                validate_idxs = test_idxs
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/{model_idx}_{2010-year+1}_lead{time_lead}_tele.para'
                para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_vector.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_imv_tele.para'
                # para_path = f'K:/lifa/lbnl/wildfire/code/wildfire_prediction/results/gfed-attn/0731/gfed_attention/hidden_dim_{hidden_dim}/nohist_notele/{model_idx}_{2010 - year + 1}_lead{time_lead}_tele.para'
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                device = "cuda"  # "cpu"
                if os.path.exists(para_path) == False:
                    regressor = Fire_model(input_dim, time_lead, time_lead,hidden_dim,teleconnection=teleconnection).cuda()

                else:
                    regressor = Fire_model(input_dim, time_lead,time_lead, hidden_dim,teleconnection=teleconnection).cuda()#
                    regressor.load_state_dict(torch.load(para_path))
                # regressor.to(device)
                model = regressor
                weights = []
                weight_temp = 1
                decay_ratio = 1
                for lead in range(1, time_lead + 1):
                    weights.append(weight_temp)
                    weight_temp = weight_temp * decay_ratio
                weights = Variable(torch.Tensor(weights), requires_grad=False).to(device=device)

                ###################################################################################
                X_merge_input = X[train_idxs]
                X_merge_input = X_merge_input[:, :, :, region_mask == model_idx]
                X_merge_input = X_merge_input.transpose((3, 0, 1, 2))
                X_merge_input = X_merge_input.reshape((-1, X_merge_input.shape[2], X_merge_input.shape[3]))
                X_merge_input = (X_merge_input - region_data_means) / (region_data_std + pow(10, -8))
                print(np.sum(np.isnan(X_merge_input.reshape(-1))))
                Y_merge_input = Y[train_idxs]
                Y_merge_input = Y_merge_input[:, :, region_mask == model_idx]
                Y_merge_input = Y_merge_input.transpose((2, 0, 1))
                Y_merge_input = Y_merge_input.reshape(-1, Y_merge_input.shape[2])
                Y_merge_input = (Y_merge_input - region_y_mean) / (region_y_std + pow(10, -8))
                print('start training of region:', model_idx)

                loss = torch.nn.MSELoss(reduction='elementwise_mean')
                lr = 0.01
                # optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1*pow(10, -4))
                # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
                X_merge_input = torch.from_numpy(X_merge_input).float()
                Y_merge_input = torch.from_numpy(Y_merge_input).float()
                dataset = TensorDataset(X_merge_input, Y_merge_input)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                batch_num = X_merge_input.shape[0] // batch_size

                batch_cnt = 0

                delta_lr = int(math.floor(batch_num * epochs / 15))
                with torch.no_grad():
                    model.eval()
                    explained_vars, mse, r = evaluate(model, X, validate_idxs, region_mask, model_idx, sequence_len,
                                                      region_data_means, region_data_std, region_y_std, region_y_mean,
                                                      device, Y, loss)
                    r = 0
                    print('eva,mse', explained_vars, mse, r)
                    torch.save(model.state_dict(), para_path)
                for epo_idx in range(epochs):
                    costs = []
                    for batch_idx, (X_batch_input, Y_batch_input) in enumerate(loader):
                        model.train()
                        X_batch_input = X_batch_input.to(device=device)
                        ##############################################
                        # start train
                        y = Variable(Y_batch_input, requires_grad=False).to(device)
                        # Reset gradient
                        optimizer.zero_grad()

                        # Forward
                        fx, alphas, betas = model(X_batch_input)
                        # fx = fx.squeeze(1)
                        output = weighted_mse_loss(fx, y,
                                                   weights)  # loss(fx, y)#weighted_mse_loss(input,target,weights)
                        # output.detach().cpu().item()
                        output.backward()

                        # Update parameters
                        optimizer.step()

                        cost, y_train_pred, y_train_true = output.detach().cpu().item(), fx.detach().cpu().data.numpy(), y.detach().cpu().data.numpy()
                        # print(cost)
                        costs.append(cost)
                        # # cost,y_train_pred,y_train_true=train_lstm(model, loss, optimizer, X_batch_input, Y_batch_input,location_input,device)#X_merge_input: sequence_len x batch_size x feature_num
                        batch_cnt += 1
                        if batch_cnt % delta_lr == 0:
                            # lr*=0.8
                            # print('new lr:',lr)
                            scheduler.step()

                    if (epo_idx + 1) % 1 == 0:
                        model.eval()
                        print(np.mean(costs))
                        with torch.no_grad():
                            temp_explained_vars, temp_mse, temp_r = evaluate(model, X, validate_idxs, region_mask,
                                                                             model_idx, sequence_len, region_data_means,
                                                                             region_data_std, region_y_std,
                                                                             region_y_mean, device, Y, loss)
                            print('eva,mse', temp_explained_vars, temp_mse, temp_r)
                            if ( temp_r > r ):  # and epo_idx>=3 (temp_explained_vars>explained_vars and temp_mse<mse) or temp_r > r and
                                explained_vars = temp_explained_vars
                                r = temp_r
                                mse = temp_mse
                                torch.save(model.state_dict(), para_path)
                        model.train()
                        # plt.plot(range(len(costs)),costs)
                        # plt.show()
                print('evaluate:', explained_vars)
                # tain process
                #####################################################################################

                model.load_state_dict(torch.load(para_path))
                model.eval()
                with torch.no_grad():
                    # weights_result=np.ones((1,sequence_len,input_dim + emd_out_dim))
                    y_true = []
                    y_region_predict = []
                    for lat in range(X.shape[3]):
                        for lon in range(X.shape[4]):
                            if region_mask[lat, lon] == model_idx:
                                sample = X[test_idxs, :, :, lat, lon]
                                sample = np.array((sample - region_data_means) / (region_data_std + pow(10, -8)))
                                sample = torch.from_numpy(sample).float()
                                predicted_y, alphas, betas = predict_lstm(model, sample, device)
                                # if len(predicted_y.shape)==2:
                                # predicted_y=predicted_y[:,-1]
                                # predicted_y = predicted_y * region_y_std[-1] + region_y_mean[-1]
                                predicted_y = predicted_y * region_y_std + region_y_mean
                                predicted_y[predicted_y < 0] = 0
                                nc_y[test_idxs, :, lat, lon] = predicted_y
                                y_region_predict.extend(predicted_y.reshape(-1))
                                y_true.extend(Y[test_idxs, :, lat, lon].reshape(-1))
                    print("test:", model_idx, 2011 - year, explained_variance_score(y_true, y_region_predict))
                    # nc_save(nc_y, gfed_result_path)
                    # plt.subplot(121)
                    # plt.plot(range(len(y_region_predict)),y_true,label='y_true')
                    # plt.plot(range(len(y_region_predict)), y_region_predict,label='y_region_predict')
                    # plt.legend()
                    #
                    # plt.subplot(122)
                    # plt.scatter(y_true,y_region_predict)
                    # plt.show()

    return nc_y
from sklearn.metrics import mean_squared_error
def evaluate(model,X,sample_idxs,region_mask,model_idx,sequence_len,region_data_means,region_data_std,region_y_std,region_y_mean,device,Y,loss):
    # (model, X, validate_idxs, region_mask, model_idx, sequence_len, region_data_means, region_data_std,region_y_std, region_y_mean, device, Y, loss)
    X_merge_input = X[sample_idxs]
    X_merge_input = X_merge_input[:, :, :, region_mask == model_idx]
    X_merge_input = X_merge_input.transpose((3, 0, 1, 2))
    X_merge_input = X_merge_input.reshape((-1, X_merge_input.shape[2], X_merge_input.shape[3]))
    X_merge_input = (X_merge_input - region_data_means) / (region_data_std + pow(10, -8))
    Y_merge_input = Y[sample_idxs]
    Y_merge_input = Y_merge_input[:, :, region_mask == model_idx]
    Y_merge_input = Y_merge_input.transpose((2, 0, 1))
    y_true = Y_merge_input.reshape(-1, Y_merge_input.shape[2])


    X_merge_input = torch.from_numpy(X_merge_input).float()
    Y_merge_input = (y_true - region_y_mean) / (region_y_std + pow(10, -8))
    # Y_merge_input=torch.from_numpy(Y_merge_input).float()
    # Y_merge_input = Y_merge_input.to(device=device)
    predicted_y, alphas, betas =predict_lstm(model, X_merge_input,device)
    output = mean_squared_error(predicted_y, Y_merge_input)
    # output=output.detach().cpu().item()
    predicted_y = predicted_y * region_y_std + region_y_mean
    predicted_y[predicted_y < 0] = 0
    return explained_variance_score(y_true, predicted_y),output,pearsonr(y_true[:,-1], predicted_y[:,-1])[0]
def load_mask():
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    temp = data.variables['TSA'][:]
    data.close()
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\wildfire_9506-1012.nc'
    data = Dataset(nc_file_path, 'r')
    wildfires = data.variables['burntArea'][:]
    data.close()
    temp=temp[0]
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            value=temp[i,j]
            zero_cnt=wildfires[:,i,j].tolist().count(0)
            if math.isnan( float(value)) or zero_cnt>187:
                temp[i,j]=0
            else:temp[i,j]=1
    print('wildfire area:',np.sum( temp == 1) )
    plt.imshow(temp)


    plt.show()
    return temp

def Relative_Error(Y,Y_predict):
    return np.median(np.abs(Y-Y_predict)/(np.abs(Y)+pow(10,-5)))
def load_region(vars):
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\basic_14regions.nc'
    data = Dataset(nc_file_path, 'r')
    regions = data.variables['basic_14regions'][:]
    data.close()
    regions[regions==0]=np.nan

    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    temp = data.variables['CWDC'][:]
    data.close()
    temp = vars[0][0, 0]
    regions[np.isnan(temp)] = np.nan
    start_idx=0
    for var in vars[-3:]:
        if start_idx==0:
            var_new=var
            start_idx=1
        else:
            var_new=var_new+var
    temp = np.nanmean(var_new,axis=0)[0]
    regions[temp<=200] = np.nan
    temp=vars[2][0,0]
    regions[temp>50] = np.nan

    ##################################
    #exclude outlier
    future_data_files = ['pr_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc',
                         'tas_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc',
                         'VPD.95-10_96x144.nc',]#'ps_Amon_CESM2-WACCM_ssp585_r1i1p1f1_gn_201501-210012.nc',
    future_keys = ['pr', 'tas', 'VPD']#'ps',
    hist_data_files = ['RAIN.95-10_96x144.nc', 'TSA.95-10_96x144.nc',  'VPD.95-10_96x144.nc']#'PBOT.95-10_96x144.nc',
    hist_keys = ['RAIN', 'TSA',  'VPD']#'PBOT',
    target_regions = [5,8,9]
    percen_threshold=95

    for key_idx in range(3):#len(future_data_files)

        # print(future_keys[key_idx])
        #############################
        # future data
        nc_file = f"D:\lbnl\phd\dataset\CESM_585\T62\corrected/{future_data_files[key_idx]}"
        data = Dataset(nc_file, mode='r')
        rain_raw_future = np.array(data.variables[future_keys[key_idx]][-80 * 12:-40 * 12])
        data.close()

        ###########################################################
        ###historical data
        nc_file = f"K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\{hist_data_files[key_idx]}"
        data = Dataset(nc_file, mode='r')
        rain_raw_hist = np.array(data.variables[hist_keys[key_idx]][:])
        data.close()

        # for region_idx in target_regions:
        #     hist_data = rain_raw_hist[:, regions == region_idx].reshape(-1)
        #     hist_max, hist_min = np.nanmax(hist_data), np.nanmin(hist_data)
        #     # if key_idx==1:
        #     #     hist_max+=4
        #     # elif key_idx==3:
        #     #     hist_max *=1.1
        #     # future_max=np.nanmax(rain_raw_future,axis=0)
        #     future_max = np.nanpercentile(rain_raw_future,percen_threshold, axis=0)
        #     mask=((future_max>hist_max)&(regions==region_idx))
        #     regions[mask]=np.nan

    # nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\Soilm.95-10_96x144.nc'
    # data = Dataset(nc_file_path, 'r')
    # temp = data.variables['Soilm'][:]
    # data.close()
    # temp = temp[0]
    # regions[np.isnan(temp)] = np.nan
    plt.imshow(regions)
    plt.show()
    return regions
def pft_data():
    dir =  r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features/'
    file_path=dir+'PCT_NAT_PFT.nc'
    nc_file = Dataset(file_path, 'r')
    var = nc_file.variables['PCT_NAT_PFT'][:][0,0]
    nc_file.close()
    file_path = dir + 'landuse_years_95-10.nc'
    nc_file = Dataset(file_path, 'r')
    var_1 = nc_file.variables['PCT_NAT_PFT'][:]
    nc_file.close()
    var_1[:,:,var>100]=np.nan
    nc_file_path = r'K:\lifa\lbnl\wildfire\code\wildfire_prediction\ELM_fire\ELM_fire\input_features\ELM.fire.sub.nc'
    data = Dataset(nc_file_path, 'r')
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()
    # root_dir = r"K:\lifa\lbnl\wildfire\code\wildfire_prediction\results\gfed-attn\0731"
    # nc_fid2 = Dataset(root_dir + '/gfed_attn.nc', 'w', format="NETCDF4")
    nc_fid2 = Dataset(dir + 'landuse_years_95-10_new.nc', 'w', format="NETCDF4")
    nc_fid2.createDimension('lat', len(lat))
    nc_fid2.createDimension('lon', len(lon))
    nc_fid2.createDimension('time', var_1.shape[0])
    nc_fid2.createDimension('types', var_1.shape[1])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('lat',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('lon',))
    time_v = nc_fid2.createVariable("time", "f8", ("time",))
    burntArea = nc_fid2.createVariable('PCT_NAT_PFT', "f8", ("time",'types', "lat", "lon",))
    time_v.units = "days since 1993-08-01 00:00:00.0"
    time_v.calendar = "gregorian"
    dates = [datetime(1997, 1, 1) + relativedelta(years=+n) for n in range(var_1.shape[0])]
    time_v[:] = date2num(dates, units=time_v.units, calendar=time_v.calendar)
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    burntArea[:] = var_1[:]
    nc_fid2.close()
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def test():
    root_dir = r'D:\lbnl\phd\dataset\CESM_585\T62/corrected/'
    file_variables = {'wildfire_9506-1012.nc': ['burntArea'],  # 187x96x144
                      'populationDensity_2015-2100.nc': ['hdm'],  # 16 yearsx96x144
                      'road_live_stock_landuse.nc':['var'],# bare,forest,grass,road,animal
                      'rsds_Amon_CESM2-WACCM_ssp585_r1i1p1f1_gn_201501-210012.nc': ['rsds'],
                      'ps_Amon_CESM2-WACCM_ssp585_r1i1p1f1_gn_201501-210012.nc':['ps'],
                      'pr_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['pr'],
                      'VPD.95-10_96x144.nc': ['VPD'],
                      'tas_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['tas'],
                      'sfcWind_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['sfcWind'],
                      'mrso_Lmon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc':['mrso'],
                      'cCwd_Lmon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['cCwd'],
                      'cVeg_Lmon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['cVeg'],
                      'cLitter_Lmon_CESM2_ssp585_r4i1p1f1_gn_201501-210012.nc': ['cLitter'],
                      }
    # time_lag = 1 # time delay used for modeling
    time_dim, lat_dim, lon_dim = 0, 2, 3 # the index of dimensions, for example: for a variable with shape（600，15，192，288）, the 0 dim (600) is time, 2 dim (192) is lat
    vars = data_loading(root_dir, file_variables)
    print(len(vars))
    target_idx = [0]# list of indexes for predicted variables, for example: if file_variables = {'Temperature.nc': 'temperature','Precipitation.nc':'precipitation'}, target_idx = [0] means to predict next monthly temperature
    # repeat_types={1:'tile',2:'repeat'}#with ltn
    repeat_types = {1: 'repeat'}  # 3:'repeat',
    # repeat: a=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  
    # >>> a.repeat(5)  
    # array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,  
    #        4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9,  
    #        9, 9, 9, 9])
    #tile: a=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #np.tile(a,2)  >>>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # mask=load_mask()
    region=load_region(vars)
    # vars[-6]=vars[-6]-3
    setup_seed(20)
    for surrogate_var_idx in range(1,len(vars)):
        train(vars, target_idx, time_dim, lat_dim, lon_dim,repeat_types,region,surrogate_var_idx,time_lead=1,teleconnection=False,include_target_history=False)
if __name__=='__main__':
    test()
    # pft_data()