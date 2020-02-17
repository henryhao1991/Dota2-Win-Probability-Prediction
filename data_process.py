import torch
import json
import requests
import numpy as np
import os, sys
import subprocess
from collections import defaultdict

def parse_replay(dem_file_path, json_path='./tmp/temp.json'):
    
    path_and_file = os.path.split(json_path)
    if not os.path.isdir(path_and_file[0]):
        os.makedirs(path_and_file[0])
        
    subprocess.call('curl -o '+json_path+' localhost:5600 --data-binary "@'+dem_file_path+'"', shell=True)
    
    if os.path.getsize(json_path) == 0:
        print("The replay file is invalid. Exit.")
        sys.exit(1)
    else:
        print("Repaly file is successfully parsed.")
    

def json_file_processing(file_path, start_time=0, time_interval=30):
    
    time_stamp = start_time
    time_slices = []
    with open(file_path,'r',encoding='utf-8') as rep_file:
        for record in rep_file:
            time_sec = json.loads(record)
            if time_sec['time'] == time_stamp and time_sec['type'] == 'interval':
                time_slices.append(time_sec)
            elif time_sec['time'] > time_stamp:
                time_stamp += time_interval
                
    return time_slices


def time_slices_to_input(time_slices):
    
    team = {}
    team_rad = []
    team_dire = []
    for i in range(10):
        if time_slices[i]['slot'] < 5:
            team_rad.append(time_slices[i]['hero_id'])
        else:
            team_dire.append(time_slices[i]['hero_id'])
    team['rad'] = team_rad
    team['dire'] = team_dire
    
    time_stamp = time_slices[0]['time']
    time_interval = time_slices[10]['time'] - time_stamp
    assert time_interval != 0, "Time interval is 0. Something wrong with the input time slices."
    times_info = []
    single_time_info = defaultdict(list)
    life_state = [0] * 2
    tower_killed = [0] * 2
    
    for data in time_slices:
        if data['time'] != time_stamp:
            single_time_info['life_state'] = life_state
            single_time_info['towers_killed'] = tower_killed
            single_time_info['gold_rad'].sort()
            single_time_info['gold_dire'].sort()
            single_time_info['xp_rad'].sort()
            single_time_info['xp_dire'].sort()
            
            time_stamp += time_interval
            times_info.append(single_time_info)
            single_time_info = defaultdict(list)
            life_state = [0] * 2
            tower_killed = [0] * 2
            
        if data['slot'] < 5:
            single_time_info['gold_rad'].append(data['gold']/10000.0)
            single_time_info['xp_rad'].append(data['xp']/10000.0)
            tower_killed[0] += data['towers_killed']
            if data['life_state'] != 0:
                life_state[0] += 1
                
        else:
            single_time_info['gold_dire'].append(data['gold']/10000.0)
            single_time_info['xp_dire'].append(data['xp']/10000.0)
            tower_killed[1] += data['towers_killed']
            if data['life_state'] != 0:
                life_state[1] += 1
                
    return {"lineup": team, "time_info": times_info}

def feature_processing(time_info, lineup, feature='team', embedding_model='./model/hero_embeddings.txt'):
    
    assert os.path.exists(embedding_model), "Embedding model not found at {}.".format(embedding_model)
    embeddings = np.loadtxt(embedding_model)
    embedding_features = np.zeros(embeddings.shape[1] * 2)
    for i in range(5):
        embedding_features[:embeddings.shape[1]] = embedding_features[:embeddings.shape[1]] + embeddings[lineup['rad'][i]] / 5.0
        embedding_features[embeddings.shape[1]:] = embedding_features[embeddings.shape[1]:] + embeddings[lineup['dire'][i]] / 5.0
            
    if feature == 'team':
        feature_input = np.zeros((len(time_info),8))
        for i in range(len(time_info)):
            feature_input[i,0] = sum(time_info[i]['gold_rad'])
            feature_input[i,1] = sum(time_info[i]['gold_dire'])
            feature_input[i,2] = sum(time_info[i]['xp_rad'])
            feature_input[i,3] = sum(time_info[i]['xp_dire'])
            feature_input[i,4:6] = time_info[i]['life_state']
            feature_input[i,6:] = time_info[i]['towers_killed']
            
    elif feature == 'individual':
        feature_input = np.zeros((len(time_info),24))
        for i in range(len(time_info)):
            feature_input[i,:5] = time_info[i]['gold_rad']
            feature_input[i,5:10] = time_info[i]['gold_dire']
            feature_input[i,10:15] = time_info[i]['xp_rad']
            feature_input[i,15:20] = time_info[i]['xp_dire']
            feature_input[i,20:22] = time_info[i]['life_state']
            feature_input[i,22:] = time_info[i]['towers_killed']
            
    else:
        print("Wrong feature option. Either \'team\' or \'individual\'. Exit.")
        exit(1)
        
    return {"features":torch.Tensor(feature_input), "embeddings":torch.Tensor(embedding_features)}
            