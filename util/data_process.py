import torch
import json
import requests
import numpy as np
import os, sys
import subprocess
from collections import defaultdict

def parse_replay(dem_file_path, json_path='./tmp/temp.json'):
    '''
    The function to parse the replay file into json file using a script provided by OpenDota:
    https://github.com/odota/parser

    Inputs:
	   dem_file_path: string. The file path for the replay file (In .dem format).
    Keywords:
	   json_path: string. The path to save the parsed json file. Directory will be created if not found.
    '''
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
    '''
    The function to get the game information at specific time stamps.

    Inputs:
	   file_path: string. The file path for the parsed replay file (.json format).
    Keywords:
	   start_time: int. The start time to get the game info.
	      time_interval: int. The time interval between time stamps.
    Return:
	   time_slices: list(dictionary). The list of dictionaries to store the game info at different time stamps.
    '''
    time_stamp = start_time
    time_slices = []
    with open(file_path,'r',encoding='utf-8') as rep_file:
        for record in rep_file:
            time_sec = json.loads(record)
	    #There are different type of game info in the parsed replay file. We only want 'interval' type.
            if time_sec['time'] == time_stamp and time_sec['type'] == 'interval':
                time_slices.append(time_sec)
            elif time_sec['time'] > time_stamp:
                time_stamp += time_interval

    return time_slices


def time_slices_to_input(time_slices):
    '''
    The function to further process the sliced game info at different time stamps and get the keys we want in the input dictionary.
    Inputs:
        time_sclices: list(dictionary). The sliced game info, which should be the return of function json_file_processing
    Return:
        dictionary:
            "lineup": dictionary. The hero lineup for radiant and dire teams (represented by the index of the heroes). keys: "rad", "dire"
            "time_info": list(dictionary). Time info of the features that we want.
    '''
    team = {}
    team_rad = []
    team_dire = []

    # Append hero index to the corresponding team.
    for i in range(10):
        if time_slices[i]['slot'] < 5:
            team_rad.append(time_slices[i]['hero_id'])
        else:
            team_dire.append(time_slices[i]['hero_id'])
    team['rad'] = team_rad
    team['dire'] = team_dire

    # Get the time_interval value.
    time_stamp = time_slices[0]['time']
    time_interval = time_slices[10]['time'] - time_stamp
    assert time_interval != 0, "Time interval is 0. Something wrong with the input time slices."

    times_info = []
    single_time_info = defaultdict(list)
    life_state = [0] * 2
    tower_killed = [0] * 2

    # Append game info we want to each time stamp.
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

        # Append info to the corresponding team. Scale gold and exp by 10000
        # Life state == 0: Alive. Not sure what is the difference between 1 and 2 (not if can buyback or not).
        # Might be useful if I figure out the difference between 1 and 2.
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

def feature_processing(time_info, lineup, feature='team', embedding_model='./saved_model/hero_embeddings.txt'):
    '''
    The function to process the selected features into the actual input to the model.
    Inputs:
        time_info: list. The "time_info" output from function time_slices_to_input.
        lineup: dictionary. The "lineup" output from function time_slices_to_input.
    Keywords:
        feature: string. The input feature for the LSTM network. Can only be 'team' or 'individual'.
        embedding_model: string. The path for the pretrained hero embedding model.
    Return:
        dictionary:
            "features": torch.Tensor. The input features for the LSTM network.
            "embeddings": torch.Tensor. The team embedding features for the hero2vec subnet.
    '''
    assert os.path.exists(embedding_model), "Embedding model not found at {}.".format(embedding_model)
    embeddings = np.loadtxt(embedding_model)
    embedding_features = np.zeros(embeddings.shape[1] * 2)
    for i in range(5):
        embedding_features[:embeddings.shape[1]] = embedding_features[:embeddings.shape[1]] + embeddings[lineup['rad'][i]] / 5.0
        embedding_features[embeddings.shape[1]:] = embedding_features[embeddings.shape[1]:] + embeddings[lineup['dire'][i]] / 5.0

    # For team features, aggregate the gold and experience for both teams.
    if feature == 'team':
        feature_input = np.zeros((len(time_info),8))
        for i in range(len(time_info)):
            feature_input[i,0] = sum(time_info[i]['gold_rad'])
            feature_input[i,1] = sum(time_info[i]['gold_dire'])
            feature_input[i,2] = sum(time_info[i]['xp_rad'])
            feature_input[i,3] = sum(time_info[i]['xp_dire'])
            feature_input[i,4:6] = time_info[i]['life_state']
            feature_input[i,6:] = time_info[i]['towers_killed']

    # For individual features, simply keep the individual gold and experience as the inputs.
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
