#PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim

#Other libraries
import numpy as np
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import streamlit as st

#Custom python files
from model.model import *
from training import *
from util.dataloader import *
from util.data_process import *

st.title("Dota2 Win Probability Prediction")

upload_file = st.sidebar.text_input("Type a input replay file path (.dem):", value='./')

selected_model = st.sidebar.selectbox("Select the model you want to use:", ('--', 'Heuristic', 'LSTM', 'LSTM + Hero2Vec'))

selected_features = st.sidebar.selectbox("Select the feature you want to use:", ('--', "Team Features", "Individual Features"))

if selected_model == '--' or selected_features == '--' or upload_file == None:
    st.warning('Please select a model, features, and upload the replay file.')
    raise st.ScriptRunner.StopException
    
if not os.path.exists(upload_file):
    st.warning('Replay file not found. Please check the path you typed.')
    raise st.ScriptRunner.StopException
    
json_path = './tmp/temp.json'
parse_replay(upload_file, json_path=json_path)
time_slices = json_file_processing(json_path)
data = time_slices_to_input(time_slices)

hidden_dim = 50
batch_size = 20
if selected_features == "Team Features":
    input_dim = 8
    feature='team'
elif selected_features == "Individual Features":
    input_dim = 24
    feature='individual'
    
inputs = feature_processing(data['time_info'], data['lineup'], feature=feature)

if selected_model == "LSTM":
    model_used = TrainingAndEvaluation(input_dim, hidden_dim, model='LSTM_baseline', collate_fn=PadSequence,
                                     batch_size=batch_size, train=False)
    if selected_features == "Team Features":
        model_used.model.model = torch.load('./saved_model/model_agg.pt')
    elif selected_features == "Individual Features":
        model_used.model.model = torch.load('./saved_model/model_indi.pt')

elif selected_model == "LSTM + Hero2Vec":
    model_used = TrainingAndEvaluation(input_dim, hidden_dim, model='LSTM_with_h2v', collate_fn=PadSequence,
                                     batch_size=batch_size, train=False)
    if selected_features == "Team Features":
        model_used.model.model = torch.load('./saved_model/model_agg_h2v_subnet.pt')
    elif selected_features == "Individual Features":
        model_used.model.model = torch.load('./saved_model/model_indi_h2v_subnet.pt')

elif selected_model == "Heuristic":
    model_used = TrainingAndEvaluation(model='heuristic', collate_fn=PadSequence)


sample_graph = model_used.get_prediction_from_file(inputs)
prob = sample_graph.numpy().squeeze()

time_stamp = 0.5*np.arange(len(prob))

fig = go.Figure()
fig.add_trace(go.Scatter(x=time_stamp, y=prob, mode='lines',
                        hovertemplate='Time: %{x} min<br>Radiant Win Probability: %{y:.2f}', name='', showlegend = False))
fig.add_shape(
        # Line Horizontal
        go.layout.Shape(
            type="line",
            x0=0,
            y0=0.5,
            x1=time_stamp[-1],
            y1=0.5,
            line=dict(
                color="Black",
                width=1,
            ),
    ))

fig['layout']['yaxis1'].update(title='',range=[0, 1], autorange=False)

st.plotly_chart(fig)
