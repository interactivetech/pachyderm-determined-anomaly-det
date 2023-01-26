import streamlit as st # web development
import numpy as np # np mean, np random 
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop 
import plotly.express as px # interactive charts 
import sys
import pandas as pd
import torch

sys.path.insert(0,'/Users/mendeza/Documents/2023_projects/pachyderm-determined-anomaly-det')
from netml.pparser.parser import PCAP

from src import classify_conv_model
from src import load_data, normalize, predict_model




NORMAL_PCAP_PATH='/Users/mendeza/Documents/2023_projects/pachyderm-determined-anomaly-det/data/srcIP_10.42.0.1_normal.pcap'
ANOMALY_PCAP_PATH='/Users/mendeza/Documents/2023_projects/pachyderm-determined-anomaly-det/data/srcIP_10.42.0.119_anomaly.pcap'

normal_feats, abnormal_feats, pcap_normal, pcap_anomaly = load_data(NORMAL_PCAP_PATH,ANOMALY_PCAP_PATH)
df1 = list(pcap_normal._iter_pcap_dict())
df2 = list(pcap_anomaly._iter_pcap_dict())
df1+=df2
input = np.concatenate([normal_feats,abnormal_feats[:,:-1]])# keep features same length
p = np.random.permutation(len(input))
input = input[p]
print("input: ",input.shape)

print("Loading Model...")

weight = torch.load("/Users/mendeza/Documents/2023_projects/netml/notebooks/anomaly_det.pt")

model = classify_conv_model()
model.load_state_dict(weight)
print("Model Loaded!")
cl_id, conf = predict_model(model,normalize(input[0]))
print("Prediction for input-{}: {}, {}".format(input[0].shape,cl_id,conf))


st.set_page_config(
    page_title = 'Real-Time Network Anomaly Detection Dashboard',
    page_icon = '✅',
    layout = 'wide'
)

# dashboard title

st.title("Real-Time Network Anomaly Detection Dashboard")

# top-level filters 

# job_filter = st.selectbox("Select the Job", pd.unique(df['job']))


# creating a single-element container.
placeholder = st.empty()

# dataframe filter 



# near real-time / live feed simulation 

for seconds in range(200):
#while True: 
    packet = input[seconds]
    packet_dict = df1[p[seconds]]
    cl_id, conf = predict_model(model,normalize(packet))
    print("Prediction for input-{}: {}, {}".format(input[0].shape,cl_id,conf))


    '''
    {'datetime': datetime.datetime(2017, 7, 3, 8, 8, 9),
    'dns_query': None,
    'dns_resp': None,
    'ip_dst': '224.0.0.22',
    'ip_src': '192.168.10.5',
    'is_dns': False,
    'length': 60,
    'mac_dst': '01:00:5e:00:00:16',
    'mac_src': 'b8:ac:6f:36:0a:8b',
    'port_dst': None,
    'port_src': None,
    'protocol': None,
    'time': Decimal('1499083689.704042')}
    '''
    class_map = {
        0:'Non Anomaly',
        1:'Anomaly'
    }
    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi5, kpi6= st.columns(2)
        # fill in those three columns with respective metrics or KPIs 
        kpi1.metric(label="Source IP + Port", value=str(packet_dict['ip_src'])+":"+str(packet_dict['port_src']))
        kpi2.metric(label="Destination IP and Port", value= str(packet_dict['ip_dst'])+":"+str(packet_dict['port_dst']) )
        kpi3.metric(label="Number of Packets", value=packet_dict['length'])
        kpi4.metric(label="Date and Time", value=packet_dict['datetime'].strftime("%m/%d/%Y, %H:%M:%S") )
        kpi5.metric(label="Class Prediction", value=class_map[cl_id])
        kpi6.metric(label="Confidence", value=conf)


        # create two columns for charts 

        fig_col1,fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Detailed Packet Information")
            # st.write(fig)
            st.write(packet_dict)
        with fig_col2:
            st.markdown("### Interarrival Time (IAT) Packet Histogram")
            fig = px.histogram(data_frame = packet)
            st.write(fig)
        st.markdown("### History")
        if cl_id == 0:
            time.sleep(1)
        elif cl_id == 1:
            time.sleep(5)
