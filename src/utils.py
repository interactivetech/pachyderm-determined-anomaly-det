import torch
from netml.pparser.parser import PCAP
import torch.optim as optim
def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=5e-4)

def extract_iat_features(pcap):
    pcap.pcap2flows(q_interval=0.9)
    pcap.flow2features('IAT', fft=False, header=False)
    return pcap.features

def load_data(pcap_normal_path, pcap_anomaly_path):
    '''
    '/Users/mendeza/Documents/2023_projects/netml/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap'
    '/Users/mendeza/Documents/2023_projects/netml/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.119_anomaly2.pcap'
    '''
    pcap_normal = PCAP(
    pcap_normal_path,
    flow_ptks_thres=2,
    random_state=42,
    verbose=10,
    )
    pcap_anomaly = PCAP(
        pcap_anomaly_path,
        flow_ptks_thres=2,
        random_state=42,
        verbose=10,
    )
    # print(df)

    normal_feats = extract_iat_features(pcap_normal)
    abnormal_feats = extract_iat_features(pcap_anomaly)
    # print("normal_feats: ",normal_feats.shape)
    return normal_feats, abnormal_feats, pcap_normal, pcap_anomaly

def normalize(input):
    input = torch.FloatTensor(input)
    input -= torch.mean(input)
    input /= torch.std(input)
    return input.unsqueeze(0).unsqueeze(-1) 
def predict_model(model,input):
    output = model(input)
    cl = torch.argmax(output)
    print(cl.item(),output[0,cl.item()].item())
    return cl.item(),output[0,cl].item()
