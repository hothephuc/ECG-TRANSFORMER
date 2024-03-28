import wfdb
import numpy as np

def read_record(path):
  record = wfdb.rdrecord(path, sampfrom=0,channels=[0])

  return record.p_signal

def read_annotate(path):
    ann = wfdb.rdann(path, extension='atr', sampfrom=0)
    return ann

def spliting_index(chunk_size,chunk_step, max_index, num_overlap, start_idx ):
  num_overlap= chunk_size-num_overlap
  chunk_size = chunk_size
  max_index = max_index -chunk_size 
  index_array = np.arange(0, chunk_size,chunk_step)[None, :]+np.arange(start_idx,max_index, num_overlap)[:,None]
  return index_array

'''
text = "/home/phuc/university_of_science/ECG_Transformer/ecg_transformer/data/mit-bih-arrhythmia-database-1.0.0/100"
signal, labels = read_record(text.encode())
#Chunk length and overlapping with 
start_idx = 0
chunk_size = 1000
chunk_step = 1
max_index = len(signal)
num_overlap = 300
#index_array = np.arange(0, chunk_size,chunk_step)[None, :]+np.arange(start_idx,2701, 270)[:,None]
idxes= spliting_index(chunk_size,chunk_step, max_index, num_overlap, start_idx)
'''
