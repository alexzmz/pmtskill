import gzip
import pickle

path="/home/zmz/Workspace/gui/src1/runtime/trajectories/AudioRecorderRecordAudio_0.pkl.gz"

with gzip.open(path, "rb") as f:
    value = pickle.load(f)

print(type(value))

if isinstance(value, dict):
    print(value.keys())

elif isinstance(value, list):
    print(len(value))
    print(type(value[0]))
