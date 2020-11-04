
from tqdm import tqdm
from utilsCIFAR100 import unpickle, reshape_by_channels

meta = unpickle('meta')
print("Opened meta. keys are:", meta.keys())
fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]  # List of decoded labels

train = unpickle('train')
filenames = [t.decode('utf8') for t in train[b'filenames']]  # List of decoded train pics (from  byte string keys)

fine_labels = train[b'fine_labels']
data = train[b'data']

images = reshape_by_channels(data)
