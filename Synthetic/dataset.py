
import torch
import numpy as np
import math
from torch.utils.data import Dataset

# Synthetic Dataset
class MultimodalDataset(Dataset):
  def __init__(self, total_data, total_labels):
    self.data = torch.from_numpy(total_data).float()
    self.labels = torch.from_numpy(total_labels)
    self.num_modalities = self.data.shape[0]
  
  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return tuple([self.data[i, idx] for i in range(self.num_modalities)] + [self.labels[idx]])

  def sample_batch(self, batch_size):
    sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
    samples = self.__getitem__(sample_idxs)
    return samples
  


def get_intersections(num_modalities):
  modalities = [i for i in range(1, num_modalities+1)]
  all_intersections = [[]]
  for i in modalities:
    new = [s + [str(i)] for s in all_intersections]
    all_intersections += new
  res = list(map(lambda x: ''.join(x), sorted(all_intersections[1:])))
  return sorted(res, key=lambda x: (len(x), x))


def generate_data(num_data, num_modalities, feature_dim_info, label_dim_info, transforms=None):
  # Standard deviation of generated Gaussian distributions
  SEP = 0.5
  default_transform_dim = 100

  total_data = [[] for i in range(num_modalities)]
  total_labels = []
  total_raw_features = dict()
  for k in feature_dim_info:
    total_raw_features[k] = []


  # define transform matrices if not provided
  modality_dims = [0]*num_modalities
  for i in range(1, num_modalities+1):
      for k, d in feature_dim_info.items():
        if str(i) in k:
          modality_dims[i-1] += d

  if transforms is None:
      transforms = []
      for i in range(num_modalities):
        transforms.append(np.random.uniform(0.0,1.0,(modality_dims[i], default_transform_dim)))


  # generate data
  for data_idx in range(num_data):

    # get Gaussian data vector for each modality
    raw_features = dict()
    for k, d in feature_dim_info.items():
      raw_features[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d)*0.5, (1,))[0]

   
    modality_concept_means = []
    for i in range(1, num_modalities+1):
      modality_concept_means.append([])
      for k, v in raw_features.items():
        if str(i) in k:
          modality_concept_means[-1].append(v)

    raw_data = [np.concatenate(modality_concept_means[i]) for i in range(num_modalities)]
    

    # Transform into high-dimensional space
    modality_data = [raw_data[i] @ transforms[i] for i in range(num_modalities)]


    # update total data
    for i in range(num_modalities):
      total_data[i].append(modality_data[i])

    # update total raw data
    for k, f in raw_features.items():
      total_raw_features[k].append(f)

    # get label vector
    label_components = []
    for k,d in label_dim_info.items():
      label_components.append(raw_features[k][:d])
   
    label_vector = np.concatenate(label_components) #+ [np.random.randint(0, 2, 1)]) 
    label_prob = 1 / (1 + math.exp(-np.mean(label_vector)))
    total_labels.append([int(label_prob >= 0.5)])

      
  total_data = np.array(total_data)
  total_labels = np.array(total_labels)
  for k, v in total_raw_features.items():
    total_raw_features[k] = np.array(v)

  return total_data, total_labels, total_raw_features



def get_labels(label_dim_info, total_raw_features):
    label_components = []
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
   
    label_vector = np.concatenate(label_components, axis=1) #+ [np.random.randint(0, 2, 1)]) 
    label_prob = 1 / (1 + np.exp(-np.mean(label_vector, axis=1)))
    total_labels = (label_prob >= 0.5).astype('float')
    total_labels = np.expand_dims(total_labels, axis=1)

    return total_labels

def get_nonlinear_labels(label_dim_info, total_raw_features):
    label_components = []
    total_label_dim = 0
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
      total_label_dim += d

    w1 = np.ones((total_label_dim,total_label_dim))
    w2 = np.ones((total_label_dim,total_label_dim))  
   
    label_vector = np.concatenate(label_components, axis=1) #+ [np.random.randint(0, 2, 1)]) 
    label_vector = label_vector @ w1 @ w2

    label_prob = 1 / (1 + np.exp(-np.mean(label_vector, axis=1)))
    total_labels = (label_prob >= 0.5).astype('float')
    total_labels = np.expand_dims(total_labels, axis=1)

    return total_labels

def get_planar_flow_labels(label_dim_info, total_raw_features):
    label_components = []
    total_label_dim = 0
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
      total_label_dim += d

    #w = np.ones((total_label_dim,total_label_dim))
    #b = np.ones(total_label_dim,)
    #u = np.ones((total_label_dim,total_label_dim))
    w = np.random.normal(2, 1, size=(total_label_dim,total_label_dim))
    b = np.random.normal(2, 1, size=(total_label_dim,))
    u = np.random.normal(2, 1, size=(total_label_dim,total_label_dim))

    #head = np.ones((total_label_dim,5))
    head = np.random.normal(2, 1, size=(total_label_dim,20))
   
    z = np.concatenate(label_components, axis=1) #+ [np.random.randint(0, 2, 1)]) 
    z = z + np.tanh(z @ w + b) @ u 
    z = z @ head

    label_prob = torch.softmax(torch.from_numpy(z), dim=1)
    total_labels = torch.argmax(label_prob, dim=1).unsqueeze(1)


    return total_labels.numpy()
