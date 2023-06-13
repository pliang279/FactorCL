from critic_objectives import*

class SupConResNetY(nn.Module):
    """backbone + projection head"""
    def __init__(self, temperature, encoders, dim_ins, feat_dims, head='mlp'):
        super(SupConResNetY, self).__init__()
        #model_fun, dim_in = model_dict[name]

        self.encoders = nn.ModuleList(encoders)
        if head == 'linear':
            self.head1 = nn.Linear(dim_ins[0], feat_dims[0])
            self.head2 = nn.Linear(dim_ins[1], feat_dims[1])
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.critic = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, y):
        feat1 = self.encoders[0](x1)
        feat1 = self.head1(feat1)

        feat2 = self.encoders[1](x2)
        feat2 = self.head2(feat2)

        feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        loss = self.critic(feat, y)

        return loss

    def get_embedding(self, x1, x2):
        return self.encoders[0](x1), self.encoders[1](x2)  
    
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, temperature, encoders, dim_ins, feat_dims, head='mlp'):
        super(SupConResNet, self).__init__()
        #model_fun, dim_in = model_dict[name]

        self.encoders = nn.ModuleList(encoders)
        if head == 'linear':
            self.head1 = nn.Linear(dim_ins[0], feat_dims[0])
            self.head2 = nn.Linear(dim_ins[1], feat_dims[1])
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.critic = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, y):
        feat1 = self.encoders[0](x1)
        feat1 = self.head1(feat1)

        feat2 = self.encoders[1](x2)
        feat2 = self.head2(feat2)

        feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        loss = self.critic(feat)

        return loss

    def get_embedding(self, x1, x2):
        return self.encoders[0](x1), self.encoders[1](x2)   

    
class SupConModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, temperature, encoders, dim_ins, feat_dims, use_label=False, head='mlp'):
        super(SupConModel, self).__init__()
        
        self.use_label = use_label
        self.encoders = nn.ModuleList(encoders)
        if head == 'linear':
            self.head1 = nn.Linear(dim_ins[0], feat_dims[0])
            self.head2 = nn.Linear(dim_ins[1], feat_dims[1])
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.critic = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, y):
        feat1 = self.encoders[0](x1)
        feat1 = self.head1(feat1)

        feat2 = self.encoders[1](x2)
        feat2 = self.head2(feat2)

        feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        loss = self.critic(feat, y) if self.use_label else self.critic(feat)

        return loss

    def get_embedding(self, x1, x2):
        return self.encoders[0](x1), self.encoders[1](x2)  


class CrossSelfModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, temperature, encoders, dim_ins, feat_dims, augment, head='mlp'):
        super(CrossSelfModel, self).__init__()
        #model_fun, dim_in = model_dict[name]

        self.encoders = nn.ModuleList(encoders)
        if head == 'linear':
            self.head1 = nn.Linear(dim_ins[0], feat_dims[0])
            self.head2 = nn.Linear(dim_ins[1], feat_dims[1])
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head2 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.critic_u1 = SupConLoss(temperature=temperature)
        self.critic_u2 = SupConLoss(temperature=temperature)
        self.critic_r = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, y):
        x1_v1, x1_v2 = augment(x1)
        x2_v1, x2_v2 = augment(x2)

        x1_embed = self.head1(self.encoders[0](x1))
        x2_embed = self.head2(self.encoders[1](x2))
        
        x1_v1_embed = self.head1(self.encoders[0](x1_v1))
        x1_v2_embed = self.head1(self.encoders[0](x1_v2))
        x2_v1_embed = self.head2(self.encoders[1](x2_v1))
        x2_v2_embed = self.head2(self.encoders[1](x2_v2))


        embed = torch.cat([x1_embed.unsqueeze(dim=1), x2_embed.unsqueeze(dim=1)], dim=1)
        v1_embed = torch.cat([x1_v1_embed.unsqueeze(dim=1), x1_v2_embed.unsqueeze(dim=1)], dim=1)
        v2_embed = torch.cat([x2_v1_embed.unsqueeze(dim=1), x2_v2_embed.unsqueeze(dim=1)], dim=1)

        loss = self.critic_r(embed) + self.critic_u1(v1_embed) + self.critic_u2(v2_embed)

        return loss

    def get_embedding(self, x1, x2):
        return self.encoders[0](x1), self.encoders[1](x2)     


class FactorCrossSelfModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, temperature, encoders1, encoders2, dim_ins, feat_dims, augment, head='mlp'):
        super(FactorCrossSelfModel, self).__init__()
        #model_fun, dim_in = model_dict[name]

        self.encoders1 = nn.ModuleList(encoders1)
        self.encoders2 = nn.ModuleList(encoders2)

        if head == 'linear':
            self.head1 = nn.Linear(dim_ins[0], feat_dims[0])
            self.head2 = nn.Linear(dim_ins[1], feat_dims[1])
        elif head == 'mlp':
            self.head11 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head12 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )
            self.head21 = nn.Sequential(
                nn.Linear(dim_ins[0], dim_ins[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[0], feat_dims[0])
            )
            self.head22 = nn.Sequential(
                nn.Linear(dim_ins[1], dim_ins[1]),
                nn.ReLU(inplace=True),
                nn.Linear(dim_ins[1], feat_dims[1])
            )

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.critic_u1 = SupConLoss(temperature=temperature)
        self.critic_u2 = SupConLoss(temperature=temperature)
        self.critic_r = SupConLoss(temperature=temperature)

    def forward(self, x1, x2, y):
        x1_v1, x1_v2 = augment(x1)
        x2_v1, x2_v2 = augment(x2)

        x1_embed = self.head11(self.encoders1[0](x1))
        x2_embed = self.head12(self.encoders1[1](x2))
        
        x1_v1_embed = self.head21(self.encoders2[0](x1_v1))
        x1_v2_embed = self.head21(self.encoders2[0](x1_v2))
        x2_v1_embed = self.head22(self.encoders2[1](x2_v1))
        x2_v2_embed = self.head22(self.encoders2[1](x2_v2))


        embed = torch.cat([x1_embed.unsqueeze(dim=1), x2_embed.unsqueeze(dim=1)], dim=1)
        v1_embed = torch.cat([x1_v1_embed.unsqueeze(dim=1), x1_v2_embed.unsqueeze(dim=1)], dim=1)
        v2_embed = torch.cat([x2_v1_embed.unsqueeze(dim=1), x2_v2_embed.unsqueeze(dim=1)], dim=1)

        loss = self.critic_r(embed) + self.critic_u1(v1_embed) + self.critic_u2(v2_embed)

        return loss

    def get_embedding(self, x1, x2):
        return torch.cat([self.encoders1[0](x1), self.encoders2[0](x1)], dim=1), torch.cat([self.encoders1[1](x2), self.encoders2[1](x2)], dim=1)



def train_supcon(model, train_loader, optimizer, num_epoch=100):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0].float().cuda()
            x2_batch = data_batch[1].float().cuda()
            y_batch = data_batch[2].float().cuda()
               
            loss = model(x1_batch, x2_batch, y_batch)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
    return

def train_supcon_mosi(model, train_loader, optimizer, modalities=[0,2], num_epoch=100):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            y_batch = label_to_binary(data_batch[3]).float().cuda()
               
            loss = model(x1_batch, x2_batch, y_batch)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
    return

def sarcasm_label(y_batch):
  res = copy.deepcopy(y_batch)
  res[y_batch == -1] = 0

  return res

def train_supcon_sarcasm(model, train_loader, optimizer, modalities=[0,2], num_epoch=100):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            y_batch = sarcasm_label(data_batch[3]).float().cuda()
               
            loss = model(x1_batch, x2_batch, y_batch)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
    return



def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )



#####Augmentations

# Inputs x are sequence data of shape [seq, dim] 
from torchvision import transforms

def permute(x):
  # shuffle the sequence order
  idx = torch.randperm(x.shape[0])
  return x[idx]

def noise(x):
  noise = torch.randn(x.shape) * 0.1
  return x + noise.to(x.device)

def drop(x):
  # drop 20% of the sequences
  drop_num = x.shape[0] // 5
  
  x_aug = torch.clone(x)
  drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
  x_aug[drop_idxs] = 0.0
  return x_aug  

def mixup(x, alpha=1.0):
    indices = torch.randperm(x.shape[0])
    lam = np.random.beta(alpha, alpha)
    aug_x = x * lam + x[indices] * (1 - lam)

    return aug_x

def image_transform(x): 
    t = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip()])
    return t(x)

def identity(x):
  return x

def augment(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 2, replace=False)
    t1 = transforms[t_idxs[0]]
    t2 = transforms[t_idxs[1]]
    v1[i] = t1(v1[i])
    v2[i] = t2(v2[i])
  
  return v1, v2

# return 1 augmented instance 
def augment_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [permute, noise, drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 1, replace=False)
    t = transforms[t_idxs[0]]
    v2[i] = t(v2[i])
  
  return v2

def augment_embed(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [noise, mixup, identity]

  t_idxs = np.random.choice(3, 2, replace=False)
  t1 = transforms[t_idxs[0]]
  t2 = transforms[t_idxs[1]]
  v1 = t1(v1)
  v2 = t2(v2)
  
  return v1, v2

# return 1 augmented instance 
def augment_embed_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [noise, mixup, identity]

  t_idxs = np.random.choice(3, 1, replace=False)
  t = transforms[t_idxs[0]]
  v2 = t(v2)
  
  return v2

def augment_image_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [image_transform, identity]

  t_idxs = np.random.choice(2, 1, replace=False)
  t = transforms[t_idxs[0]]
  v2 = t(v2)
  
  return v2

def augment_audio_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [noise, mixup, identity]

  t_idxs = np.random.choice(3, 1, replace=False)
  t = transforms[t_idxs[0]]
  v2 = t(v2)
  
  return v2

def augment_mimic(x_batch):
  if x_batch.dim() == 2:
    return augment_embed_single(x_batch)
  else:
    return augment_single(x_batch) 

def augment_avmnist(x_batch):
  if x_batch.shape[-1] == 28: #image
    return augment_embed_single(x_batch)
  else: #audio
    return augment_single(x_batch) 


#####Models
class RUSModel(nn.Module):
    def __init__(self, encoders, feat_dims, y_ohe_dim, temperature=1, activation='relu', lr=1e-4, ratio=1):
        super(RUSModel, self).__init__()
        self.critic_hidden_dim = 512
        #self.critic_embed_dim = 128
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio
        self.y_ohe_dim = y_ohe_dim
        self.temperature = temperature

        #self.club_prob_hidden_size = 15
        
        #encoder
        #self.dim_in = 2048
        self.feat_dims = feat_dims
        self.backbones = nn.ModuleList(encoders)

        #linears
        self.linears_infonce_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        self.linears_infonce_x1y = mlp_head(self.feat_dims[0], self.feat_dims[0])
        self.linears_infonce_x2y = mlp_head(self.feat_dims[1], self.feat_dims[1])
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        #critics
        self.infonce_x1x2 = InfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)
        self.club_x1x2_cond = CLUBInfoNCECritic(self.feat_dims[0] + self.y_ohe_dim, self.feat_dims[1] + self.y_ohe_dim, 
                                                self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 

        self.infonce_x1y = InfoNCECritic(self.feat_dims[0], 1, self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x2y = InfoNCECritic(self.feat_dims[1], 1, self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x1x2_cond = InfoNCECritic(self.feat_dims[0] + self.y_ohe_dim, self.feat_dims[1] + self.y_ohe_dim, 
                                               self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.club_x1x2 = CLUBInfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe    
                         
    def forward(self, x1, x2, y): 
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), y),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), y)
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed), y_ohe], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed), y_ohe], dim=1)),
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, y):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        # Calculate InfoNCE loss for CLUB
        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                                             torch.cat([self.linears_club_x1x2_cond[1](x2_embed), y_ohe], dim=1))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x1, x2):
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)
         
        x1_reps = [self.linears_infonce_x1x2[0](x1_embed), 
                   self.linears_club_x1x2[0](x1_embed), 
                   self.linears_infonce_x1y(x1_embed),
                   self.linears_infonce_x1x2_cond[0](x1_embed),
                   self.linears_club_x1x2_cond[0](x1_embed)]

        x2_reps = [self.linears_infonce_x1x2[1](x2_embed),
                   self.linears_club_x1x2[1](x2_embed),
                   self.linears_infonce_x2y(x2_embed),
                   self.linears_infonce_x1x2_cond[1](x2_embed),
                   self.linears_club_x1x2_cond[1](x2_embed)]
        
        return torch.cat(x1_reps, dim=1), torch.cat(x2_reps, dim=1)

    def get_four_embeddings(self, x1, x2):
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)
         
        x1_s = [self.linears_infonce_x1x2[0](x1_embed), 
                self.linears_club_x1x2_cond[0](x1_embed)]

        x1_u = [self.linears_club_x1x2[0](x1_embed), 
                self.linears_infonce_x1y(x1_embed),
                self.linears_infonce_x1x2_cond[0](x1_embed)]

        x2_s = [self.linears_infonce_x1x2[1](x2_embed),
                self.linears_club_x1x2_cond[1](x2_embed)]

        x2_u = [self.linears_club_x1x2[1](x2_embed),
                self.linears_infonce_x2y(x2_embed),
                self.linears_infonce_x1x2_cond[1](x2_embed)]
        
        return torch.cat(x1_s, dim=1), torch.cat(x1_u, dim=1), torch.cat(x2_s, dim=1), torch.cat(x2_u, dim=1)


    def get_optims(self):
        non_CLUB_params = [self.backbones.parameters(),
                           self.infonce_x1x2.parameters(), 
                           self.infonce_x1y.parameters(), 
                           self.infonce_x2y.parameters(), 
                           self.infonce_x1x2_cond.parameters(), 
                           self.linears_infonce_x1x2.parameters(), 
                           self.linears_infonce_x1y.parameters(), 
                           self.linears_infonce_x2y.parameters(), 
                           self.linears_infonce_x1x2_cond.parameters(), 
                           self.linears_club_x1x2_cond.parameters(), 
                           self.linears_club_x1x2.parameters()]
                  

        CLUB_params = [self.club_x1x2_cond.parameters(), 
                       self.club_x1x2.parameters()]

        non_CLUB_optims = [optim.Adam(param, lr=self.lr) for param in non_CLUB_params]
        CLUB_optims = [optim.Adam(param, lr=self.lr) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims

class RUSAugment(nn.Module):
    def __init__(self, encoders, feat_dims, y_ohe_dim, temperature=1, activation='relu', lr=1e-4, ratio=1):
        super(RUSAugment, self).__init__()
        self.critic_hidden_dim = 512
        #self.critic_embed_dim = 128
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio
        self.y_ohe_dim = y_ohe_dim
        self.temperature = temperature

        #self.club_prob_hidden_size = 15
        
        #encoder
        #self.dim_in = 2048
        self.feat_dims = feat_dims
        self.backbones = nn.ModuleList(encoders)

        #linears
        self.linears_infonce_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        self.linears_infonce_x1y = mlp_head(self.feat_dims[0], self.feat_dims[0])
        self.linears_infonce_x2y = mlp_head(self.feat_dims[1], self.feat_dims[1])
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp_head(self.feat_dims[i], self.feat_dims[i]) for i in range(2)])

        #critics
        self.infonce_x1x2 = InfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)
        self.club_x1x2_cond = CLUBInfoNCECritic(self.feat_dims[0]*2, self.feat_dims[1]*2, 
                                                self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 

        self.infonce_x1y = InfoNCECritic(self.feat_dims[0], self.feat_dims[0], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x2y = InfoNCECritic(self.feat_dims[1], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.infonce_x1x2_cond = InfoNCECritic(self.feat_dims[0]*2, self.feat_dims[1]*2, 
                                               self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature) 
        self.club_x1x2 = CLUBInfoNCECritic(self.feat_dims[0], self.feat_dims[1], self.critic_hidden_dim, self.critic_layers, activation, temperature=temperature)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe    
                         
    def forward(self, x1, x2, x1_aug, x2_aug):         
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), self.linears_infonce_x1y(x1_aug_embed)),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), self.linears_infonce_x2y(x2_aug_embed))
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed), 
                                                         self.linears_infonce_x1x2_cond[0](x1_aug_embed)], dim=1), 
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed), 
                                                         self.linears_infonce_x1x2_cond[1](x2_aug_embed)], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), 
                                                      self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1), 
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                      self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
        ]                    
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, x1_aug, x2_aug):
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)

        # Calculate InfoNCE loss for CLUB
        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), 
                                                                        self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1), 
                                                             torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                                        self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x1, x2):
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)
         
        x1_reps = [self.linears_infonce_x1x2[0](x1_embed), 
                   self.linears_club_x1x2[0](x1_embed), 
                   self.linears_infonce_x1y(x1_embed),
                   self.linears_infonce_x1x2_cond[0](x1_embed),
                   self.linears_club_x1x2_cond[0](x1_embed)]

        x2_reps = [self.linears_infonce_x1x2[1](x2_embed),
                   self.linears_club_x1x2[1](x2_embed),
                   self.linears_infonce_x2y(x2_embed),
                   self.linears_infonce_x1x2_cond[1](x2_embed),
                   self.linears_club_x1x2_cond[1](x2_embed)]
        
        return torch.cat(x1_reps, dim=1), torch.cat(x2_reps, dim=1)


    def get_optims(self):
        non_CLUB_params = [self.backbones.parameters(),
                           self.infonce_x1x2.parameters(), 
                           self.infonce_x1y.parameters(), 
                           self.infonce_x2y.parameters(), 
                           self.infonce_x1x2_cond.parameters(), 
                           self.linears_infonce_x1x2.parameters(), 
                           self.linears_infonce_x1y.parameters(), 
                           self.linears_infonce_x2y.parameters(), 
                           self.linears_infonce_x1x2_cond.parameters(), 
                           self.linears_club_x1x2_cond.parameters(), 
                           self.linears_club_x1x2.parameters()]
                  

        CLUB_params = [self.club_x1x2_cond.parameters(), 
                       self.club_x1x2.parameters()]

        non_CLUB_optims = [optim.Adam(param, lr=self.lr) for param in non_CLUB_params]
        CLUB_optims = [optim.Adam(param, lr=self.lr) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims
    

#####Regular Training

def train_rusmodel(model, train_loader, num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0].float().cuda()
            x2_batch = data_batch[1].float().cuda()
            y_batch = data_batch[2].float().cuda()
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(x1_batch, x2_batch, y_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(x1_batch, x2_batch, y_batch)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
                #print([l.item() for l in losses])
                #print([t.item() for t in ts])
    return losses


#####MOSI/MOSEI Training
def label_to_binary(y_batch):
  res = copy.deepcopy(y_batch)

  res[y_batch >= 0] = 1
  res[y_batch < 0] = 0
  
  return res

def train_rusmodel_mosi(model, train_loader, modalities=[0,2], num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            y_batch = label_to_binary(data_batch[3]).float().cuda()
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(x1_batch, x2_batch, y_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(x1_batch, x2_batch, y_batch)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
                #print([l.item() for l in losses])
                #print([t.item() for t in ts])
    return losses

def train_rusaug_mosi(model, train_loader, modalities=[0,2], num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            #y_batch = label_to_trinary(data_batch[3]).float().cuda()

            x1_aug = augment_single(x1_batch)
            x2_aug = augment_single(x2_batch)
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(x1_batch, x2_batch, x1_aug, x2_aug)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(x1_batch, x2_batch, x1_aug, x2_aug)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
                #print([l.item() for l in losses])
                #print([t.item() for t in ts])
    return losses


#####Sarcasm/Humor Training

def train_rusmodel_sarcasm(model, train_loader, modalities=[0,2], num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            y_batch = sarcasm_label(data_batch[3]).float().cuda()
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(x1_batch, x2_batch, y_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(x1_batch, x2_batch, y_batch)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
                #print([l.item() for l in losses])
                #print([t.item() for t in ts])
    return losses


def train_rusaug_sarcasm(model, train_loader, modalities=[0,2], num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0][modalities[0]].float().cuda()
            x2_batch = data_batch[0][modalities[1]].float().cuda()
            #y_batch = label_to_trinary(data_batch[3]).float().cuda()

            x1_aug = augment_single(x1_batch)
            x2_aug = augment_single(x2_batch)
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(x1_batch, x2_batch, x1_aug, x2_aug)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(x1_batch, x2_batch, x1_aug, x2_aug)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
                #print([l.item() for l in losses])
                #print([t.item() for t in ts])
    return losses
