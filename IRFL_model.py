from critic_objectives import*

class SupConModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, processor, temperature, dim_ins, feat_dims, use_label=False, head='mlp'):
        super(SupConModel, self).__init__()
        self.use_label = use_label

        self.model = model
        self.processor = processor
        self.device = 'cuda'

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

    def forward(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)

        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds

        feat1 = self.head1(feat1)

        feat2 = self.head2(feat2)

        feat = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        loss = self.critic(feat, label) if self.use_label else self.critic(feat)

        return loss

    def get_embedding(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)

        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds
        return feat1, feat2
    

    

def train_supcon(model, train_loader, optimizer, num_epoch=100):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
               
            loss = model(data_batch)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
    return



####CrossSelf

def image_two_augment(image_batch):
    aug_batch_1 = augment_image(image_batch)
    aug_batch_2 = augment_image(image_batch)
    
    return aug_batch_1, aug_batch_2

def text_two_augment(text_batch):
    aug_batch_1 = augment_text(text_batch)
    aug_batch_2 = augment_text(text_batch)
    
    return aug_batch_1, aug_batch_2

class CrossSelfModel(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, processor, temperature, dim_ins, feat_dims, head='mlp'):
        super(CrossSelfModel, self).__init__()
        #model_fun, dim_in = model_dict[name]

        self.model = model
        self.processor = processor
        self.device = 'cuda'

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

    def forward(self, x):
        x1, x2, y = x
      
        x1_v1, x1_v2 = image_two_augment(x1)
        x2_v1, x2_v2 = text_two_augment(x2)

        x = (x1, x2, y)
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)
        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds

        x1_embed = self.head1(feat1)
        x2_embed = self.head2(feat2)

        x_v1 = (x1_v1, x2_v1, y)
        inputs, label = process_fn(x_v1)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)
        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds

        x1_v1_embed = self.head1(feat1)
        x2_v1_embed = self.head2(feat2)

        x_v2 = (x1_v2, x2_v2, y)
        inputs, label = process_fn(x_v2)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)
        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds

        x1_v2_embed = self.head1(feat1)
        x2_v2_embed = self.head2(feat2)



        embed = torch.cat([x1_embed.unsqueeze(dim=1), x2_embed.unsqueeze(dim=1)], dim=1)
        #v1_embed = torch.cat([x1_v1_embed.unsqueeze(dim=1), x2_v1_embed.unsqueeze(dim=1)], dim=1)
        #v2_embed = torch.cat([x1_v2_embed.unsqueeze(dim=1), x2_v2_embed.unsqueeze(dim=1)], dim=1)
        v1_embed = torch.cat([x1_v1_embed.unsqueeze(dim=1), x1_v2_embed.unsqueeze(dim=1)], dim=1)
        v2_embed = torch.cat([x2_v1_embed.unsqueeze(dim=1), x2_v2_embed.unsqueeze(dim=1)], dim=1)

        loss = self.critic_r(embed) + self.critic_u1(v1_embed) + self.critic_u2(v2_embed)

        return loss

    def get_embedding(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)
        outputs = model(**inputs)

        feat1 = outputs.image_embeds
        feat2 = outputs.text_embeds
        return feat1, feat2     
    

####RUS Model

def mlp_head(dim_in, feat_dim):
    return nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

# Inputs x are sequence data of shape [seq, dim] 

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

def identity(x):
  return x


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

class RUSModel(nn.Module):
    def __init__(self, model, processor, feat_dims, y_ohe_dim, device, temperature=1, activation='relu', lr=1e-4, ratio=1):
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
        self.feat_dims = feat_dims
        self.model = model
        self.processor = processor
        self.device = device


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
        # y must have shape (N,1)

        N = y.shape[0]
        if y.dim() < 2:
            y = y.unsqueeze(1)
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1

        return y_ohe.to(self.device)
                         
    def forward(self, x): 
        # x is (images, texts, labels)
        # Get embeddings
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        label = label.unsqueeze(1)
        y_ohe = self.ohe(label)

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), label),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), label)
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed), y_ohe], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed), y_ohe], dim=1)),
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        label = label.unsqueeze(1)
        y_ohe = self.ohe(label)

        # Calculate InfoNCE loss for CLUB
        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                                             torch.cat([self.linears_club_x1x2_cond[1](x2_embed), y_ohe], dim=1))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        return x1_embed, x2_embed


    def get_optims(self):
        non_CLUB_params = [self.model.parameters(),
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

#####RUS Training
def train_rusmodel(model, train_loader, num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
             
            #loss, losses, ts = model(x_batch, y_batch)   
            loss = model(data_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(data_batch)  
                    
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

class RUSAugment(nn.Module):
    def __init__(self, model, processor, feat_dims, y_ohe_dim, device, temperature=1, activation='relu', lr=1e-4, ratio=1):
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
        self.model = model
        self.processor = processor
        self.device = device

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
                         
    def forward(self, x1, x1_aug, x2, x2_aug, label):     
        x = (x1, x2, label)
        x_aug = (x1_aug, x2_aug, label)    
  
        # Get embeddings
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        inputs_aug, _ = process_fn(x_aug)
        inputs_aug = inputs_aug.to(self.device)
        outputs_aug = model(**inputs_aug)
        x1_aug_embed = outputs_aug.image_embeds
        x2_aug_embed = outputs_aug.text_embeds


        label = label.unsqueeze(1)
        y_ohe = self.ohe(label)

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
                                                      self.linears_club_x1x2_cond[0](x1_embed)], dim=1), 
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                      self.linears_club_x1x2_cond[1](x2_embed)], dim=1))
        ]                    
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x1_aug, x2, x2_aug, label):
        # Get embeddings
        x = (x1, x2, label)
        x_aug = (x1_aug, x2_aug, label)    
        # Get embeddings
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        inputs_aug, _ = process_fn(x_aug)
        inputs_aug = inputs_aug.to(self.device)
        outputs_aug = model(**inputs_aug)
        x1_aug_embed = outputs_aug.image_embeds
        x2_aug_embed = outputs_aug.text_embeds


        label = label.unsqueeze(1)
        y_ohe = self.ohe(label)

        # Calculate InfoNCE loss for CLUB
        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), 
                                                                        self.linears_club_x1x2_cond[0](x1_aug_embed)], dim=1), 
                                                             torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                                        self.linears_club_x1x2_cond[1](x2_aug_embed)], dim=1))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x):
        inputs, label = process_fn(x)
        inputs, label = inputs.to(self.device), label.to(self.device)    
        outputs = model(**inputs)
        x1_embed = outputs.image_embeds
        x2_embed = outputs.text_embeds

        return x1_embed, x2_embed


    def get_optims(self):
        non_CLUB_params = [self.model.parameters(),
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
    

#####Aug Training
from torchvision import transforms

def augment_image(image_batch):
    aug_batch = []
    for img in image_batch:
        img = img.convert('RGB')
        img_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
        ])  
        aug_batch.append(img_transform(img))
    
    return aug_batch

def augment_image_unique(image_batch):
    aug_batch = []
    for img in image_batch:
        img = img.convert('RGB')
        img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
        ])  
        aug_batch.append(img_transform(img))
    
    return aug_batch

def augment_text(text_batch):
    aug_batch = []
    for text in text_batch:
        words = text.split(" ")
        random_idx = np.random.choice(len(words), 1).item(0)
        words_aug = words[:random_idx] + words[random_idx+1:]
        text_aug = " ".join(words_aug) 

        aug_batch.append(text_aug)
 
    return aug_batch



def train_rusaug(model, train_loader, num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
             
            #loss, losses, ts = model(x_batch, y_batch)   
            image_batch, text_batch, label_batch = data_batch
            image_aug = augment_image(image_batch)
            text_aug = augment_text(text_batch)

            loss = model(image_batch, image_aug, text_batch, text_aug, label_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(image_batch, image_aug, text_batch, text_aug, label_batch)  
                    
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

def train_rusaug_unique(model, train_loader, num_epoch=50, num_club_iter=1, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
             
            #loss, losses, ts = model(x_batch, y_batch)   
            image_batch, text_batch, label_batch = data_batch
            image_aug = augment_image_unique(image_batch)
            text_aug = augment_text(text_batch)

            loss = model(image_batch, image_aug, text_batch, text_aug, label_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): # increase number of iteration

                learning_loss = model.learning_loss(image_batch, image_aug, text_batch, text_aug, label_batch)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())

    return losses
