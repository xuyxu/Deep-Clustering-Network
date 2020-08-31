import torch
import numbers
import numpy as np
import torch.nn as nn
from kmeans import KMeans
from autoencoder import AutoEncoder


class DCN(nn.Module):
    
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.lamda = args.lamda
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        # Core modules
        self.kmeans = KMeans(args)
        self.autoencoder = AutoEncoder(args).to(self.device)
        
        self.criterion  = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.wd)
    
    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        rec_loss = self.criterion(X, rec_X)
        
        # Regularization term on the clustering performance
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.lamda * torch.squeeze(sample_dist_loss)
        
        return (rec_loss + dist_loss, 
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())
    
    def pretrain(self, train_loader, epoch=100, verbose=True):
        
        if not self.args.pretrain:
            return
        
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value={}'
            raise ValueError(msg.format(epoch))
        
        if verbose:
            print('========== Start pretraining ==========')
        
        self.train()
        for e in range(epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)
                
                if verbose and batch_idx % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx, 
                                     loss.detach().cpu().numpy()))
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()
        
        if verbose:
            print('========== End pretraining ==========\n')
        
        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

    def fit(self, epoch, train_loader, verbose=True):
        
        # [Step-1] Update the assignment results
        cluster_id_list = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.view(batch_size, -1).to(self.device)
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()
                cluster_id = self.kmeans.update_assign(latent_X)
                cluster_id_list.append(cluster_id)
        
        # [Step-2] Update clusters in Kmeans module in an online fashion
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)
            latent_X = self.autoencoder(data, latent=True)
            latent_X = latent_X.detach().cpu().numpy()
            elem_count = np.bincount(cluster_id_list[batch_idx],
                                     minlength=self.args.n_clusters)
            
            for k in range(self.args.n_clusters):
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(
                    latent_X[cluster_id_list[batch_idx] == k], k)
        
        # [Step-3] Update the network parameters
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)            
            loss, rec_loss, dist_loss = self._loss(data, 
                                                   cluster_id_list[batch_idx])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and batch_idx % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                      'Loss: {:.3f} | Dist-Loss: {:.3f}'
                print(msg.format(epoch, batch_idx,
                                 loss.detach().cpu().numpy(),
                                 rec_loss, dist_loss))
        