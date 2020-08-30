import torch
import numbers
import numpy as np
import torch.nn as nn
from clustering import Clustering
from autoencoder import AutoEncoder


class DCN(nn.Module):
    
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.autoencoder = AutoEncoder(args).to(self.device)
        self.clustering = Clustering(args)
        self.lamda = args.lamda
        self.criterion  = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.wd)
    
    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, centroid_id):
        
        batch_size = X.size()[0]
        
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder.latent_forward(X)

        # Reconstruction error
        rec_loss = self.criterion(X, rec_X)
        
        # Regularization term on the clustering performance
        dist_loss = torch.tensor(0.).to(self.device)
        for i in range(batch_size):
            centroid = torch.FloatTensor(
                self.clustering.centroids[centroid_id[i]]).to(self.device)
            diff_vec = latent_X[i] - centroid
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                     diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.lamda * torch.squeeze(sample_dist_loss)
        
        return (rec_loss + dist_loss, 
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())
    
    def pretrain(self, train_loader, epoch=10, verbose=True):
        
        if not self.args.pretrain:
            return
        
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value={}'
            raise ValueError(msg.format(epoch))
        
        if verbose:
            print('======== Start pretraining ========')
        
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
        
        if verbose:
            print('======== End pretraining ========\n')
        
        # Update centroids in self.clustering after pre-training
        # TODO: Efficient initialization with reservoir sampling on batches
        self.eval()
        batch_X = []
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_output = self.autoencoder.latent_forward(data)
            batch_X.append(latent_output.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.clustering.init_centroid(batch_X)

    def fit(self, epoch, train_loader, verbose=True):
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)
            
            # Update assignment results and centroids in the clustering module
            with torch.no_grad():
                latent_X = self.autoencoder.latent_forward(data)
                latent_X = latent_X.cpu().numpy()
                assert latent_X.shape[1] == self.args.n_centroids
                centroid_id = self.clustering.update_assign(latent_X)
                self.clustering.update_centroid(latent_X, centroid_id)
            
            # Update the network parameters
            loss, rec_loss, dist_loss = self._loss(data, centroid_id)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and batch_idx % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                      'Loss: {:.3f} | Dist-Loss: {:.3f}'
                print(msg.format(epoch, batch_idx,
                                 loss.detach().cpu().numpy(),
                                 rec_loss, dist_loss))
