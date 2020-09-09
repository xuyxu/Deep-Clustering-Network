import torch
import argparse
import numpy as np
from DCN import DCN
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

def evaluate(model, test_loader):
    y_test = []
    y_pred = []
    for data, target in test_loader:
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.kmeans.update_assign(latent_X).reshape(-1, 1))
    
    y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    return (normalized_mutual_info_score(y_test, y_pred),
            adjusted_rand_score(y_test, y_pred))

def solver(args, model, train_loader, test_loader):
    
    rec_loss_list = model.pretrain(train_loader)
    nmi_list = []
    ari_list = []

    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)
        
        model.eval()
        NMI, ARI = evaluate(model, test_loader)  # evaluation on the test_loader
        nmi_list.append(NMI)
        ari_list.append(ARI)
        
        print('\nEpoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}\n'.format(
            e, NMI, ARI))

    return rec_loss_list, nmi_list, ari_list


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Deep Clustering Network')

    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist', 
                        help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=28*28, 
                        help='input dimension')
    parser.add_argument('--n-classes', type=int, default=10, 
                        help='output dimension')

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4, 
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256, 
                        help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=100, 
                        help='number of epochs to train')
    parser.add_argument('--pretrain', type=bool, default=True, 
                        help='whether use pre-training')
    
    # Model parameters
    parser.add_argument('--lamda', type=float, default=1, 
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1, 
                        help='coefficient of the regularization term on ' \
                            'clustering')
    parser.add_argument('--hidden-dims', default=[500, 500, 2000, 10], 
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=10, 
                        help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=10, 
                        help='number of clusters in the latent space')

    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1, 
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True, 
                        help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100, 
                        help='how many batches to wait before logging the ' \
                            'training status')

    args = parser.parse_args()
    
    # Load data
    transformer = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),
                                                           (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=True, download=False, 
                       transform=transformer), 
        batch_size=args.batch_size, shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=False, transform=transformer), 
        batch_size=args.batch_size, shuffle=True)

    # Main body
    model = DCN(args)    
    rec_loss_list, nmi_list, ari_list = solver(
        args, model, train_loader, test_loader)
