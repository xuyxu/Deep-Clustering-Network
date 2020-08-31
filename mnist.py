import torch
import argparse
import numpy as np
from dcn import DCN
from torchvision import datasets, transforms
from sklearn.metrics import normalized_mutual_info_score

best = float('-inf')

def evaluate(model, test_loader):
    y_test = []
    y_pred = []
    for data, target in test_loader:
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)
        latent_X = model.autoencoder(data, latent=True)
        latent_X = latent_X.detach().cpu().numpy()

        y_test.append(target.view(-1, 1).numpy())
        y_pred.append(model.clustering.update_assign(latent_X).reshape(-1, 1))
    
    y_test = np.vstack(y_test).reshape(-1)
    y_pred = np.vstack(y_pred).reshape(-1)
    return normalized_mutual_info_score(y_test, y_pred)

def solver(args, model, train_loader, test_loader):
    global best
    model.pretrain(train_loader)

    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)
        
        model.eval()
        score = evaluate(model, train_loader)
        best = max(score, best)
        
        print('\nEpoch: {:02d} | NMI: {:.3f} | Best: {:.3f}\n'.format(
            e, score, best))


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
    parser.add_argument('--epoch', type=int, default=50, 
                        help='number of epoch to train')
    parser.add_argument('--pretrain', type=bool, default=True, 
                        help='whether use pre-training')

    # Model parameters
    parser.add_argument('--lamda', type=float, default=1, 
                        help='coefficient of the regularization term on ' \
                            'clustering')
    parser.add_argument('--hidden-dims', default=[500, 500, 2000, 10], 
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=10, 
                        help='latent space dimension')
    parser.add_argument('--n-centroids', type=int, default=10, 
                        help='number of centroids in the latent space')

    # Utility parameters
    parser.add_argument('--seed', type=int, default=0, 
                        help='random seed')
    parser.add_argument('--n-jobs', type=int, default=1, 
                        help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True, 
                        help='whether to use GPU')
    parser.add_argument('--evaluate', type=bool, default=True, 
                        help='whether evaluate testing dataset')
    parser.add_argument('--save', type=bool, default=True, 
                        help='whether save model parameters and training logs')
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
        batch_size=args.batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dir, train=False, transform=transformer), 
        batch_size=args.batch_size, shuffle=True)

    model = DCN(args)    
    solver(args, model, train_loader, test_loader)
