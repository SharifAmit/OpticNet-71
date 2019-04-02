import argparse
import keras
from src.dataloader import Kermany2018,Srinivasan2014

def train(data_dir, logdir, input_size, dataset, batch_size, weights, epoch, pre_trained_model):
    
    if dataset=='Srinivasan2014':
        train_batches, test_batches = Srinivasan2014(batch_size, input_size, data_dir)
    else :
        train_batches, test_batches = Kermany2018(batch_size, input_size, data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Choosing between 2 OCT datasets', choices=['Srinivasan2014','Kermany2018'])
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--weights', type=str,default=None, help='Resuming training from previous weights')
    parser.add_argument('--model',type=str, default=None,help='Pretrained weights for transfer learning',choices=['ResNet50',
                                 'MobileNetV2','Xception'])

    args = parser.parse_args()
    train(args.datadir, args.logdir, args.input_dim, args.dataset, args.batch, args.weights, args.epoch, args.model)