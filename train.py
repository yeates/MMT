
import os
import argparse
from model import Model
from dataset import Dataset
from torch.utils.data import DataLoader


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CelebAMask-HQ', help='dataset name. set of {CelebAMask-HQ, OST, CityScape}')
    parser.add_argument('--data_root', type=str, default='../training_data/')
    parser.add_argument('--num_iters', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--gpu_id', type=str, default="0,1")
    parser.add_argument('--auxiliary_type', type=str, default="feature", help="auxiliary information type in MSSA")
    parser.add_argument('--local_canny_edge', type=bool, default=False, help="load canny edge map from local for True, generate canny in real-time for False")
    args = parser.parse_args()
    
    args.img_root, args.edge_root, args.anno_root = os.path.join(args.data_root + args.dataset + '/img/train/'), \
                                    os.path.join(args.data_root + args.dataset + '/edge/train/'), \
                                    os.path.join(args.data_root + args.dataset + '/anno/train/')
    args.model_save_path = os.path.join('checkpoint/', args.dataset)
    args.resume_ckpt = f'checkpoint/{args.dataset}/g_{args.num_iters}.pth'
    if args.dataset == 'CelebAMask-HQ':
        args.n_cate_anno = 15
    elif args.dataset == 'OST':
        args.n_cate_anno = 9
    elif args.dataset == 'CityScape':
        args.n_cate_anno = 20
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = Model(args)
    # pretrain
    model.initialize_model('', True, False)
    model.cuda()
    dataset = Dataset(args.img_root, args.edge_root, args.anno_root, mask_path = '', mask_mode = 1, \
                        target_size=256, n_anno_cate = args.n_cate_anno, mask_reverse = True, local_canny=args.local_canny_edge)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
    model.train(dataloader, args.model_save_path, args.num_iters)
    # fintune
    model = Model(args)
    model.initialize_model(args.resume_ckpt, True, True)
    model.cuda()
    dataset = Dataset(args.img_root, args.edge_root, args.anno_root, mask_path = '', mask_mode = 1, \
                        target_size=256, n_anno_cate = args.n_cate_anno, mask_reverse = True, local_canny=args.local_canny_edge)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
    model.train(dataloader, args.model_save_path, args.num_iters*3)
        
        
if __name__ == '__main__':
    run()