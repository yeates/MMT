import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import os
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn as nn
from PIL import Image

from modules.MMT import MMTNet
from utils.utils import PSNR, stitch_images, coloring_parsing, save_parsing, color_edge
from utils.functions import bce2d, AdversarialLoss, Discriminator, VGG16FeatureExtractor

class Model():
    def __init__(self, args):
        self.lossNet = None
        self.iter = None
        self.l1_loss_val = 0.
        self.seg_loss_val = 0.
        self.edge_loss_val = 0.
        self.dataset = args.dataset
        
        self.summary = {}
        self.psnr = PSNR(255.0)
        self.mean_psnr = 0.
        self.board_writer = SummaryWriter(f"./logdir/")
        self.n_cate_anno = args.n_cate_anno
        self.auxiliary_type = args.auxiliary_type   # one of {"feature", "pred", "hybrid"}
        
        self.adv_loss = AdversarialLoss()
    
    def initialize_model(self, path = None, train = True, finetune = False):
        self.G = MMTNet(self.n_cate_anno, self.auxiliary_type)
        self.D = Discriminator(in_channels=1, use_sigmoid=True)
        if train: 
            self.__train(finetune=finetune)
            self.G = nn.DataParallel(self.G, [0,1])
            self.D = nn.DataParallel(self.D, [0,1])
        self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-4)
        self.optm_D = optim.Adam(self.D.parameters(), lr = 1e-5)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer', self.optm_G)])
            dirname, version = os.path.dirname(path), os.path.basename(path).split('_')[-1]
            load_ckpt(os.path.join(dirname, 'd_'+version), [('discriminator', self.D)], [('optimizer', self.optm_D)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 2e-5)
                self.optm_D = optim.Adam(self.D.parameters(), lr = 1e-6)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            self.iter = 0
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
            self.G.cuda()
            self.D.cuda()
            self.psnr.cuda()
            self.adv_loss.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, save_path, iters=120000):
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        pbar = tqdm(total=iters)
        if self.iter > 0: pbar.update(self.iter)
        while self.iter<iters:
            for items in train_loader:
                
                real, real_e, real_s, masks = self.__cuda__(*items)
                masked = real * masks
                fake, comp, fake_e, fake_s, ms_pred_auxiliary = self.forward(masked, masks, real)
                self.run_discriminator_one_step(real, fake, comp, real_e, fake_e, real_s, fake_s, masks, ms_pred_auxiliary)
                self.run_generator_one_step(real, fake, comp, real_e, fake_e, real_s, fake_s, masks, ms_pred_auxiliary)
                
                self.iter += 1
                pbar.update(1)
                pbar.set_postfix(psnr=self.psnr_val)
                
                if self.iter % 50 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, psnr:%.4f, edge_loss:%.4f, seg_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/50, self.mean_psnr/50, self.edge_loss_val/50, self.seg_loss_val/50, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                    self.mean_psnr = 0.
                    self.seg_loss_val = 0.
                    self.edge_loss_val = 0.
                
                if self.iter % 1000 == 0:
                    if self.iter % 10000 == 0:
                        if not os.path.exists('{:s}'.format(save_path)):
                            os.makedirs('{:s}'.format(save_path))
                        save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter), [('generator', self.G)], [('optimizer', self.optm_G)], self.iter)
                        save_ckpt('{:s}/d_{:d}.pth'.format(save_path, self.iter), [('discriminator', self.D)], [('optimizer', self.optm_D)], self.iter)
                        print('save model to ' + save_path)
                    
                    colored_parsings = []
                    for idx in range(real_s.shape[0]):
                        parsing = real_s[idx]
                        colored_parsings.append(np.array(coloring_parsing(parsing.permute(1,2,0).cpu().data.numpy())))
                    colored_gt_parsings = torch.Tensor(np.stack(colored_parsings, axis=0)).cuda().int()    # [bs, h, w, c]; uint8; [0, 255]
                    
                    bs, n_cate_anno, h, w = fake_s.shape
                    fake_anno = torch.zeros(bs, n_cate_anno, h, w).to(self.device).scatter(1, torch.argmax(fake_s, 1).view(bs, 1, h, w), 1)
                    colored_parsings = []
                    for idx in range(fake_anno.shape[0]):
                        parsing = fake_anno[idx]
                        colored_parsings.append(np.array(coloring_parsing(parsing.permute(1,2,0).cpu().data.numpy())))
                    colored_fake_parsings = torch.Tensor(np.stack(colored_parsings, axis=0)).cuda().int()    # [bs, h, w, c]; uint8; [0, 255]
                    
                    images = stitch_images(
                        self.__postprocess(real),
                        self.__postprocess(real_e),
                        self.__postprocess(fake_e),
                        colored_gt_parsings,
                        colored_fake_parsings,
                        self.__postprocess(masked),
                        self.__postprocess(fake),
                        self.__postprocess(comp)
                    )
                    if not os.path.exists('{:s}'.format(f"samples/{self.dataset}")):
                        os.makedirs('{:s}'.format(f"samples/{self.dataset}"))
                    samples_save_path = f"samples/{self.dataset}/{self.iter}.png"
                    images.save(samples_save_path)
                    
                if self.iter >= iters: break
                    
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
            
    def test(self, test_loader, result_save_path, verbose):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        count = 0
        pbar = tqdm(total=len(test_loader))
        for items in test_loader:
            items, files_name =items[:-1], items[-1]
            gt_images, _, _, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            masks = torch.cat([masks]*1, dim = 1)
            fake_B, other = self.G(masked_images, masks)
            fake_edge, fake_anno, _ = other
            
            comp_B = fake_B * (1 - masks) + gt_images * masks
            os.makedirs('{:s}/inpainting/{}/'.format(result_save_path, self.dataset), exist_ok=True)
            
            if verbose:
                os.makedirs('{:s}/mask/{}/'.format(result_save_path, self.dataset), exist_ok=True)
                os.makedirs('{:s}/masked_img/{}/'.format(result_save_path, self.dataset), exist_ok=True)
                os.makedirs('{:s}/pred_edge/{}/'.format(result_save_path, self.dataset), exist_ok=True)
                os.makedirs('{:s}/pred_anno/{}/'.format(result_save_path, self.dataset), exist_ok=True)

                bs, n_cate_anno, h, w = fake_anno.shape
                fake_anno = torch.zeros(bs, n_cate_anno, h, w).to(self.device).scatter(1, torch.argmax(fake_anno, 1).view(bs, 1, h, w), 1)
                colored_parsings = []
                for idx in range(fake_anno.shape[0]):
                    parsing = fake_anno[idx]
                    colored_parsings.append(np.array(coloring_parsing(parsing.permute(1,2,0).cpu().data.numpy())))
                colored_fake_parsings = torch.Tensor(np.stack(colored_parsings, axis=0)).cuda().permute(0,3,1,2) / 255.

            for k in range(comp_B.size(0)):
                grid = make_grid(comp_B[k:k+1])
                file_path = '{:s}/inpainting/{}/{}'.format(result_save_path, self.dataset, files_name[k])
                save_image(grid, file_path)

                if verbose:
                    grid = make_grid(masked_images[k:k+1] +1 - masks[k:k+1])
                    file_path = '{:s}/masked_img/{}/{}'.format(result_save_path, self.dataset, files_name[k])
                    save_image(grid, file_path)

                    grid = make_grid(1 - masks[k:k+1])
                    file_path = '{:s}/mask/{}/{}'.format(result_save_path, self.dataset, files_name[k])
                    save_image(grid, file_path)

                    grid = make_grid(colored_fake_parsings[k:k+1])
                    file_path = '{:s}/pred_anno/{}/{}'.format(result_save_path, self.dataset, files_name[k])
                    save_image(grid, file_path)
                
                    grid = make_grid(color_edge(fake_edge[k:k+1], masks[k:k+1, 0]))  # color_edge(fake_edge[k:k+1], masks[k:k+1, 0])
                    file_path = '{:s}/pred_edge/{}/{}'.format(result_save_path, self.dataset, files_name[k])
                    save_image(grid, file_path)
                    
            count += 1
            pbar.update(1)

            if count >= 1000:
                break
                
    def forward(self, masked_image, mask, gt_images):
        fake_B, other = self.G(masked_image, mask)
        fake_edge, fake_anno, ms_pred_auxiliary = other
        comp_B = fake_B * (1 - mask) + gt_images * mask
        # ms_pred_auxiliary: [ms_pred_edge, ms_pred_anno]
        return fake_B, comp_B, fake_edge, fake_anno, ms_pred_auxiliary
    
    def run_generator_one_step(self, gt_images, fake_images, comp_images, gt_edge, fake_edges, gt_anno, fake_annos, masks, ms_pred_auxiliary):
        self.optm_G.zero_grad()
    
        loss_G = self.get_g_loss(gt_images, fake_images, comp_images, gt_edge, fake_edges, gt_anno, fake_annos, masks, ms_pred_auxiliary)

        loss_G.backward()
        self.optm_G.step()
    
    def run_discriminator_one_step(self, gt_images, fake_images, comp_images, gt_edge, fake_edges, gt_anno, fake_annos, masks, ms_pred_auxiliary):
        self.optm_D.zero_grad()
        
        loss_D = self.get_d_loss(gt_images, fake_images, gt_edge, fake_edges)
        
        loss_D.backward()
        self.optm_D.step()
    
    def get_g_loss(self, real, fake, comp, real_e, fake_e, real_s, fake_s, mask, ms_pred_auxiliary):
        is_pred = (self.auxiliary_type in ["pred", "hybrid"])    # pred multi-scale structure?(edge, segmentation map)
        real_B, fake_B, comp_B = real, fake, comp
        real_anno, fake_anno = real_s, fake_s
        real_edge, fake_edge = real_e, fake_e
        ms_pred_edge, ms_pred_anno = ms_pred_auxiliary

        # (1) inpainting loss
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.TV_loss(comp_B * (1 - mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - mask))

        psnr = self.psnr(self.__postprocess(comp_B), self.__postprocess(real_B))
        self.psnr_val = psnr.item()
        self.mean_psnr += self.psnr_val
    
        # (2) segmentation loss
        fake_anno_logits = F.log_softmax(fake_anno, dim=1)
        label_anno = torch.argmax(real_anno, 1)
        bs, h, w = label_anno.shape
        weight = (label_anno.view(bs, -1) == (self.n_cate_anno-1)).sum(1) == h * w  # for cityscapes datasets segmentation
        if weight.sum() == bs:  # if null segmentation notation
            seg_loss = 0.
        else:
            _seg_loss = F.nll_loss(fake_anno_logits, torch.argmax(real_anno, 1), reduce=False).view(bs, -1).mean(1) * (~weight)
            seg_loss = final_seg_loss = _seg_loss.sum() / (~weight).sum()
            if is_pred: # is multi-scale predicted segmentation
                ms_seg_losses = []
                for pred_seg in ms_pred_anno:
                    h, w = pred_seg.shape[-2:]
                    pred_seg_logits = F.log_softmax(pred_seg, dim=1)
                    _seg_loss = F.nll_loss(pred_seg_logits, torch.argmax(F.interpolate(real_anno.float(), [h,w]).long(), 1), reduce=False).view(bs, -1).mean(1) * (~weight)
                    ms_seg_losses.append(_seg_loss.sum() / (~weight).sum())
                ms_seg_loss = torch.sum(torch.stack(ms_seg_losses))
                seg_loss = (final_seg_loss + ms_seg_loss) / 4
            self.seg_loss_val += seg_loss.detach()
        
        # (3) edge loss
        edge_loss = final_edge_loss = bce2d(real_edge, fake_edge)
        if is_pred: # is multi-scale predicted segmentation
            ms_edge_losses = []
            for pred_edge in ms_pred_edge:
                h, w = pred_edge.shape[-2:]
                ms_edge_losses.append(bce2d(F.interpolate(real_edge, size=(h,w)), torch.sigmoid(pred_edge)))
            ms_edge_loss = torch.sum(torch.stack(ms_edge_losses))
            edge_loss = (final_edge_loss + ms_edge_loss) / 4
        self.edge_loss_val += edge_loss.detach()
        
        # (4) adversial loss
        fakes = fake_edge
        gen_edge_feat = self.D(fakes)
        gen_edge_loss = self.adv_loss(gen_edge_feat, True, False)

        self.__add_summary(self.board_writer, 'gan_loss/gen_adv', gen_edge_loss.item(), self.iter)
        self.__add_summary(self.board_writer, 'loss/tv', tv_loss.item() * 0.1, self.iter)
        self.__add_summary(self.board_writer, 'loss/style', style_loss.item() * 120, self.iter)
        self.__add_summary(self.board_writer, 'loss/perceptual', preceptual_loss.item() * 0.05, self.iter)
        self.__add_summary(self.board_writer, 'loss/valid', valid_loss.item(), self.iter)
        self.__add_summary(self.board_writer, 'loss/hole', hole_loss.item() * 6, self.iter)
        self.__add_summary(self.board_writer, 'metric/psnr', psnr.item(), self.iter)
        self.__add_summary(self.board_writer, 'loss/seg_ce', seg_loss * 0.5, self.iter)
        self.__add_summary(self.board_writer, 'loss/edge_bce', edge_loss.item() * 0.5, self.iter)

        loss_G = (  
                    tv_loss * 0.1 
                    + gen_edge_loss
                    + style_loss * 120
                    + preceptual_loss * 0.05
                    + valid_loss * 1
                    + hole_loss * 6
                    + seg_loss * 0.5
                    + edge_loss * 0.5
                )

        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G
    
    def get_d_loss(self, real, fake, real_e, fake_e):
        reals, fakes = real_e, fake_e

        real_feat = self.D(reals)               
        fake_feat = self.D(fakes.detach())  
        dis_real_loss = self.adv_loss(real_feat, True, True)    
        dis_fake_loss = self.adv_loss(fake_feat, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        self.__add_summary(
            self.board_writer, 'gan_loss/dis_adv', dis_loss.item(), self.iter)
        
        return dis_loss


    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)

    def __add_summary(self, writer, name, val, iteration, prompt=False):
        INTERVAL = 10
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if prompt or (writer is not None and iteration % INTERVAL == 0):
            writer.add_scalar(name, self.summary[name]/INTERVAL, iteration)
            self.summary[name] = 0

    def __train(self, mode=True, finetune=False):
        if mode:
            super(MMTNet, self.G).train(mode)
        if finetune:
            for name, module in self.G.named_modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.InstanceNorm2d):
                    module.eval()
                    
    def __postprocess(self, img):
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()