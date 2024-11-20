import copy
import datetime
import math
import os
import random
import sys
import numpy as np
import torch 
from torch import nn
from torchvision import models, transforms
import clip
from tqdm import tqdm
from torchvision.models import vgg16, resnet18
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
from datasets import load_dataset

from brush_stroke import BrushStroke
from cofrida import get_instruct_pix2pix_model
from options import Options
from paint_utils3 import format_img, load_img, show_img
from painting import Painting
from my_tensorboard import TensorBoard
from losses.clip_loss import CLIPConvLoss, Dict2Class, clip_conv_loss, clip_conv_loss_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StrokePredictorEncoder(nn.Module):
    def __init__(self):
        super(StrokePredictorEncoder, self).__init__()
    

class PaintTransformer(nn.Module):

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        factor = 2
        cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32*factor, 3, 1),
            nn.BatchNorm2d(32*factor),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32*factor, 64*factor, 3, 2),
            nn.BatchNorm2d(64*factor),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64*factor, 128*factor, 3, 2),
            nn.BatchNorm2d(128*factor),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128*factor, 128, 3, 2),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2)
            )
        self.curr_canvas_enc = copy.deepcopy(cnn)
        self.diff_enc = copy.deepcopy(cnn)
        self.target_enc = copy.deepcopy(cnn)

        self.conv = nn.Conv2d(128 * 3, hidden_dim, 1)

        self.dim_reducer = nn.Linear(32**2, hidden_dim)
        self.clip_feat_dim_reducer = nn.Linear(50, total_strokes)

        self.to_stroke_features = nn.Linear(hidden_dim, total_strokes)


        self.to_stroke_features2 = nn.Linear(hidden_dim+768, hidden_dim)

    def forward(self, target, curr_canvas, diff, clip_conv_feats):
        b, _, H, W = curr_canvas.shape
        target_feat = self.target_enc(target)
        curr_canvas_feat = self.curr_canvas_enc(curr_canvas)
        diff_feat = self.diff_enc(diff)
        h, w = target_feat.shape[-2:]
        feat = torch.cat([target_feat, curr_canvas_feat, diff_feat], dim=1)
        feat_conv = self.conv(feat)
        # print('feat_conv', feat_conv.shape)

        feat = feat_conv.flatten(2)
        # print('feat_conv', feat.shape)
        clip_conv_feats = clip_conv_feats[-1].permute(0,2,1).float() # 1x50x768 -> 1x768x50
        clip_conv_feats = self.clip_feat_dim_reducer(clip_conv_feats) # 1x768x50 -> 1x768x1
        clip_conv_feats = clip_conv_feats.permute(0,2,1) #  1x768x1 -> 1x1x768
        # print('clip_conv_feats', clip_conv_feats.shape)

        feat = self.dim_reducer(feat) # -> n x hidden_dim x hidden_dim
        # print('feat', feat.shape)
        feat = self.to_stroke_features(feat).permute(0,2,1) # n x hidden_dim x hidden_dim -> # n x n_strokes x hidden_dim
        # print('feat', feat.shape)

        feat = torch.cat([feat, clip_conv_feats], dim=2) # -> n x n_strokes x hidden_dim+768
        # print('feat', feat.shape)

        feat = self.to_stroke_features2(feat) # -> n x n_strokes x hidden_dim
        # print('feat', feat.shape)

        return feat


class StrokePredictor(nn.Module):
    def __init__(self, opt,
                 n_strokes=1):
        '''
            n_strokes (int) : number of strokes to predict with each forward pass
        '''
        super(StrokePredictor, self).__init__()
        self.n_strokes = n_strokes
        self.opt = opt

        # self.vit_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        #     "facebook/deit-tiny-patch16-224", "gaunernst/bert-tiny-uncased"
        #     # "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
        #     # "facebook/deit-small-distilled-patch16-224", "gaunernst/bert-tiny-uncased"
        # )
        # # Replace first layer with new conv2d that can take 5 channels instead of 3 (add coords)
        # # self.channel_reducer \
        # #     = torch.nn.Conv2d(5,3, kernel_size=(5, 5), padding='same')

        # # print(self.vit_model)

        # image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        # self.vit_model.config.decoder_start_token_id = tokenizer.cls_token_id
        # self.vit_model.config.pad_token_id = tokenizer.pad_token_id

        # from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig
        # config_encoder = ViTConfig(
        #     hidden_size=128,
        #     num_hidden_layers=2,
        #     num_attention_heads=8
        # )
        # config_decoder = BertConfig(
        #     hidden_size=64,
        #     num_hidden_layers=1,
        #     num_attention_heads=8
        # )
        # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        # self.vit_model = VisionEncoderDecoderModel(config=config)
        # tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        # self.vit_model.config.decoder_start_token_id = tokenizer.cls_token_id
        # self.vit_model.config.pad_token_id = tokenizer.pad_token_id

        # self.clip_model = clip_conv_loss_model#.model
        # self.clip_model, clip_preprocess = clip.load(
        #     'ViT-B/32', device, jit=False)

        clip_conv_layer_weights = [0, 0, 0, 0, 1.0]
        a = {'clip_model_name':'ViT-B/32','clip_conv_loss_type':'Cos','device':device,
            'num_aug_clip':1,'augemntations':['affine'],
            'clip_fc_loss_weight':0.0,'clip_conv_layer_weights':clip_conv_layer_weights}
        self.clip_model = CLIPConvLoss(Dict2Class(a))

        # self.vit_out_size = 768
        self.vit_out_size = 256


        self.vit_model = PaintTransformer(
            666, total_strokes=self.n_strokes, hidden_dim=self.vit_out_size, n_heads=8, n_enc_layers=2, n_dec_layers=2
        )

        def make_nn(in_dim, out_dim, hidden_dim, n_layers, activation=nn.LeakyReLU(0.2)):
            if n_layers == 1:
                return nn.Linear(in_dim, out_dim)
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)

            for i in range(n_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
            
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        # self.position_head = nn.Linear(self.vit_out_size, 2)
        # self.rotation_head = nn.Linear(self.vit_out_size, 1)

        # self.stroke_length_head = nn.Linear(self.vit_out_size, 1)
        # self.stroke_bend_head = nn.Linear(self.vit_out_size, 1)
        # self.stroke_alpha_head = nn.Linear(self.vit_out_size, 1)
        # self.stroke_z_head = nn.Linear(self.vit_out_size, 1)
        # self.color_head    = nn.Linear(self.vit_out_size, 3)

        h_dim = 64
        n_layers = 3
        # self.position_head = make_nn(self.vit_out_size, 2, hidden_dim=h_dim, n_layers=n_layers)
        self.x_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        self.y_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        self.rotation_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)

        self.stroke_length_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        self.stroke_bend_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        self.stroke_alpha_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        self.stroke_z_head = make_nn(self.vit_out_size, 1, hidden_dim=h_dim, n_layers=n_layers)
        # self.color_head    = nn.Linear(self.vit_out_size, 3)

        # self.resize_normalize = transforms.Compose([
        #     transforms.Resize((224,224), antialias=True),
        #     # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        # ])
        self.resize_normalize = transforms.Compose([
            transforms.Resize((256,256), antialias=True),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        self.resize_normalize_clip = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        # size_x, size_y = 224, 224
        # idxs_x = torch.arange(size_x) / size_x
        # idxs_y = torch.arange(size_y) / size_y
        # x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # G x G
        # self.coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).to(device)

        # Labels is [batch_size, sequence_length]
        self.labels = torch.zeros((16, self.n_strokes), device=device, dtype=int)

    def forward(self, current_canvas, target_canvas, n_strokes=None, training=True):
        '''
            Given the current canvas and the target, predict the next n_strokes
            return:
                List[List[BrushStroke()]] : batch_size x n_strokes
        '''

        current_canvas = self.resize_normalize(current_canvas)
        target_canvas = self.resize_normalize(target_canvas)

        diff = target_canvas - current_canvas

        with torch.no_grad():
            clip_input = self.resize_normalize_clip(diff)
            # print('clip_inptu', clip_input.shape)
            # clip_feats = self.clip_model.encode_image(clip_input)
            fc_features, clip_conv_feats = self.clip_model.visual_encoder(clip_input)
            # print('clip_feats', clip_conv_feats[-1].shape)

        # if len(self.coords) < len(diff):
        #     self.coords = self.coords.repeat(len(diff),1,1,1) # So you don't have to repeatedly create this
        # coords = self.coords[:len(diff)]
        # coords = coords * diff.mean(dim=1).unsqueeze(1).repeat(1,2,1,1)
        # diff = torch.cat([diff, coords], dim=1)
        # diff = self.channel_reducer(diff)

        n_strokes = n_strokes if n_strokes != None else self.n_strokes

        if (len(self.labels) < len(diff)) or (self.labels.shape[1] < n_strokes):
            self.labels = torch.zeros((len(current_canvas), n_strokes), device=device, dtype=int)

        # if training:
        #     # Labels is [batch_size, sequence_length]
        #     feats = self.vit_model(pixel_values=diff, output_hidden_states=True, labels=self.labels[:len(diff),:n_strokes])#.float()
        # else:
        #     feats = self.vit_model(pixel_values=diff, output_hidden_states=True)#.float()
            
        # feats = self.vit_model(target_canvas, current_canvas)
        feats = self.vit_model(target_canvas, current_canvas, diff, clip_conv_feats)
        
        # print('feats', feats)
            
        # print('encoder_last_hidden_state', feats.encoder_last_hidden_state.shape)
        # print('decoder_hidden_states', feats.decoder_hidden_states[-1].shape)
        # print('decoder_attentions', feats.decoder_attentions[-1].shape)

        # feats = feats.decoder_hidden_states[-1]
        # print(feats.shape)

        # position = self.position_head(feats)#.float()
        position_x = self.x_head(feats)#.float()
        position_y = self.y_head(feats)#.float()
        rotation = self.rotation_head(feats)#.float()
        # print('predicted rotation size', rotation.shape)
        # colors = self.color_head(feats)

        lengths = self.stroke_length_head(feats) / 1000. # m to mm
        bends = self.stroke_bend_head(feats) / 1000. # m to mm
        zs = self.stroke_z_head(feats)
        alphas = self.stroke_alpha_head(feats)

        # Change the values of these functions into the target range
        sigmoid = torch.nn.functional.sigmoid
        position_x = sigmoid(position_x)*2 - 1
        position_y = sigmoid(position_y)*2 - 1
        rotation = sigmoid(rotation)*7 - 3.14

        lengths = sigmoid(lengths)*opt.MAX_STROKE_LENGTH
        bends = opt.MAX_BEND*(sigmoid(bends)*2 - 1)
        zs = sigmoid(zs)
        alphas = sigmoid(alphas)*opt.MAX_ALPHA
        # print(position.mean(), lengths.mean(), bends.mean(), zs.mean(), alphas.mean())
        

        # Convert the output of the Transformer into BrushStroke Classes
        paintings = []
        brush_strokes_list = []
        for batch_ind in range(len(current_canvas)):
            predicted_brush_strokes = []
            for stroke_ind in range(n_strokes):
                a =     rotation[batch_ind, stroke_ind, :]
                # xt =    position[batch_ind, stroke_ind, :1]
                # yt =    position[batch_ind, stroke_ind, -1:]
                xt =    position_x[batch_ind, stroke_ind, :]
                yt =    position_y[batch_ind, stroke_ind, :]

                length = lengths[batch_ind, stroke_ind, :] 
                bend = bends[batch_ind, stroke_ind, :]
                z = zs[batch_ind, stroke_ind, :]
                alpha = alphas[batch_ind, stroke_ind, :]

                bs = BrushStroke(
                        self.opt, init_differentiably=True,
                        ink=True,  
                        stroke_length=length, stroke_bend=bend, 
                        stroke_alpha=alpha, stroke_z=z,
                        a=a,xt=xt, yt=yt
                )
                predicted_brush_strokes.append(bs)
            # predicted_painting = Painting(self.opt,
            #         background_img=current_canvas[batch_ind:batch_ind+1], 
            #         brush_strokes=predicted_brush_strokes)
            # paintings.append(predicted_painting)
            brush_strokes_list.append(predicted_brush_strokes)
        return brush_strokes_list

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def get_random_painting(opt, n_strokes=random.randint(0,4), background_img=None):
    painting = Painting(opt, n_strokes=n_strokes, background_img=background_img)
    return painting

def get_random_brush_strokes(opt, n_strokes=random.randint(0,20)):
    painting = Painting(opt, n_strokes=n_strokes)
    return painting.brush_strokes

l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()

param2img_global = None

def stroke_distance(bs0, bs1):
    # Euclidian distance between starting points of two strokes
    # return ((bs0.xt-bs1.xt)**2 + (bs0.yt-bs1.yt)**2)**0.5
    # return l1_loss(bs0.get_transformed_path(param2img_global),
    #                bs1.get_transformed_path(param2img_global))
    return ((bs0.transformation.xt-bs1.transformation.xt)**2 \
            + (bs0.transformation.yt-bs1.transformation.yt)**2)**0.5

from scipy.optimize import linear_sum_assignment
def match_brush_strokes(strokes0, strokes1):
    '''
        Re-order a list of strokes (strokes1) to have the strokes in an order
        such that comparing 1-to-1 to strokes0 minimizes differences in stroke position (x,y) 
        args:
            strokes0 List[BrushStroke()]
            strokes1 List[BrushStroke()]
        return:
            List[BrushStroke()] : the re-ordered strokes1
    '''
    # Create the cost matrix
    cost = np.empty((len(strokes0), len(strokes1)))
    for i in range(len(strokes0)):
        for j in range(len(strokes1)):
            cost[i,j] = stroke_distance(strokes0[i], strokes1[j])
    # print('cost\n', cost)
            
    # Perform linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # print(row_ind, col_ind)
    
    # Re-order strokes1 to reduce costs
    reordered_strokes1 = [strokes1[i] for i in col_ind]
    # print(len(reordered_strokes1), len(strokes1), len(strokes0))

    # Confirm that the costs go down or are equal when comparing
    # cost_prev_order = np.sum(np.array([stroke_distance(strokes0[i], strokes1[min(i,len(strokes1)-1)]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # cost_new_order = np.sum(np.array([stroke_distance(strokes0[i], reordered_strokes1[i]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # if (cost_prev_order-cost_new_order) < 0:
    #     print(cost_prev_order, cost_new_order, cost_prev_order-cost_new_order, sep='\t')
    
    return reordered_strokes1

def wasserstein_distance(bs0, bs1):
    # Adapted from https://arxiv.org/pdf/2108.03798
    mu_u = torch.cat([bs0.transformation.xt, bs0.transformation.yt])
    # print('mu_u', mu_u)
    def sigma_wasserstein(theta):
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return torch.Tensor([
            [cos_theta**2 + sin_theta**2, cos_theta*sin_theta],
            [cos_theta*sin_theta, sin_theta**2 + cos_theta**2]
        ])
    sigma_u = sigma_wasserstein(bs0.transformation.a)
    # print('sigma_u', sigma_u)
    mu_v = torch.cat([bs1.transformation.xt, bs1.transformation.yt])
    sigma_v = sigma_wasserstein(bs1.transformation.a)

    return l2_loss(mu_u, mu_v) \
        + torch.trace(sigma_u**2 + sigma_v**2 - (2*(sigma_u@(sigma_v**2)@sigma_u))**0.5)

def brush_stroke_parameter_loss_fcn(predicted_strokes, true_strokes, param2img):
    '''
        Calculate loss between brush strokes
        args:
            predicted_strokes List[List[BrushStroke()]]
            true_strokes List[List[BrushStroke()]]
    '''
    global param2img_global
    loss_x, loss_y, loss_rot, loss_wasserstein, loss_path = 0,0,0,0,0
    loss_length, loss_z, loss_bend, loss_alpha = 0,0,0,0

    for batch_ind in range(len(predicted_strokes)):
        with torch.no_grad():
            if param2img_global is None: param2img_global = param2img
            true_strokes_reordered = match_brush_strokes(predicted_strokes[batch_ind], true_strokes[batch_ind])
        for stroke_ind in range(len(predicted_strokes[batch_ind])):
            pred_bs = predicted_strokes[batch_ind][stroke_ind]
            # true_bs = true_strokes_reordered[stroke_ind]
            true_bs = true_strokes_reordered[min(len(true_strokes_reordered)-1, stroke_ind)]

            loss_x += l1_loss(pred_bs.transformation.xt, true_bs.transformation.xt)
            # print(pred_bs.xt, true_bs.xt)
            loss_y += l1_loss(pred_bs.transformation.yt, true_bs.transformation.yt)
            loss_rot += l1_loss(pred_bs.transformation.a, true_bs.transformation.a)

            loss_length += l1_loss(pred_bs.stroke_length, true_bs.stroke_length)
            loss_bend += l1_loss(pred_bs.stroke_bend, true_bs.stroke_bend)
            loss_z += l1_loss(pred_bs.stroke_z, true_bs.stroke_z)
            loss_alpha += l1_loss(pred_bs.stroke_alpha, true_bs.stroke_alpha)


            # print(true_bs.stroke_length, true_bs.stroke_bend, true_bs.stroke_z, true_bs.stroke_alpha)

            loss_wasserstein += wasserstein_distance(pred_bs, true_bs)

            # loss_path += l1_loss(pred_bs.get_transformed_path(param2img),
            #                      true_bs.get_transformed_path(param2img))

    n_batch, n_strokes = len(predicted_strokes), len(predicted_strokes[0])
    n = n_batch * n_strokes
    loss_x, loss_y, loss_rot, loss_path, loss_wasserstein = loss_x/n, loss_y/n, loss_rot/n, loss_path/n, loss_wasserstein/n
    loss_length, loss_z, loss_bend, loss_alpha = loss_length/n, loss_z/n, loss_bend/n, loss_alpha/n

    loss_length *= 1000 # mm to cm
    loss_bend *= 1000

    loss_x *= 10 # Veryy important
    loss_y *= 10

    loss = loss_x + loss_y + loss_rot + loss_length + loss_z + loss_bend + loss_alpha + loss_path #+ loss_wasserstein
    # loss = loss_path

    return loss, loss_x, loss_y, loss_rot, loss_length, loss_z, loss_bend, loss_alpha, loss_path, loss_wasserstein

def compute_stroke_parameter_loss():
    current_canvases = []
    target_canvases = []
    true_brush_strokes = []
    predicted_brush_strokes = []
    predicted_next_canvases = []

    # Get the ground truth data
    for it in range(batch_size):
        # Get a current canvas
        with torch.no_grad():
            current_painting = get_random_painting(opt, n_strokes=0,
                    background_img=current_canvas_aug(blank_canvas)).to(device)
            current_canvas = current_painting(h_render, w_render, use_alpha=False, zoom_factor=zoom_factor)[:,:3]
            current_canvas = current_canvas_aug(current_canvas)
            current_canvases.append(current_canvas)

        # Generate a random brush stroke(s) to add. Render it to create the target canvas
        with torch.no_grad():
            true_brush_stroke = get_random_brush_strokes(opt, 
                    n_strokes=opt.n_gt_strokes)
                    # n_strokes=random.randint(opt.n_predicted_strokes, 20)) # Variable number of target strokes

            # Render the strokes onto the current canvas
            target_painting = Painting(opt, background_img=current_canvases[it], 
                    brush_strokes=true_brush_stroke).to(device)
            target_canvas = target_painting(h_render, w_render, use_alpha=False, zoom_factor=zoom_factor)

            true_brush_strokes.append(true_brush_stroke)
            target_canvases.append(target_canvas)
        
    current_canvases = torch.cat(current_canvases, dim=0)[:,:3]
    target_canvases = torch.cat(target_canvases, dim=0)

    # Augment the target_canvases to reduce sim2real gap
    with torch.no_grad():
        target_canvases = target_img_aug(target_canvases)

    # Perform the prediction to estimate the added stroke(s)
    if opt.n_predicted_strokes_low is not None:
        n_strokes = random.randint(opt.n_predicted_strokes_low,opt.n_predicted_strokes_high) 
    else: 
        n_strokes = opt.n_predicted_strokes
    predicted_brush_strokes = stroke_predictor(current_canvases, target_canvases,
            n_strokes=n_strokes)

    loss, loss_x, loss_y, loss_rot, loss_length, loss_z, loss_bend, loss_alpha, loss_path, loss_wasserstein \
            = brush_stroke_parameter_loss_fcn(predicted_brush_strokes, true_brush_strokes, param2img=current_painting.param2img)
    
    loss.backward(retain_graph=True)

    # Log losses
    if batch_ind % 10 == 0:
        opt.writer.add_scalar('loss/loss_stroke_parameters', loss, batch_ind)

        opt.writer.add_scalar('loss/loss_stroke_length', loss_length, batch_ind)
        opt.writer.add_scalar('loss/loss_stroke_z', loss_z, batch_ind)
        opt.writer.add_scalar('loss/loss_stroke_bend', loss_bend, batch_ind)
        opt.writer.add_scalar('loss/loss_stroke_alpha', loss_alpha, batch_ind)
        opt.writer.add_scalar('loss/loss_x', loss_x, batch_ind)
        opt.writer.add_scalar('loss/loss_y', loss_y, batch_ind)
        opt.writer.add_scalar('loss/loss_rot', loss_rot, batch_ind)
        opt.writer.add_scalar('loss/loss_path', loss_path, batch_ind)
        opt.writer.add_scalar('loss/loss_wasserstein', loss_wasserstein, batch_ind)

        opt.writer.add_scalar('loss/lr', optim.param_groups[0]['lr'], batch_ind)

    # Log images
    if batch_ind % 200 == 0:
        with torch.no_grad():
            if len(predicted_next_canvases) == 0:
                # Render the predicted strokes
                for it in range(batch_size):
                    # Render the strokes onto the current canvas
                    predicted_painting = Painting(opt, background_img=current_canvases[it:it+1], 
                                brush_strokes=predicted_brush_strokes[it]).to(device)
                    predicted_next_canvas = predicted_painting(h_render, w_render, use_alpha=False, zoom_factor=zoom_factor)[:,:3]
                    predicted_next_canvases.append(predicted_next_canvas)
                predicted_next_canvases = torch.cat(predicted_next_canvases, dim=0)
            # Log some images
            for log_ind in range(min(10, batch_size)):
                # Log canvas with gt strokes, canvas with predicted strokes
                t = target_canvases[log_ind:log_ind+1].clone()
                t[:,:,:,-2:] = 0
                log_img = torch.cat([t, predicted_next_canvases[log_ind:log_ind+1]], dim=3)
                opt.writer.add_image('images_stroke_param/train{}'.format(str(log_ind)), 
                        format_img(log_img), batch_ind)
                
                # Log target_strokes-current_canvas and predicted_strokes-current_canvas
                pred_diff_img = torch.abs(predicted_next_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                true_diff_img = torch.abs(target_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                
                pred_diff_img_bool = pred_diff_img.mean(dim=1) > 0.3
                true_diff_img_bool = true_diff_img.mean(dim=1) > 0.3
                colored_img = torch.zeros(true_diff_img.shape).to(device)
                colored_img[:,1][pred_diff_img_bool & true_diff_img_bool] = 1 # Green for true positives
                colored_img[:,0][~pred_diff_img_bool & true_diff_img_bool] = 1 # Red for False negatives
                colored_img[:,2][pred_diff_img_bool & ~true_diff_img_bool] = 1 # Blue for False positives
                pred_diff_img[:,:,:,:2] = 1 # Draw a border
                colored_img[:,:,:,:2] = 1 # Draw a border

                log_img = torch.cat([true_diff_img, pred_diff_img, colored_img], dim=3)
                opt.writer.add_image('images_stroke_param/train{}_diff'.format(str(log_ind)), 
                        format_img(log_img), batch_ind)
    return loss

target_canvas_bank = []
cofrida_starting_canvases_og = None
parti_prompts = None
# zoom_factor = 1 # Global variable for how much to zoom in on strokes. Bigger initial strokes help the learning algorithm converge

def compute_cofrida_loss():
    global target_canvas_bank, cofrida_starting_canvases_og, parti_prompts

    if parti_prompts is None:
        dataset = load_dataset("nateraw/parti-prompts")['train']
        dataset = dataset.filter(lambda example: \
                                 (example["Challenge"] == 'Basic') \
                                 or (example["Challenge"] == 'Simple Detail')) 
        parti_prompts = dataset['Prompt']

    # Generate a new bunch of target images every few iterations
    if batch_ind % 25 == 0:
        with torch.no_grad():
            # Generate target images
            target_canvas_bank = []
            og_blank_canvas_dims = (blank_canvas.shape[2], blank_canvas.shape[3])
            blank_canvas_512 = transforms.Resize((512,512), antialias=True)(blank_canvas)
            for it in range(batch_size*5):
                target_canvas_bank.append(sd_interactive_pipeline(
                    parti_prompts[random.randint(0,len(parti_prompts)-1)], 
                    blank_canvas_512, num_inference_steps=20, 
                    num_images_per_prompt=1,
                    output_type='pt',
                    # image_guidance_scale=2.5,#1.5 is default
                ).images[0].unsqueeze(0))
            target_canvas_bank = torch.cat(target_canvas_bank, dim=0).clamp(0,1)
            target_canvas_bank = transforms.Resize((og_blank_canvas_dims), antialias=True)(target_canvas_bank)
        
    if cofrida_starting_canvases_og is None:
        with torch.no_grad():
            cofrida_starting_canvases = blank_canvas.repeat((batch_size, 1,1,1)).to(device)
            cofrida_starting_canvases_og = cofrida_starting_canvases.detach().clone()

    # Get target canvases from the canvas bank
    with torch.no_grad():
        n_bank = len(target_canvas_bank)
        # Random sample from bank
        rand_ind = torch.randperm(n_bank)
        target_canvases = target_canvas_bank[rand_ind[:batch_size]]
        # Augment the target_canvases to reduce sim2real gap
        target_canvases = cofrida_target_img_aug(target_canvases).float()

    # Compute the loss
    current_canvases = cofrida_starting_canvases_og.clone()
    custom_pix_loss_tot, pix_loss_tot, clip_loss_tot, loss_tot = 0,0,0,0
    for pred_it in range(opt.num_prediction_rounds): # num times to predict a batch of strokes
        predicted_brush_strokes = []
        predicted_next_canvases = []
        # optim.zero_grad()

        # Perform the prediction to estimate the added stroke(s)
        predicted_brush_strokes = stroke_predictor(current_canvases, target_canvases)

        # Render the predicted strokes
        for it in range(batch_size):
            

            # Render the strokes onto the current canvas
            predicted_painting = Painting(opt, background_img=current_canvases[it:it+1], 
                        brush_strokes=predicted_brush_strokes[it]).to(device)
            predicted_next_canvas = predicted_painting(h_render, w_render, use_alpha=False, zoom_factor=zoom_factor)
            predicted_next_canvases.append(predicted_next_canvas.detach())

            # Calculate losses. pix_loss in pixel space, and stroke_param_loss in stroke space
            target_canvas = target_canvases[it:it+1]

            pix_loss = pix_loss_fcn(predicted_next_canvas, target_canvas)
            custom_pix_loss = custom_pix_loss_fcn(predicted_next_canvas, target_canvas)
            clip_loss = clip_conv_loss(target_canvas, predicted_next_canvas[:,:3]) 

            # loss = custom_pix_loss*0.25 + clip_loss*0.75
            loss = clip_loss
            
            custom_pix_loss_tot += custom_pix_loss.item()
            pix_loss_tot += pix_loss.item()
            clip_loss_tot += clip_loss.item()
            loss_tot += loss.item()

            loss.backward(retain_graph=True)

        predicted_next_canvases = torch.cat(predicted_next_canvases, dim=0)
        current_canvases = predicted_next_canvases.detach()

    # Log losses
    if batch_ind % 10 == 0:
        opt.writer.add_scalar('loss/pix_loss', pix_loss_tot, batch_ind)
        opt.writer.add_scalar('loss/custom_pix_loss', custom_pix_loss_tot, batch_ind)
        opt.writer.add_scalar('loss/clip_loss', clip_loss_tot, batch_ind)
        opt.writer.add_scalar('loss/loss_cofrida', loss_tot, batch_ind)

        opt.writer.add_scalar('loss/lr', optim.param_groups[0]['lr'], batch_ind)

    # Log images
    if batch_ind % 50 == 0:
        with torch.no_grad():
            # Log some images
            for log_ind in range(min(20, batch_size)):
                # Log canvas with gt strokes, canvas with predicted strokes
                t = target_canvases[log_ind:log_ind+1].clone()
                t[:,:,:,-2:] = 0
                log_img = torch.cat([t, predicted_next_canvases[log_ind:log_ind+1]], dim=3)
                opt.writer.add_image('images_cofrida/train{}'.format(str(log_ind)), 
                        format_img(log_img), batch_ind)
                
                # Log target_strokes-current_canvas and predicted_strokes-current_canvas
                pred_diff_img = torch.abs(predicted_next_canvases[log_ind:log_ind+1] - cofrida_starting_canvases_og[log_ind:log_ind+1])
                true_diff_img = torch.abs(target_canvases[log_ind:log_ind+1] - cofrida_starting_canvases_og[log_ind:log_ind+1])
                
                pred_diff_img_bool = pred_diff_img.mean(dim=1) > 0.3
                true_diff_img_bool = true_diff_img.mean(dim=1) > 0.3
                colored_img = torch.zeros(true_diff_img.shape).to(device)
                colored_img[:,1][pred_diff_img_bool & true_diff_img_bool] = 1 # Green for true positives
                colored_img[:,0][~pred_diff_img_bool & true_diff_img_bool] = 1 # Red for False negatives
                colored_img[:,2][pred_diff_img_bool & ~true_diff_img_bool] = 1 # Blue for False positives
                pred_diff_img[:,:,:,:2] = 1 # Draw a border
                colored_img[:,:,:,:2] = 1 # Draw a border

                log_img = torch.cat([true_diff_img, pred_diff_img, colored_img], dim=3)
                opt.writer.add_image('images_cofrida/train{}_diff'.format(str(log_ind)), 
                        format_img(log_img), batch_ind)
                    
    return loss_tot

if __name__ == '__main__':
    global zoom_factor
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/sp_{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    save_dir = os.path.join('./stroke_predictor_models/', run_name)
    os.makedirs(save_dir, exist_ok=True)

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    blank_canvas = load_img('../cofrida/blank_canvas.jpg',h=h_render, w=w_render).to(device)[:,:3]/255.

    stroke_predictor = StrokePredictor(opt, 
            n_strokes=opt.n_predicted_strokes)
            # n_strokes=2)
    if opt.continue_training is not None:
        stroke_predictor.load_state_dict(torch.load(opt.continue_training))

    stroke_predictor.to(device)
    for param in stroke_predictor.vit_model.parameters(): # These might not be necessary
        param.requires_grad = True
    for param in stroke_predictor.parameters():
        param.requires_grad = True

    print('# of parameters in stroke_predictor: ', get_n_params(stroke_predictor))
    print('# of parameters in stroke_length_head: ', get_n_params(stroke_predictor.stroke_length_head))
    print('# of parameters in vit_model: ', get_n_params(stroke_predictor.vit_model))
    # print('# of parameters in encoder: ', get_n_params(stroke_predictor.vit_model.encoder))
    # print('# of parameters in decoder: ', get_n_params(stroke_predictor.vit_model.decoder))
    print('# of parameters in vit_model curr_canvas_enc: ', get_n_params(stroke_predictor.vit_model.curr_canvas_enc))
    print('# of parameters in vit_model dim_reducer: ', get_n_params(stroke_predictor.vit_model.dim_reducer))
    # print('# of parameters in vit_model enc_img: ', get_n_params(stroke_predictor.vit_model.enc_img))
    # print('# of parameters in vit_model transformer: ', get_n_params(stroke_predictor.vit_model.transformer))
    
    optim = torch.optim.Adam(stroke_predictor.parameters(), lr=opt.sp_lr)

    pix_loss_fcn = torch.nn.L1Loss()

    def custom_pix_loss_fcn(canvas, goal_canvas):
        black_mask = (goal_canvas <= 0.2).float() # B x CANVAS_SIZE x CANVAS_SIZE
        white_mask = 1 - black_mask # B x CANVAS_SIZE x CANVAS_SIZE
        l2 = ((canvas - goal_canvas) ** 2) # B x CANVAS_SIZE x CANVAS_SIZE
        black_loss = (l2 * black_mask).mean(1).mean(1).mean(1) # B
        white_loss = (l2 * white_mask).mean(1).mean(1).mean(1) # B
        return (0.6 * black_loss + 0.4 * white_loss).mean() # 1
    
    # torch.autograd.set_detect_anomaly(True)
    batch_size = opt.sp_training_batch_size

    sd_interactive_pipeline = get_instruct_pix2pix_model(
        opt.cofrida_model, 
        device=device)
    
    import torchvision.transforms.v2 as transforms
    # target_img_aug = transforms.RandomPhotometricDistort(
    #     brightness=(0.75,1.25),
    #     contrast=(0.3,1.7),
    #     saturation=(0.3,1.7),
    #     hue=(-0.1,0.1),
    #     p=0.75
    # )
    # current_canvas_aug = transforms.RandomPhotometricDistort(
    #     brightness=(0.75,1.25),
    #     contrast=(0.3,1.7),
    #     saturation=(0.3,1.7),
    #     hue=(-0.1,0.1),
    #     p=0.75
    # )
    # blank_canvas_aug = transforms.Compose([
    #     current_canvas_aug,
    #     transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0), antialias=True)
    # ])
    # cofrida_target_img_aug = transforms.Compose([
    #     transforms.RandomPhotometricDistort(
    #         brightness=(0.75,1.05),
    #         contrast=(0.5,1.5),
    #         saturation=(0.5,1.5),
    #         hue=(-0.1,0.1),
    #         p=0.75
    #     ),
    #     transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0), antialias=True)
    # ])
    target_img_aug = transforms.RandomPhotometricDistort(
        brightness=(0.95,1.05),
        contrast=(0.95,1.05),
        saturation=(0.95,1.05),
        hue=(-0.05,0.05),
        p=0.75
    )
    current_canvas_aug = transforms.RandomPhotometricDistort(
        brightness=(0.95,1.05),
        contrast=(0.95,1.05),
        saturation=(0.95,1.05),
        hue=(-0.05,0.05),
        p=0.75
    )
    blank_canvas_aug = transforms.Compose([
        current_canvas_aug,
        transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0), antialias=True)
    ])
    cofrida_target_img_aug = transforms.Compose([
        transforms.RandomPhotometricDistort(
            brightness=(0.95,1.05),
            contrast=(0.95,1.05),
            saturation=(0.95,1.05),
            hue=(-0.05,0.05),
        ),
        transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0), antialias=True)
    ])

    total_epochs = 100000
    for batch_ind in tqdm(range(total_epochs)):
        # Use zoom_factor to make strokes really big so it can learn well initially, then reduce size
        zoom_factor = 1#max(1, 3*(1-batch_ind/total_epochs))

        optim.param_groups[0]['lr'] *= 0.99995
        stroke_predictor.train()
        optim.zero_grad()

        loss_stroke_param = compute_stroke_parameter_loss() if opt.sp_stroke_loss_weight > 0 else 0
        loss_cofrida = compute_cofrida_loss() if opt.sp_cofrida_loss_weight > 0 else 0

        loss_total = (loss_stroke_param * opt.sp_stroke_loss_weight) + (loss_cofrida * opt.sp_cofrida_loss_weight)

        # loss_total.backward(retain_graph=True) # It's already been back propagated
        optim.step()

        if batch_ind % 10 == 0:
            opt.writer.add_scalar('loss/loss', loss_total, batch_ind)

            # print(stroke_predictor.position_head[0].weight)

        # Periodically save
        if batch_ind % 1000 == 0:
            torch.save(stroke_predictor.state_dict(), os.path.join(save_dir, 'stroke_predictor_weights.pth'))

