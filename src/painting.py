import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import numpy as np
from brush_stroke import BrushStroke
from param2stroke import get_param2img

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Painting(nn.Module):
    def __init__(self, opt, n_strokes=None, background_img=None, brush_strokes=None):
        # h, w are canvas height and width in pixels
        super(Painting, self).__init__()
        self.n_strokes = n_strokes

        self.background_img = background_img

        if self.background_img is not None and self.background_img.shape[1] == 3: # add alpha channel
            t =  torch.zeros((1,1,self.background_img.shape[2],self.background_img.shape[3])).to(device)
            # t[:,:3] = self.background_img
            self.background_img = torch.cat((self.background_img, t), dim=1)

        if brush_strokes is None:
            self.brush_strokes = nn.ModuleList([BrushStroke(opt, ink=opt.ink) for _ in range(n_strokes)])
        else:
            self.brush_strokes = nn.ModuleList(brush_strokes)
        
        self.param2img = get_param2img(opt)

    def get_optimizers(self, multiplier=1.0, ink=False):
        xt = []
        yt = []
        a = []
        length = []
        z = []
        bend = []
        color = []
        
        for n, p in self.named_parameters():
            if "xt" in n.split('.')[-1]: xt.append(p)
            if "yt" in n.split('.')[-1]: yt.append(p)
            if "a" in n.split('.')[-1]: a.append(p)
            if "stroke_length" in n.split('.')[-1]: length.append(p)
            if "stroke_z" in n.split('.')[-1]: z.append(p)
            if "stroke_bend" in n.split('.')[-1]: bend.append(p)
            if "color_transform" in n.split('.')[-1]: color.append(p)

        position_opt = torch.optim.RMSprop(xt + yt, lr=5e-3*multiplier)
        rotation_opt = torch.optim.RMSprop(a, lr=1e-2*multiplier)
        color_opt = None if ink else torch.optim.RMSprop(color, lr=5e-3*multiplier)
        bend_opt = torch.optim.RMSprop(bend, lr=3e-3*multiplier)
        length_opt = torch.optim.RMSprop(length, lr=3e-3*multiplier)
        thickness_opt = torch.optim.RMSprop(z, lr=1e-2*multiplier)
        
        # t=5
        # position_opt = torch.optim.Adam(xt + yt, lr=5e-3*multiplier*t)
        # rotation_opt = torch.optim.Adam(a, lr=1e-2*multiplier*t)
        # color_opt = torch.optim.Adam(color, lr=5e-3*multiplier*t)
        # bend_opt = torch.optim.Adam(bend, lr=3e-3*multiplier*t)
        # length_opt = torch.optim.Adam(length, lr=1e-2*multiplier*t)
        # thickness_opt = torch.optim.Adam(z, lr=1e-2*multiplier*t)

        return position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt


    def forward(self, h, w, use_alpha=True, return_alphas=False, opacity_factor=1.0, efficient=False, zoom_factor=1):
        if self.background_img is None:
            canvas = torch.ones((1,4,h,w)).to(device)
        else:
            canvas = T.Resize((h,w), bicubic, antialias=True)(self.background_img).detach()
        canvas[:,3] = 1 # alpha channel

        mostly_opaque = False#True
        if return_alphas: stroke_alphas = []

        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(h,w, self.param2img, zoom_factor=zoom_factor)

            if mostly_opaque: single_stroke[:,3][single_stroke[:,3] > 0.5] = 1.
            if return_alphas: stroke_alphas.append(single_stroke[:,3:])
            
            if efficient:
                mask = single_stroke[:,3:].detach()>0.5
                mask = torch.cat([mask,]*4, dim=1)
                canvas[mask] *= 0
                canvas[mask] += 1
                canvas[:,:3][mask[:,:3]] *= single_stroke[:,:3][mask[:,:3]]
            else:
                if use_alpha:
                    canvas = canvas * (1 - single_stroke[:,3:]*opacity_factor) + single_stroke[:,3:]*opacity_factor * single_stroke
                else:
                    canvas = canvas[:,:3] * (1 - single_stroke[:,3:]*opacity_factor) + single_stroke[:,3:]*opacity_factor * single_stroke[:,:3]
        
        if return_alphas: 
            alphas = torch.cat(stroke_alphas, dim=1)
            # alphas, _ = torch.max(alphas, dim=1) ###################################################
            alphas = torch.sum(alphas, dim=1) ###################################################
            return canvas, alphas
        
        return canvas

    def get_alpha(self, h, w):
        # return the alpha values of the strokes of the painting

        stroke_alphas = []
        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(h,w)

            stroke_alphas.append(single_stroke[:,3:])

        alphas = torch.cat(stroke_alphas, dim=1)
        alphas, _ = torch.max(alphas, dim=1)#alphas.max(dim=1)
        return alphas

    def to_csv(self):
        ''' To csv string '''
        csv = ''
        for bs in self.brush_strokes:
            # Translation in proportions from top left
            # x = str((bs.transformation.weights[1].detach().cpu().item()+1)/2)
            # y = str((bs.transformation.weights[2].detach().cpu().item()+1)/2)
            # r = str(bs.transformation.weights[0].detach().cpu().item())
            x = str((bs.transformation.xt[0].detach().cpu().item()+1)/2)
            y = str((bs.transformation.yt[0].detach().cpu().item()+1)/2)
            r = str(bs.transformation.a[0].detach().cpu().item())
            length = str(bs.stroke_length.detach().cpu().item())
            thickness = str(bs.stroke_z.detach().cpu().item())
            bend = str(bs.stroke_bend.detach().cpu().item())
            alpha = str(bs.stroke_alpha.detach().cpu().item())
            color = bs.color_transform.detach().cpu().numpy()
            csv += ','.join([x,y,r,length,thickness,bend,alpha,str(color[0]),str(color[1]),str(color[2])])
            csv += '\n'
        csv = csv[:-1] # remove training newline
        return csv

    def validate(self):
        ''' Make sure all brush strokes have valid parameters '''
        for s in self.brush_strokes:
            BrushStroke.make_valid(s)


    def cluster_colors(self, n_colors):
        colors = [b.color_transform[:3].detach().cpu().numpy() for b in self.brush_strokes]
        colors = np.stack(colors)[None,:,:]

        from sklearn.cluster import KMeans
        from paint_utils3 import rgb2lab, lab2rgb
        # Cluster in LAB space
        colors = rgb2lab(colors)
        kmeans = KMeans(n_clusters=n_colors, random_state=0)
        kmeans.fit(colors.reshape((colors.shape[0]*colors.shape[1],3)))
        colors = [kmeans.cluster_centers_[i] for i in range(len(kmeans.cluster_centers_))]

        colors = np.array(colors)

        # Back to rgb
        colors = lab2rgb(colors[None,:,:])[0]
        return torch.from_numpy(colors).to(device)# *255., labels
    
    def pop(self):
        ''' Remove and return first stroke in the plan '''
        bs = self.brush_strokes[0]
        # Remove the stroke
        self.brush_strokes = nn.ModuleList([self.brush_strokes[i] for i in range(1,len(self.brush_strokes),1)])
        return bs
    
    def __len__(self):
        return len(self.brush_strokes)