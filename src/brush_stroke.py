import copy
import math
import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import warnings
import numpy as np

from param2stroke import special_sigmoid


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
        https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    # roll = (roll/360.)*2.*math.pi
    # pitch = (pitch/360.)*2.*math.pi
    # yaw = (yaw/360.)*2.*math.pi
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def spherical_to_quaternion(theta, phi):
    return np.cos(theta)*np.sin(phi), \
            np.sin(theta)*np.sin(phi), \
            np.cos(phi)

cos = torch.cos
sin = torch.sin
def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(a.device)
    a = -1.*a
    A[0,0,0] = cos(a)
    A[0,0,1] = -sin(a)
    A[0,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) + xt#xt
    A[0,1,0] = sin(a)
    A[0,1,1] = cos(a)
    A[0,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) + yt#yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A

class RigidBodyTransformation(nn.Module):
    def __init__(self, a, xt, yt, init_differentiably=False, device='cuda'):
        super(RigidBodyTransformation, self).__init__()

        if init_differentiably:
            self.xt = xt
            self.yt = yt 
            self.a = a
        else:
            self.xt = nn.Parameter(torch.ones(1, device=device)*xt)
            self.yt = nn.Parameter(torch.ones(1, device=device)*yt)
            self.a = nn.Parameter(torch.ones(1, device=device)*a)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        # M = rigid_body_transform(self.weights[0], self.weights[1]*(w/2), self.weights[2]*(h/2), anchor_x, anchor_y)
        M = rigid_body_transform(self.a[0], self.xt[0]*(w/2), self.yt[0]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings(): # suppress annoing torchgeometry warning
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class BrushStroke(nn.Module):
    def __init__(self, 
                 opt,
                stroke_length=None, stroke_z=None, stroke_bend=None, stroke_alpha=None,
                color=None, 
                ink=False,
                a=None, xt=None, yt=None,
                init_differentiably=False,
                device='cuda'):
        super(BrushStroke, self).__init__()

        self.MAX_STROKE_LENGTH = opt.MAX_STROKE_LENGTH
        self.MIN_STROKE_LENGTH = opt.MIN_STROKE_LENGTH
        self.MIN_STROKE_Z = opt.MIN_STROKE_Z
        self.MAX_ALPHA = opt.MAX_ALPHA
        self.MAX_BEND = opt.MAX_BEND


        if init_differentiably:
            # self.xt = xt # Range [0,1]
            # self.yt = yt # Range [0,1]
            # self.a = a # Range [-2pi,2pi]
            self.transformation = RigidBodyTransformation(a, xt, yt, 
                    init_differentiably=init_differentiably)

            self.stroke_length = stroke_length
            self.stroke_z = stroke_z
            self.stroke_bend = stroke_bend
            self.stroke_alpha = stroke_alpha
        else:
            if color is None: color=(torch.rand(3).to(device)*.4)+0.3
            if a is None: a=(torch.rand(1, device=device)*2-1)*3.14
            if xt is None: xt=(torch.rand(1, device=device)*2-1)
            if yt is None: yt=(torch.rand(1, device=device)*2-1)


            if stroke_length is None: stroke_length=torch.rand(1, device=device)*self.MAX_STROKE_LENGTH
            if stroke_z is None: stroke_z = torch.rand(1, device=device).clamp(self.MIN_STROKE_Z, 0.95)
            if stroke_alpha is None: stroke_alpha=(torch.rand(1, device=device)*2-1)*self.MAX_ALPHA
            if stroke_bend is None: stroke_bend = (torch.rand(1, device=device)*2 - 1) * self.MAX_BEND
            stroke_bend = min(stroke_bend, stroke_length) if stroke_bend > 0 else max(stroke_bend, -1*stroke_length)

            self.transformation = RigidBodyTransformation(a, xt, yt)
            
            self.stroke_length = stroke_length
            self.stroke_z = stroke_z
            self.stroke_bend = stroke_bend
            self.stroke_alpha = stroke_alpha

            self.stroke_length.requires_grad = True
            self.stroke_z.requires_grad = True
            self.stroke_bend.requires_grad = True
            self.stroke_alpha.requires_grad = True

            self.stroke_length = nn.Parameter(self.stroke_length)
            self.stroke_z = nn.Parameter(self.stroke_z)
            self.stroke_bend = nn.Parameter(self.stroke_bend)
            self.stroke_alpha = nn.Parameter(self.stroke_alpha)

        if not ink:
            self.color_transform = nn.Parameter(color)
        else:
            self.color_transform = torch.zeros(3).to(device)

    def forward(self, h, w, param2img, zoom_factor=1):
        # # Do rigid body transformation
        # full_param = torch.zeros((1,16)).to(self.stroke_length.device)
        
        # # X
        # full_param[0,0] = 0
        # full_param[0,4] = self.stroke_length/3 
        # full_param[0,8] = 2*self.stroke_length/3
        # full_param[0,12] = self.stroke_length
        # # Y
        # full_param[0,1] = 0
        # full_param[0,5] = self.stroke_bend
        # full_param[0,9] = self.stroke_bend
        # full_param[0,13] = 0
        # # Z
        # full_param[0,2] = 0.2
        # full_param[0,6] = self.stroke_z
        # full_param[0,10] = self.stroke_z
        # full_param[0,14] = 0.2
        # # alpha
        # full_param[0,3] = self.stroke_alpha
        # full_param[0,7] = self.stroke_alpha
        # full_param[0,11] = self.stroke_alpha
        # full_param[0,15] = self.stroke_alpha

        # full_param = torch.zeros((1,16)).to(self.stroke_length.device)
        full_param = torch.cat([self.stroke_length, self.stroke_bend, self.stroke_z, self.stroke_alpha]).to(self.stroke_length.device)
        full_param = full_param.unsqueeze(0)

        # model_ind = np.random.randint(len(param2imgs))
        # # print(model_ind)
        # stroke = param2imgs[model_ind](full_param).unsqueeze(0)
        # stroke = pad_for_full(stroke)
        stroke = param2img(full_param, h, w).unsqueeze(0)

        # Pad 1 or two to make it fit
        # print('ffff', stroke.shape, h, w)
        if stroke.shape[2] != h or stroke.shape[3] != w:
            stroke = T.Resize((h, w), bicubic, antialias=True)(stroke)

        if zoom_factor > 1:
            # Calculate the new dimensions
            new_height = int(stroke.shape[2] / zoom_factor)
            new_width = int(stroke.shape[3] / zoom_factor)

            # Create a center crop transform
            center_crop = T.CenterCrop((new_height, new_width))

            # Apply the transform to the image
            stroke = center_crop(stroke)
            stroke = T.Resize((h, w), bicubic, antialias=True)(stroke)

        if zoom_factor > 1:
            h2,w2 = int(h/2), int(w/2)
            stroke[:,0,h2-5:h2+5, w2-5:w2+5] *= 0
            # stroke[:,0,h2-5:h2+5, w2-5:w2+5] += 0.15

        # x = self.transformation(strokes[self.stroke_ind].permute(2,0,1).unsqueeze(0))
        # from plan import show_img
        # show_img(stroke)
        x = self.transformation(stroke)

        # Remove stray color from the neural network being sloppy
        # x[x < 0.1] = 0
        # # x[x >= 0.1] = 1
        #x = 1/(1+torch.exp(-1.*((x*2-1)+0.5) / 0.05))
        x = special_sigmoid(x)
        # import kornia as K
        # x = K.filters.median_blur(x, (3,3))
        

        # show_img(x)
        x = torch.cat([x,x,x,x], dim=1)
        # print('forawrd brush', x.shape)
        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)


        return x

    def make_valid(stroke):
        with torch.no_grad():
            og_len = stroke.stroke_length.item()
            stroke.stroke_length.data.clamp_(stroke.MIN_STROKE_LENGTH+0.002, 
                                             stroke.MAX_STROKE_LENGTH-0.002)
            # if stroke.stroke_length.item() != og_len:
            #     print('length constrained')
            
            stroke.stroke_bend.data.clamp_(-1*stroke.stroke_length, stroke.stroke_length)
            stroke.stroke_bend.data.clamp_(-1.0*stroke.MAX_BEND, stroke.MAX_BEND)

            stroke.stroke_alpha.data.clamp_(-1.0*stroke.MAX_ALPHA, stroke.MAX_ALPHA)

            stroke.stroke_z.data.clamp_(stroke.MIN_STROKE_Z,1.0-0.01)

            # stroke.transformation.weights[1:3].data.clamp_(-1.,1.)
            stroke.transformation.xt.data.clamp_(-1.,1.)
            stroke.transformation.yt.data.clamp_(-1.,1.)


            #stroke.color_transform.data.clamp_(0.02,0.75)
            if stroke.color_transform.min() < 0.35:
                # If it's a colored stroke, don't let it go to a flourescent color
                stroke.color_transform.data.clamp_(0.02,0.70)
            else:
                # Well balanced RGB, less constraint
                stroke.color_transform.data.clamp_(0.02,0.85)
    
    def simple_parameterization_to_bezier_points(stroke_length, bend, z, alpha=0):
        xs = (np.arange(4)/3.) * stroke_length

        trajectory=[
            [xs[0], 0, .2, alpha],
            [xs[1], bend, z, alpha],
            [xs[2], bend, z, alpha],
            [xs[3], 0, .2, alpha],
        ]
        return trajectory

    def execute(self, painter, x_start, y_start, rotation, step_size=.005, curve_angle_is_rotation=False):
        # x_start, y_start in global coordinates. rotation in radians
        # curve_angle_is_rotation if true, then the brush is angled constantly down towards theta
        smooth = True
        if smooth:
            all_positions = []
            all_orientations = []

        # Need to translate x,y a bit to be accurate according to camera
        if painter.H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x_start, y_start, 1.])
            real_coords = painter.H_coord.dot(sim_coords)
            x_start, y_start = real_coords[0]/real_coords[2], real_coords[1]/real_coords[2]

        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        trajectory = BrushStroke.simple_parameterization_to_bezier_points(
                self.stroke_length.detach().cpu().item(),
                self.stroke_bend.detach().cpu().item(),
                self.stroke_z.detach().cpu().item(),
                self.stroke_alpha.detach().cpu().item(),
            )
        path = BrushStroke.get_rotated_trajectory(rotation, trajectory)
        
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.03, speed=0.4)
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.005, speed=0.1)

        p0 = path[0,0], path[0,1], path[0,2]
        p3 = None

        alpha = self.stroke_alpha.item() # Same alpha throughout stroke, for now.
        # print('alpha', alpha)

        n_points_off_canvas, n_points_total = 0, 0

        for i in range(1, len(path)-1, 3):
            p1 = path[i+0,0], path[i+0,1], path[i+0,2]
            p2 = path[i+1,0], path[i+1,1], path[i+1,2]
            p3 = path[i+2,0], path[i+2,1], path[i+2,2]

            stroke_length = ((p3[0]-p0[0])**2 + (p3[1] - p0[1])**2)**.5
            n = max(2, int(stroke_length/step_size))
            n=10#5#30 # TODO: something more than this? see previous line
            for t in np.linspace(0,1,n):
                x = (1-t)**3 * p0[0] \
                      + 3*(1-t)**2*t*p1[0] \
                      + 3*(1-t)*t**2*p2[0] \
                      + t**3*p3[0]
                y = (1-t)**3 * p0[1] \
                      + 3*(1-t)**2*t*p1[1] \
                      + 3*(1-t)*t**2*p2[1] \
                      + t**3*p3[1]
                if t < 0.333:
                    z = (1 - t/.333) * p0[2] + (t/.333)*p1[2]
                elif t < 0.666:
                    z = (1 - (t-.333)/.333) * p1[2] + ((t-.333)/.333)*p2[2]
                else:
                    z = (1 - (t-.666)/.333) * p2[2] + ((t-.666)/.333)*p3[2]

                def deriv_cubic_bez(p0,p1,p2,p3,t):
                    return -3*(1-t)**2*p0 \
                            + 3*(1-t)**2*p1 \
                            - 6*t*(1-t)*p1 \
                            - 3*t**2*p2 \
                            + 6*t*(1-t)*p2 \
                            + 3*t**2*p3
                dx_dt = deriv_cubic_bez(p0[0], p1[0], p2[0], p3[0], t)
                dy_dt = deriv_cubic_bez(p0[1], p1[1], p2[1], p3[1], t)
                dy_dx = dy_dt / dx_dt
                curve_angle = np.arctan(dy_dx)
                # print('curve_angle', curve_angle)

                def rad_to_deg(rad):
                    return 1.0*rad/math.pi * 180
                def deg_to_rad(deg):
                    return 1.0*deg/180*math.pi

                theta_sphere = np.arctan2(dy_dt, dx_dt) + np.pi/2 # the pi makes it perpendicular to trajectory

                if curve_angle_is_rotation:
                    theta_sphere = rotation

                phi_sphere = alpha
                # print(theta_sphere, phi_sphere)
                roll = np.cos(theta_sphere)*np.sin(phi_sphere)
                # pitch =  np.pi - np.sin(theta_sphere)*np.cos(phi_sphere)*np.sin(phi_sphere)
                pitch =  np.pi - np.sin(theta_sphere)*np.sin(phi_sphere)
                yaw = deg_to_rad(270.) # Constant yaw
                q = get_quaternion_from_euler(roll,pitch,yaw)

                ######
                # TODO: fix the tilt of the brush for the Franka robot
                q = None
                ####

                #brush_length = 0.095
                l = painter.opt.brush_length
                
                if l is not None:
                    r = l * np.sin(phi_sphere)
                    dx = r * np.cos(theta_sphere)
                    dy = r * np.sin(theta_sphere)
                    dz = l - l * np.cos(phi_sphere)
                    # print('dx dy', dx, dy)
                    x += dx
                    y += dy
                    #z -= dz
                    # print('dz', dz)
                    new_z_range = z_range * np.abs(np.cos(phi_sphere))
                    # print('z range', z_range, new_z_range)
                    # print('dz', dz)
                    # print('painter.Z_CANVAS', painter.Z_CANVAS)
                    # print('painter.Z_MAX_CANVAS', painter.Z_MAX_CANVAS)

                    z = painter.Z_CANVAS - z * new_z_range - dz #+ 0.07
                    # print(x,y,z)
                else:
                    z = painter.Z_CANVAS - z * z_range 

                x_next = x_start + x 
                y_next = y_start + y


                # If off the canvas, lift up
                if (x_next > painter.opt.X_CANVAS_MAX) or (x_next < painter.opt.X_CANVAS_MIN) or \
                        (y_next > painter.opt.Y_CANVAS_MAX) or (y_next < painter.opt.Y_CANVAS_MIN):
                    z += 0.005
                    n_points_off_canvas += 1
                n_points_total += 1

                # Don't over shoot the canvas
                x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
                y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)

                if smooth:
                    if t == 0 and i==0:
                        all_positions.append([x_next, y_next, z+0.02])
                        all_orientations.append(q)
                        all_positions.append([x_next, y_next, z+0.005])
                        all_orientations.append(q)
                    all_positions.append([x_next, y_next, z])
                    all_orientations.append(q)
                    if t == 1 and (i == len(path)-4):
                        all_positions.append([x_next, y_next, z+0.01])
                        all_orientations.append(q)
                        all_positions.append([x_next, y_next, z+0.02])
                        all_orientations.append(q)
                else:
                    if t == 0 and i==0:
                        painter.move_to(x_next, y_next, z+0.02, q=q, method='direct', speed=0.1)
                        painter.move_to(x_next, y_next, z+0.005, q=q, method='direct', speed=0.03)
                    painter.move_to(x_next, y_next, z, q=q, method='direct', speed=0.05)
                    if t == 1 and (i == len(path)-4):
                        painter.move_to(x_next, y_next, z+0.01, q=q, method='direct', speed=0.03)
                        painter.move_to(x_next, y_next, z+0.02, q=q, method='direct', speed=0.1)
                # time.sleep(0.02)
            p0 = p3

        stroke_complete = False
        if smooth and ((n_points_off_canvas/n_points_total) < 0.9):
            stroke_complete = painter.move_to_trajectories(all_positions, all_orientations)
        

        # Don't over shoot the canvas
        x_next = x_start+path[-1,0]
        y_next = y_start+path[-1,1]
        x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
        y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)
        painter.move_to(x_next, y_next, painter.Z_CANVAS + 0.04, speed=0.3)
        # painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

        return stroke_complete
    
    def get_rotated_trajectory(rotation, trajectory):
        # Rotation in radians
        ret = copy.deepcopy(trajectory)
        for i in range(len(ret)):
            ret[i][0] = math.cos(rotation) * trajectory[i][0] \
                     - math.sin(rotation) * trajectory[i][1]
            ret[i][1] = math.sin(rotation) * trajectory[i][0] \
                     + math.cos(rotation) * trajectory[i][1]
        ret = np.array(ret)
        return ret
    
    def dot_stroke(self, opt):
        return BrushStroke(
            opt,
            stroke_length=torch.zeros(1), 
            stroke_z=torch.ones(1)*0.5, 
            stroke_bend=torch.zeros(1), 
            stroke_alpha=torch.zeros(1)
        )