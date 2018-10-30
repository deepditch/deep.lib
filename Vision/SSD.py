import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import util

class StdConv(nn.Module):
    def __init__(self, n_in, n_out, stride=2, drop_p=0.1):
        super().__init__()
        self.conv = nn.Conv2d(n_in, n_out, 3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(n_out)
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x):
        return self.dropout(self.batch_norm(self.relu(self.conv(x))))


def flatten_conv(x,k=1):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)


class SSDOut(nn.Module):
    def __init__(self, n_in, k=1):
        super().__init__()
        self.out_classes = nn.Conv2d(n_in, (num_classes + 1) * k, 3, padding=1) # Output for each class + background class
        self.out_boxes = nn.Conv2d(n_in, 4*k, 3, padding=1) # Output for bounding boxes
        
    def forward(self, x):
        return [flatten_conv(self.out_classes(x), k), F.tanh(flatten_conv(self.out_boxes(x), k))]


class SSDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv_0 = StdConv(512, 256, stride=1)
        self.conv_1 = StdConv(256, 256)
        self.out = SSDOut(256)
        
    def forward(self, x):
        x = self.dropout(F.relu(x))
        x = self.conv_0(x)
        x = self.conv_1(x)
        return self.out(x) 


class SSD_MultiHead(nn.Module):
    def __init__(self, k, bias):
        super().__init__()
        self.drop = nn.Dropout(.4)
        self.sconv0 = StdConv(512,256, stride=1, drop_p=.4)
        self.sconv1 = StdConv(256,256, drop_p=.4)
        self.sconv2 = StdConv(256,256, drop_p=.4)
        self.sconv3 = StdConv(256,256, drop_p=.4)
        self.out0 = SSDOut(256, k)
        self.out1 = SSDOut(256, k)
        self.out2 = SSDOut(256, k)
        self.out3 = SSDOut(256, k)

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out1(x)
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        return [torch.cat([o1c,o2c,o3c], dim=1),
                torch.cat([o1l,o2l,o3l], dim=1)]       

def center_to_corners(bb): 
    x1 = bb[:,0] - bb[:,2] / 2
    y1 = bb[:,1] - bb[:,3] / 2
    x2 = bb[:,0] + bb[:,2] / 2
    y2 = bb[:,1] + bb[:,3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def intersect(box_a, box_b, log=False):
    if log: print("Intersect"); print("box_a: ", box_a); print("box_b: ", box_b)
        
    corn_a = center_to_corners(box_a)
    corn_b = center_to_corners(box_b)
    
    if log: print("corn_a: ", corn_a); print("corn_b: ", corn_b)
    
    max_xy = torch.min(corn_a[:, None, 2:].double(), corn_b[None, :, 2:].double())
    min_xy = torch.max(corn_a[:, None, :2].double(), corn_b[None, :, :2].double())
    
    if log: print("max_xy: ", max_xy); print("min_xy: ", min_xy)
    
    inter = torch.clamp((max_xy - min_xy), min=0)
    
    if log: print("inter: ", inter)
    
    return inter[:, :, 0] * inter[:, :, 1]

def box_size(b): return (b[:, 2] * b[:, 3]) # Input [_, _, width, height]

def jaccard(box_a, box_b, log=False):
    inter = intersect(box_a, box_b, log)
    union = box_size(box_a).double().unsqueeze(1) + box_size(box_b).double().unsqueeze(0) - inter
    return inter / union

class SSD_Loss:
    def __init__(self, anchors, grid_sizes):
        self.anchors = anchors
        self.grid_sizes = grid_sizes

    """
    Remove padding from labels. 
    Split concatenated bounding boxes into arrays of 4. 
    Divide bounding box values by image size.
    Labels are padded with -1s so they are the same shape and may be broadcast together in batches.
    """
    def format_label(self, bbs, classes, log=False):
        if log: print("format_label"); print("bbs: ", bbs); print("classes: ", classes)
        bbs = bbs.view(-1,4)/imsize
        keep_idxs = (classes>-1).nonzero().view(-1)
        if log: print("Output"); print("bbs: ", bbs[keep_idxs]); print("classes: ", classes[keep_idxs])
        return bbs[keep_idxs], classes[keep_idxs]


    """ 
    Maps bounding box outputs to bounding boxes. 
    The model's bounding box outputs are not bounding boxes and instead represent changes to the anchor boxes.  
    """
    def map_bb_outputs_to_pred_bbs(self, outputs, anchors, log=False):
        if log: print("map_bb_outputs_to_pred_bbs"); print("outputs :", outputs); print("anchors :", anchors)
        
        # The first two values in the output represent a translation of the anchor box's center.
        # Grid size is the width and height of the receptive field
        # delta_center is bounded on the range (-grid_size / 2, grid_size / 2); 
        # that is, the center remains within the original receptive field. 
        delta_center = outputs[:,:2] / 2 * grid_sizes 
        
        if log: print("delta_center :", delta_center); print("grid_sizes :", grid_sizes)
        
        # The last two values in the output represent the width and height of the bounding box.
        # These values are interpreted as a precentage of the original anchor box's width and height.
        # percent_sizes is on the range (.5, 1.5). We add 1 since actn_bbs is on the range (-1, 1)
        percent_sizes = outputs[:,2:] / 2 + 1 
        
        if log: print("percent_sizes :", percent_sizes);
        
        actn_centers = delta_center + anchors[:,:2]  # Calculate predicted center_x and center_y  
        actn_wh = percent_sizes * anchors[:,2:]      # Calculate predicted width and height
        
        if log: print("returns :", torch.cat([actn_centers, actn_wh], dim=1));
        
        return torch.cat([actn_centers, actn_wh], dim=1)


    def map_label_to_ground_truth(self, raw_label_bbs, raw_label_classes, anchors, log=False):
        label_bbs, label_classes = format_label(raw_label_bbs, raw_label_classes)
            
        if log: print("map_label_to_ground_truth"); print("label_bbs: ", label_bbs); print("label_classes: ", label_classes)
        
        overlaps = jaccard(label_bbs, anchors)
        
        if log: print("overlaps: ", overlaps)
        
        prior_overlap, prior_idx = overlaps.max(1)
        
        if log: print("prior_overlap: ", prior_overlap); print("prior_idx: ", prior_idx)
        
        gt_overlap, gt_idx = overlaps.max(0)
        
        if log: print("gt_overlap: ", gt_overlap); print("gt_idx: ", gt_idx)
        
        gt_overlap[prior_idx] = 1.99
        
        for i,o in enumerate(prior_idx): gt_idx[o] = i
            
        if log: print("gt_overlap: ", gt_overlap); print("gt_idx: ", gt_idx)
            
        gt_classes = label_classes[gt_idx]
        
        if log: print("gt_classes: ", gt_classes)
        
        matches = gt_overlap > 0.4
        
        if log: print("matches: ", matches)
        
        matching_idxs = torch.nonzero(matches)[:,0]
        
        if log: print("matching_idxs: ", matching_idxs)
        
        gt_classes[matches != 1] = 0
        
        gt_bbs = label_bbs[gt_idx]
        
        if log: print("gt_classes: ", gt_classes[matching_idxs]); print("gt_bbs: ", gt_bbs[matching_idxs]);
            
        return gt_bbs, gt_classes, matching_idxs


    """ ssd loss for a single example """
    def ssd_1_loss(self, pred_classes, bb_outputs, label_classes, label_bbs):      
        gt_bbs, gt_classes, matching_idxs = map_label_to_ground_truth(label_bbs, label_classes, self.anchors)
        
        pred_bbs = map_bb_outputs_to_pred_bbs(bb_outputs, self.anchors)
        
        loc_loss = ((pred_bbs[matching_idxs].float() - gt_bbs[matching_idxs].float()).abs()).mean()
        
        clas_loss  = loss_f(pred_classes, gt_classes)
        
        return loc_loss, clas_loss


    """ ssd loss for a batch """
    def __call__(self, preds, target, log=False):
        total_location_loss = 0.
        total_class_loss = 0.
        
        for pred_clas, pred_bb, label_clas, label_bb in zip(*preds, target["CAT"], target["BB"]):
            
            loc_loss, clas_loss = ssd_1_loss(pred_clas, pred_bb, 
                                            label_clas, label_bb)
            
            total_location_loss += loc_loss
            total_class_loss += clas_loss
            
        if log: print(f'location: {total_location_loss.data[0]}, class: {total_class_loss.data[0]}')
            
        return total_location_loss + total_class_loss