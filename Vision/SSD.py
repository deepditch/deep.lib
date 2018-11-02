from Vision.FocalLoss import FocalLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from validation import _AccuracyMeter
import util
import numpy as np

def torch_center_to_corners(bb): 
    x1 = bb[:,0] - bb[:,2] / 2
    y1 = bb[:,1] - bb[:,3] / 2
    x2 = bb[:,0] + bb[:,2] / 2
    y2 = bb[:,1] + bb[:,3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def torch_corners_to_center(bb):
    if len(bb.size()) == 0: return bb

    center_x = (bb[:,0] + bb[:,2]) / 2
    center_y = (bb[:,1] + bb[:,3]) / 2
    width = bb[:,2] - bb[:,0]
    height = bb[:,3] - bb[:,1]
    return torch.stack([center_x, center_y, width, height], dim=1)

def intersect(box_a, box_b, log=False):
    if log: print("Intersect"); print("box_a: ", box_a); print("box_b: ", box_b)
        
    corn_a = torch_center_to_corners(box_a)
    corn_b = torch_center_to_corners(box_b)
    
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

"""
Remove padding from labels. 
Split concatenated bounding boxes into arrays of 4. 
Divide bounding box values by image size.
Labels are padded with -1s so they are the same shape and may be broadcast together in batches.
"""
def format_label(bbs, classes, imsize, log=False):
    if log: print("format_label"); print("bbs: ", bbs); print("classes: ", classes)
    bbs = bbs.view(-1,4)/imsize
    keep_idxs = (classes>-1).nonzero().view(-1)
    if log: print("Output"); print("bbs: ", bbs[keep_idxs]); print("classes: ", classes[keep_idxs])
    return bbs[keep_idxs], classes[keep_idxs]


""" 
Maps bounding box outputs to bounding boxes. 
The model's bounding box outputs are not bounding boxes and instead represent changes to the anchor boxes.  
"""
def map_bb_outputs_to_pred_bbs(outputs, anchors, grid_sizes, log=False):
    if log: print("map_bb_outputs_to_pred_bbs"); print("outputs :", outputs); print("anchors :", anchors)
    
    # The first two values in the output represent a translation of the anchor box's center.
    # Grid size is the width and height of the receptive field
    # delta_center is bounded on the range (-grid_size / 2, grid_size / 2); 
    # that is, the center remains within the original receptive field. 
    delta_center = outputs[:,:2] / 2 * util.to_gpu(grid_sizes) 
    
    if log: print("delta_center :", delta_center); print("grid_sizes :", grid_sizes)
    
    # The last two values in the output represent the width and height of the bounding box.
    # These values are interpreted as a precentage of the original anchor box's width and height.
    # percent_sizes is on the range (.5, 1.5). We add 1 since actn_bbs is on the range (-1, 1)
    percent_sizes = outputs[:,2:] / 2 + 1 
    
    if log: print("percent_sizes :", percent_sizes);
    
    actn_centers = delta_center + util.to_gpu(anchors)[:,:2]  # Calculate predicted center_x and center_y  
    actn_wh = percent_sizes * util.to_gpu(anchors)[:,2:]      # Calculate predicted width and height
    
    if log: print("returns :", torch.cat([actn_centers, actn_wh], dim=1));
    
    return torch.cat([actn_centers, actn_wh], dim=1)


def map_label_to_ground_truth(raw_label_bbs, raw_label_classes, anchors, imsize, log=False):
    label_bbs, label_classes = format_label(raw_label_bbs, raw_label_classes, imsize)
        
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
    
    if log: print("gt_classes: ", gt_classes[matching_idxs]); print("gt_bbs: ", gt_bbs[matching_idxs])
   
    return util.to_gpu(gt_bbs), gt_classes, util.to_gpu(matching_idxs)


class SSDLoss():
    def __init__(self, anchors, grid_sizes, num_classes, imsize):
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.imsize = imsize
        self.loss_f = FocalLoss(num_classes)

    """ ssd loss for a single example """
    def single_example_loss(self, pred_classes, bb_outputs, label_classes, label_bbs):      
        gt_bbs, gt_classes, matching_idxs = map_label_to_ground_truth(label_bbs, label_classes, self.anchors, self.imsize)
        
        pred_bbs = map_bb_outputs_to_pred_bbs(bb_outputs, self.anchors, self.grid_sizes)
        
        loc_loss = ((pred_bbs[matching_idxs].float() - gt_bbs[matching_idxs].float()).abs()).mean()
        
        clas_loss = self.loss_f(pred_classes, gt_classes)
        
        return loc_loss, clas_loss

    def __call__(self, preds, target, log=False):
        total_location_loss = 0.
        total_class_loss = 0.
        
        for pred_clas, pred_bb, label_clas, label_bb in zip(*preds, target["CAT"], target["BB"]):
            
            loc_loss, clas_loss = self.single_example_loss(pred_clas, pred_bb, label_clas, label_bb)
            
            total_location_loss += loc_loss
            total_class_loss += clas_loss
            
        if log: print(f'location: {total_location_loss.data[0]}, class: {total_class_loss.data[0]}')
            
        return total_location_loss + total_class_loss


def non_maximum_supression(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def make_output(pred_classes, bb_outputs, anchors, grid_sizes, log=False):
    pred_bbs = torch_center_to_corners(map_bb_outputs_to_pred_bbs(bb_outputs, anchors, grid_sizes))
    
    if log: print("pred_bbs: ", pred_bbs)
        
    class_preds, clas_idxs = pred_classes.max(1)
    
    class_preds = class_preds.sigmoid()
    
    if log: print("class_preds: ", class_preds)
    
    conf_scores = pred_classes.sigmoid().t().data   
    
    if log: print("conf_scores: ", conf_scores)
    
    out1,out2,cc = [],[],[]
    for class_idx in range(1, len(conf_scores)):
        
        if log: print("class_idx: ", class_idx)
        
        c_mask = conf_scores[class_idx] > 0.2
        
        if log: print("c_mask: ", c_mask)
        
        if c_mask.sum() == 0: continue
            
        scores = conf_scores[class_idx][c_mask]
        
        if log: print("scores: ", scores)
            
        l_mask = c_mask.unsqueeze(1).expand_as(pred_bbs)
        
        if log: print("l_mask: ", l_mask)
        
        boxes = pred_bbs[l_mask].view(-1, 4)
        
        if log: print("boxes: ", boxes)
        
        ids, count = non_maximum_supression(boxes.data, scores, 0.4, 5)
        
        if log: print("ids: ", ids, " count: ", count)
        
        ids = ids[:count]
        
        if log: print("ids: ", ids)
             
        out1.append(scores[ids])
        
        if log: print("scores: ", scores[ids])
        
        out2.append(boxes.data[ids])
        
        if log: print("boxes: ", boxes[ids])
        
        cc.append([class_idx]*count)
        
        if log: print("classes: ", [class_idx]*count)
    
    if(len(cc) == 0): return torch.Tensor(), torch.Tensor(), torch.Tensor()
    
    cc = torch.from_numpy(np.concatenate(cc))
    out1 = torch.cat(out1)
    out2 = torch.cat(out2)
    
    return cc, out1, out2


class JaccardAccuracy(_AccuracyMeter):
    def __init__(self, anchors, grid_sizes, imsize):
        self.reset()
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.imsize = imsize
        
    def reset(self):
        self.num_true_positives = 0
        self.num_false_positives = 0
        self.num_false_negatives = 0
        
    def update(self, output, label, log=False):
        pred_classes, bb_outputs = output
        for i, x in enumerate(label['CAT']): 
            if log: print("############ Next Example ###############")
                    
            pred_classes_i, bb_outputs_i = pred_classes[i], bb_outputs[i]   
              
            label_bbs, label_classes = format_label(label['BB'][i], label['CAT'][i], self.imsize)    
            label_bbs, label_classes = label_bbs.data.cpu(), label_classes.data.cpu()
            nms_classes, nms_conf, nms_bbs = make_output(pred_classes_i, bb_outputs_i, self.anchors, self.grid_sizes)
            nms_classes, nms_conf, nms_bbs = nms_classes.cpu(), nms_conf.cpu(), torch_corners_to_center(nms_bbs.cpu())
            
            if log: print("Ground Truth Classes: ", label_classes); print("Ground Truth Bounding Boxes: ", label_bbs)
            
            if(len(nms_classes.size()) == 0): 
                if log: print("No Predictions")
                if log: print(f"Adding {label_classes.size()[0]} false negative(s)")
                self.num_false_negatives += label_classes.size()[0]
                continue   
            
            if log: print("Predicted Classes: ", nms_classes); print("Predicted Bounding Boxes: ", nms_bbs)
                
            pred_hits = torch.zeros(nms_classes.size()[0])
            label_hits = torch.zeros(len(label_classes))
                
            overlaps = jaccard(label_bbs, nms_bbs)
            
            for idx, cls, overlap in zip(range(len(label_classes)), label_classes, overlaps):              
                if log: print("------------ Next Label In Example -------------")
                if log: print("Ground Truth Class: ", cls); print("Overlaps For Class: ", overlap)
                
                matches = (overlap >= .5) * (nms_classes == cls)       
                
                if(matches.sum() > 0):
                    label_hits[idx] = 1
                                         
                if log: print("Predicted Bounding Boxes with the Correct Class Label and an Overlap >= .5: ", matches)
                    
                pred_hits[matches] = 1
            
            if log: print("------------ Results -------------")
                
            if log: print("Ground Truth Hits: ", label_hits)           
            if log: print("Prediction Hits: ", pred_hits) 
                
            self.num_true_positives += label_hits.sum()
            
            # self.num_correct_positives += pred_hits.sum()
            
            self.num_false_negatives += (label_hits != 1).sum()
            self.num_false_positives += (pred_hits != 1).sum()
            
            if log: print(f"{label_hits.sum()} true positives. {(pred_hits != 1).sum()} false positives. {(label_hits != 1).sum()} false negatives.")            
                                     
        
    def accuracy(self):
        precision = self.num_true_positives / (self.num_true_positives + self.num_false_positives)
        recall = self.num_true_positives / (self.num_true_positives + self.num_false_negatives) 
        return 2 * (precision * recall / (precision + recall))