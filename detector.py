from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class FaceDetector(object):
    '''
    Class for face detection
    '''
    def __init__(self, trained_model, cpu=True, nms_threshold=0.5, top_k=1000, confidence_threshold=0.8, keep_top_k=10):
        super(FaceDetector, self).__init__()
        self.trained_model = trained_model
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)
        self.net = load_model(self.net, trained_model, cpu)
        self.net.eval()
        print('Finished loading model', trained_model)

        self.device = torch.device("cpu" if cpu else "cuda")
        self.net = self.net.to(self.device)

        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.keep_top_k = keep_top_k
        self.cpu = cpu

    def predict(self, img_name):
        img = np.float32(cv2.imread(img_name, cv2.IMREAD_COLOR))
        resize = 1
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        _t = {'forward_pass': Timer(), 'misc': Timer()}
        _t['forward_pass'].tic()
        loc, conf = self.net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, self.nms_threshold)
        keep = nms(dets, self.nms_threshold,force_cpu=self.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        _t['misc'].toc()

        return dets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FaceBoxes')
    
    parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.3, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--images', type=str, nargs='+', help='images to test and show', required=True)
    args = parser.parse_args()


    torch.set_grad_enabled(False)

    f_detector = FaceDetector(args.trained_model, args.cpu, args.nms_threshold, args.top_k, args.confidence_threshold, args.keep_top_k)
    for i, img_name in enumerate(args.images):
        dets = f_detector.predict(img_name)
        print(dets)

        show_img = cv2.imread(img_name)
        print(dets)
        for k in range(dets.shape[0]):
            bbx = dets[k].astype(np.int32).tolist()
            cv2.rectangle(show_img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), 255, 1)
        cv2.imshow('rest', show_img)
        cv2.waitKey()
