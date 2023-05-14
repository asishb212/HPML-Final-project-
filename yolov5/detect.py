import argparse
from pathlib import Path
import torch


from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (LOGGER, Profile, check_img_size, cv2,non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def run(
        weights,  # model path
        source, 
        data, 
        imgsz,  # inference size (height, width)
        conf_thres,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        ):
    source = str(source)

    # Directories
    save_dir = '/content/run'  # increment run

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, _, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
          seen += 1
          p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

          p = Path(p)  # to Path
          save_path = str(save_dir)+'/'+p.name
          s += '%gx%g ' % im.shape[2:]  # print string
          annotator = Annotator(im0, line_width=3, example=str(names))
          if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, 5].unique():
              n = (det[:, 5] == c).sum()  # detections per class
              s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            #Write results
            for *xyxy, conf, cls in reversed(det):
              c = int(cls)  # integer class
              label = f'{names[c]} {conf:.2f}'
              annotator.box_label(xyxy, label, color=colors(c, True))
                    
          # Stream results
          im0 = annotator.result()
          cv2.imwrite(save_path, im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
  


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str)
    parser.add_argument('--source', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--device', default='')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

