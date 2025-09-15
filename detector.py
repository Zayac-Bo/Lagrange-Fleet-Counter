import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

class Detector:
    def __init__(self, weights_path='weights/best.pt', template_path='assets/template_fleet.png'):
        self.weights_path = weights_path
        self.template_path = template_path
        self.model = None
        if ULTRALYTICS_AVAILABLE and os.path.exists(weights_path):
            try:
                self.model = YOLO(weights_path)
                print('Loaded YOLO model from', weights_path)
            except Exception as e:
                print('Failed to load YOLO model:', e)
                self.model = None
        else:
            print('Ultralytics not available or weights not found. Using template fallback.')
        # ensure template exists
        if not os.path.exists(self.template_path):
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            self._create_default_template(self.template_path)

    def _create_default_template(self, path):
        # create a simple 'wide A' like white icon on black background
        w,h = 80,60
        img = np.zeros((h,w,3), dtype=np.uint8)
        # left slanted polygon
        pts_left = np.array([[8,55],[28,10],[36,10],[16,55]], dtype=np.int32)
        pts_right = np.array([[w-8,55],[w-28,10],[w-36,10],[w-16,55]], dtype=np.int32)
        cv2.fillPoly(img, [pts_left], (255,255,255))
        cv2.fillPoly(img, [pts_right], (255,255,255))
        # save
        cv2.imwrite(path, img)
        print('Wrote default template to', path)

    def process_image(self, input_path, out_path_base):
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError('failed to read image '+input_path)
        detections = []
        if self.model:
            detections = self._detect_with_yolo(img)
        else:
            detections = self._detect_with_template(img)
        # classify colors
        for d in detections:
            x1,y1,x2,y2 = d['bbox']
            d['color'] = self._classify_color(img, (x1,y1,x2,y2))
        # aggregate counts
        counts = {'blue':0,'yellow':0,'white':0,'unknown':0}
        for d in detections:
            c = d.get('color','unknown')
            if c not in counts:
                counts['unknown'] += 1
            else:
                counts[c] += 1
        # annotate image
        annotated = img.copy()
        for i,d in enumerate(detections):
            x1,y1,x2,y2 = map(int, d['bbox'])
            color = d.get('color','unknown')
            if color == 'blue':
                col = (255,0,0)
            elif color == 'yellow':
                col = (0,255,255)
            elif color == 'white':
                col = (200,200,200)
            else:
                col = (0,255,0)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), col, 2)
            cv2.putText(annotated, f"{color}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        out_path = out_path_base.rsplit('.',1)[0] + '_annotated.png'
        cv2.imwrite(out_path, annotated)
        return detections, out_path, counts

    def _detect_with_yolo(self, img):
        # uses ultralytics YOLO object if available
        results = self.model.predict(img, imgsz=640, conf=0.25, device='cpu')
        dets = []
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for b in boxes:
                # b.xyxy is a tensor with shape (n,4)
                try:
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    score = float(b.conf[0])
                except Exception:
                    vals = b.xyxy.numpy()[0]
                    xyxy = vals.astype(int)
                    score = float(b.conf.numpy()[0])
                dets.append({'bbox':[int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])], 'score': score})
        return dets

    def _detect_with_template(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        tw, th = template.shape[::-1]
        detections = []
        # multi-scale and rotation template matching
        scales = np.linspace(0.6, 1.4, 9)
        angles = list(range(-40,41,10))
        for scale in scales:
            sw = max(4, int(tw*scale))
            sh = max(4, int(th*scale))
            ts = cv2.resize(template, (sw, sh))
            for angle in angles:
                M = cv2.getRotationMatrix2D((sw//2, sh//2), angle, 1.0)
                rotated = cv2.warpAffine(ts, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if rotated.shape[0] >= gray.shape[0] or rotated.shape[1] >= gray.shape[1]:
                    continue
                res = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.58)
                for pt in zip(*loc[::-1]):
                    x,y = pt
                    score = float(res[y, x])
                    detections.append({'bbox':[x, y, x+sw, y+sh], 'score': score})
        # NMS
        if not detections:
            return []
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        keep = self._nms(boxes, scores, iou_thresh=0.25)
        final = []
        for idx in keep:
            final.append({'bbox': boxes[idx].tolist(), 'score': float(scores[idx])})
        return final

    def _nms(self, boxes, scores, iou_thresh=0.5):
        if boxes.shape[0] == 0:
            return []
        x1 = boxes[:,0].astype(float)
        y1 = boxes[:,1].astype(float)
        x2 = boxes[:,2].astype(float)
        y2 = boxes[:,3].astype(float)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep.tolist()

    def _classify_color(self, img, bbox):
        x1,y1,x2,y2 = bbox
        h_img, w_img = img.shape[:2]
        x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(w_img-1, int(x2)); y2 = min(h_img-1, int(y2))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return 'unknown'
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_mean = float(np.mean(hsv[:,:,0]))
        s_mean = float(np.mean(hsv[:,:,1]))
        v_mean = float(np.mean(hsv[:,:,2]))
        # thresholds (tweakable)
        if v_mean > 200 and s_mean < 40:
            return 'white'
        if 90 <= h_mean <= 140:
            return 'blue'
        if 8 <= h_mean <= 50 and s_mean > 40:
            return 'yellow'
        return 'unknown'
