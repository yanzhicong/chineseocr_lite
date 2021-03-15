from config import *
from crnn import CRNNHandle
from angnet import  AngleNetHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import copy
from dbnet.dbnet_infer import DBNET
import time
import traceback



class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)
        if angle_detect:
            self.angle_handle = AngleNetHandle(angle_net_path)


    def crnnRecWithBox(self, im, boxes_list, score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))

            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_detect and angle_res:
                partImg = partImg.rotate(180)

            if not is_rgb:
                partImg = partImg.convert('L')

            try:
                if is_rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
                count += 1

        return results




    def text_predict(self, img, short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.crnnRecWithBox(np.array(img), boxes_list,score_list)
        return result







def tr_run_img(img, ocrhandle):

    img = img.convert("RGB")
    short_size = 960
    
    res = []
    do_det = True

    img_w, img_h = img.size
    if max(img_w, img_h) * (short_size * 1.0 / min(img_w, img_h)) > dbnet_max_size:
        # logger.error(exc_info=True)
        # res.append("图片reize后长边过长，请调整短边尺寸")
        do_det = False
        # self.finish(json.dumps({'code': 400, 'msg': '图片reize后长边过长，请调整短边尺寸'}, cls=NpEncoder))
        # return


        return False, []


    if do_det:
        res = ocrhandle.text_predict(img,short_size)
        

    return True, []

        # img_detected = img.copy()

        # img_draw = ImageDraw.Draw(img_detected)
        # colors = ['red', 'green', 'blue', "purple"]
        # for i, r in enumerate(res):
        #     rect, txt, confidence = r

        #     x1,y1,x2,y2,x3,y3,x4,y4 = rect.reshape(-1)
        #     size = max(min(x2-x1,y3-y2) // 2 , 20 )

        #     myfont = ImageFont.truetype("仿宋_GB2312.ttf", size=size)
        #     fillcolor = colors[i % len(colors)]
        #     img_draw.text((x1, y1 - size ), str(i+1), font=myfont, fill=fillcolor)
        #     for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:
        #         img_draw.line(xy=xy, fill=colors[i % len(colors)], width=2)

        # output_buffer = BytesIO()
        # img_detected.save(output_buffer, format='JPEG')
        # byte_data = output_buffer.getvalue()
        # img_detected_b64 = base64.b64encode(byte_data).decode('utf8')
    
    # else:
    #     output_buffer = BytesIO()
    #     img.save(output_buffer, format='JPEG') 
    #     byte_data = output_buffer.getvalue()
    #     img_detected_b64 = base64.b64encode(byte_data).decode('utf8')









if __name__ == "__main__":

    ocrhandle = OcrHandle()
    # img = 



    img_list = os.listdir("./test_imgs")

    for img_name in os.listdir('./test_imgs'):
        img_buffer, img = 





    pass


