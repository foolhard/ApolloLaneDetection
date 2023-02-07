"""torch2onnx.py
    @Create By dxg 2022.11.18
"""
import cv2
import torch
import scipy.special
import numpy as np
from torchvision import transforms
from model.model import parsingNet
from data.constant import tusimple_row_anchor
import torch.onnx
import onnxruntime

# Tusimple的超参数
backbone = '18'
griding_num = 100
cls_num_per_lane = 56
row_anchor = tusimple_row_anchor

def img_init(img):
    """图片初始化，和归一化

    Args:
        img (cv2:ndarry): cv格式的图片

    Returns:
        blob (tensor): BCHW格式的torch.tensor
    """
    # 图像格式统一：(288, 800)，图像张量，归一化
    img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    img_o = img_transforms(img)
    img_o.unsqueeze_(0)
    return img_o

def torch_model_init(model_p):
    """模型初始化

    Args:
        model_p (str): 模型路径

    Returns:
        model_torch: 返回根据权重读取的torch模型
    """
    torch_net = parsingNet(pretrained=False, backbone=backbone, cls_dim=(griding_num+1,cls_num_per_lane,4),use_aux=False).cuda()
    state_dict = torch.load(model_p, map_location='cuda')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    torch_net.load_state_dict(compatible_state_dict, strict=False)
    torch_net.eval()
    return torch_net

def post_process(out, vis, save_fn):
    img_h, img_w = vis.shape[:2]
    print(img_w, img_h)
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    out_j = out[0].data.cpu().numpy()  # 数据类型转换成numpy [101,56,4]
    out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,4]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # [100,56,4]softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
    idx = np.arange(griding_num) + 1  # 产生 1-100
    idx = idx.reshape(-1, 1, 1)  # [100,1,1]
    loc = np.sum(prob * idx, axis=0)  # [56,4]
    out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
    loc[out_j == griding_num] = 0  # 若最大值的索引=griding_num，归零
    out_j = loc  # [56,4]
    # import pdb; pdb.set_trace()
    # vis = cv2.imread(os.path.join(data_root,names[0]))  # 读取图像 [720,1280,3]

    for i in range(out_j.shape[1]):  # 遍历列
        if np.sum(out_j[:, i] != 0) > 2:  # 非0单元格的数量大于2
            for k in range(out_j.shape[0]):  # 遍历行
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    cv2.circle(vis,ppp,5,(0,255,0),-1)
    # 保存检测结果图
    cv2.imwrite(save_fn,vis)

def post_process_onnx(out, vis, save_fn):
    img_h, img_w = vis.shape[:2]
    print(img_w, img_h)
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    
    out_j = out[0]  # 数据类型转换成numpy [101,56,4]
    out_j = out_j[:, ::-1, :]  # 将第二维度倒着取[101,56,4]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # [100,56,4]softmax 计算（概率映射到0-1之间且沿着维度0概率总和=1）
    idx = np.arange(griding_num) + 1  # 产生 1-100
    idx = idx.reshape(-1, 1, 1)  # [100,1,1]
    loc = np.sum(prob * idx, axis=0)  # [56,4]
    out_j = np.argmax(out_j, axis=0)  # 返回最大值的索引
    loc[out_j == griding_num] = 0  # 若最大值的索引=griding_num，归零
    out_j = loc  # [56,4]
    # import pdb; pdb.set_trace()
    # vis = cv2.imread(os.path.join(data_root,names[0]))  # 读取图像 [720,1280,3]

    for i in range(out_j.shape[1]):  # 遍历列
        if np.sum(out_j[:, i] != 0) > 2:  # 非0单元格的数量大于2
            for k in range(out_j.shape[0]):  # 遍历行
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    cv2.circle(vis,ppp,5,(0,255,0),-1)
    # 保存检测结果图
    cv2.imwrite(save_fn,vis)
    
def torch_infer(img_p, model_p):
    img = cv2.imread(img_p)
    blob = img_init(img)
    blob = blob.cuda()
    model = torch_model_init(model_p)
    with torch.no_grad():
        output = model(blob)
    post_process(out=output, vis=img, save_fn='out.jpg')

def torch2onnx(img_p, model_p):
    img = cv2.imread(img_p)
    blob = img_init(img)
    blob = blob.cuda()
    model = torch_model_init(model_p)
    with torch.no_grad():
        output = model(blob)
        torch.onnx.export(
            model=model,
            args=blob,
            f='ufld.onnx',
            opset_version=11,
            input_names=['input'],
            output_names=['output'])
    post_process(out=output, vis=img, save_fn='out.jpg')
    
def onnx_eval(onnx_p):
    import onnx
    onnx_model = onnx.load(onnx_p) 
    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect") 
    else: 
        print("Model correct")
def onnx_infer(img_p, onnx_p):
    img = cv2.imread(img_p)
    blob = img_init(img)
    ort_session = onnxruntime.InferenceSession(onnx_p)
    ort_inputs = {'input': blob.numpy()}
    ort_output = ort_session.run(['output'], ort_inputs)[0] 
    post_process_onnx(out=ort_output, vis=img, save_fn='out_onnx.jpg')
    
if __name__ == "__main__":
    img_p = 'test_input/tusimple/0.jpg'
    model_p = 'weights/tusimple_18.pth'
    onnx_p = 'ufld.onnx'
    # torch_infer(img_p, model_p) # torch_model 的推理
    # torch2onnx(img_p, model_p) # 将torch_model 生成onnx_model
    # onnx_eval(onnx_p) # 检查onnx_model是否正确
    onnx_infer(img_p, onnx_p) # onnx_model 推理
# end main
