import ipdb
import os, sys
import torch
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
from Examples.common import image_net_config
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator
from Examples.torch.utils.image_net_trainer import ImageNetTrainer
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from torchvision.models import resnet50, mobilenet_v2
from aimet_torch.model_preparer import prepare_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim import load_encodings_to_sim
from aimet_torch.qc_quantize_op import StaticGridQuantWrapper
from functools import partial
from tqdm import tqdm
from collections import namedtuple 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


TEST_NUM = 100
DATASET_DIR   = '/data/dataset/ImageNet_small'
Calibrate_DIR = '/data/dataset/ImageNet_small'
QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

cached_input_output = {}
def hook(name, module, input, output):
    if module not in cached_input_output:
        cached_input_output[module] = []
    cached_input_output[module].append((input[0].detach().cpu(), output.detach().cpu(),name))


class Round(InplaceFunction):
    @staticmethod
    def forward(ctx, input, inplace):
        ctx.inplace = inplace                                                                          
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        output.round_()
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input,None



def Qparam_checker(m):
    QI = False
    QO = False
    QP = False
    if m.input_quantizers[0].encoding is not None :
        QI = True
    if m.output_quantizers[0].encoding is not None :
        QO = True
    if m.param_quantizers['weight'].encoding is not None:
        QP = True
    return QI, QO, QP





class wrapped_layer(torch.nn.Module):
    def __init__(self, layer, act_quant_first, act_range, act_zero_point, w_range, w_zero_point, b_range, b_zero_point, num_bits = 8, signed = False):
        super(wrapped_layer, self).__init__()
        self.layer = layer
        self.act_quant_first = act_quant_first
        self.act_range = act_range
        self.act_zero_point = act_zero_point
        self.w_range = w_range
        self.w_zero_point = w_zero_point
        self.b_range = b_range
        self.b_zero_point = b_zero_point
        self.num_bits = num_bits
        self.signed = signed
    def forward(self, x):
        
        if self.act_quant_first:
            x = self.qdq(x, self.act_zero_point, self.act_range, self.num_bits, self.signed)
        w_ = self.qdq(self.layer._module_to_wrap.weight, self.w_zero_point, self.w_range, self.num_bits, self.signed)
        b_ = self.qdq(self.layer._module_to_wrap.bias, self.b_zero_point, self.b_range, self.num_bits, self.signed) if self.b_range is not None else self.layer._module_to_wrap.bias
    
        # original_weight = self.layer._module_to_wrap.weight.data.clone()
        # original_bias = self.layer._module_to_wrap.bias.data.clone()
        # self.layer._module_to_wrap.weight.data = w_
        # self.layer._module_to_wrap.bias.data = b_
        # x = self.layer._module_to_wrap(x)
        # self.layer._module_to_wrap.weight.data = original_weight
        # self.layer._module_to_wrap.bias.data = original_bias

        param_dict= {}
        for p in self.layer._module_to_wrap.__dict__.keys():
            if p.startswith('_') : continue
            if p=='training': continue
            param_dict[p]=self.layer._module_to_wrap.__dict__[p]      

        if isinstance(self.layer._module_to_wrap,torch.nn.Conv2d):
            x = F.conv2d(input=x, weight=w_,bias=b_,stride=param_dict['stride'],
                           padding=param_dict['padding'],dilation=param_dict['dilation'],groups=param_dict['groups'])        
        elif isinstance(self.layer._module_to_wrap,torch.nn.Linear):
            x = F.linear(input=x, weight=w_,bias=b_)

        if not self.act_quant_first:
            x = self.qdq(x, self.act_zero_point, self.act_range, self.num_bits, self.signed)
        return x
    
    def qdq(self, input, zero_point, range, num_bits, signed):
        output = input.clone() 
        qmin = -(2.**(num_bits - 1)) if signed else 0.
        qmax = qmin + 2.**num_bits - 1.
        # ZP quantization for HW compliance
        running_range=range.clamp(min=1e-6,max=1e5)
        scale = running_range / (qmax - qmin)
        running_zero_point_round = Round().apply(qmin-zero_point/scale,False)
        zero_point = (qmin-running_zero_point_round.clamp(qmin,qmax))*scale
        output.add_(qmin * scale - zero_point).div_(scale)
        output = Round().apply(output.clamp_(qmin, qmax),False)
        # if dequantize:
        output.mul_(scale).add_(zero_point - qmin * scale)
        return output


def adaquant(layer, cached_inps, cached_Qouts, iters=100,  batch_size=50, signed = False, num_bits=8):
    
    # qdq params
    act_quant_first = False

    act_range = None
    act_zero_point = None

    w_range = None
    w_zero_point = None

    b_range = None
    b_zero_point = None
    opt_bias = None

    # lr params
    # lr_factor = 1e-2
    lr_qpin = 1e-1# lr_factor * (test_inp.max() - test_inp.min()).item()  # 1e-1
    lr_qpw = 1e-3#lr_factor * (layer._module_to_wrap.weight.max() - layer._module_to_wrap.weight.min()).item()  # 1e-3
    lr_w = 1e-5#lr_factor * layer.weight.std().item()  # 1e-5
    lr_b = 1e-3#lr_factor * layer.bias.std().item()  # 1e-3

    qmin = -(2.**(num_bits - 1)) if signed else 0.
    qmax = qmin + 2.**num_bits - 1.
    # ipdb.set_trace()
    if layer.input_quantizers[0].encoding is not None:
        act_quant_first = True
        size = (1,)*cached_inps.dim()
        act_range      = torch.full(size=size, fill_value=layer.input_quantizers[0].encoding.max-layer.input_quantizers[0].encoding.min ,dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        act_zero_point = torch.full(size=size, fill_value=(layer.input_quantizers[0].encoding.offset+qmin)*layer.input_quantizers[0].encoding.delta ,dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        # act_range      = torch.tensor([[[[layer.input_quantizers[0].encoding.max-layer.input_quantizers[0].encoding.min]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        # act_zero_point = torch.tensor([[[[(layer.input_quantizers[0].encoding.offset+qmin)*layer.input_quantizers[0].encoding.delta]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)   
    else:
        act_quant_first = False
        size = (1,)*cached_inps.dim()
        act_range      = torch.full(size=size,fill_value = layer.output_quantizers[0].encoding.max-layer.output_quantizers[0].encoding.min ,dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        act_zero_point = torch.full(size=size,fill_value = (layer.output_quantizers[0].encoding.offset+qmin)*layer.output_quantizers[0].encoding.delta  ,dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        # act_range      = torch.tensor([[[[layer.output_quantizers[0].encoding.max-layer.output_quantizers[0].encoding.min]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        # act_zero_point = torch.tensor([[[[(layer.output_quantizers[0].encoding.offset+qmin)*layer.output_quantizers[0].encoding.delta]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)

    act_range_ = act_range.clone()
    act_zero_point_ = act_zero_point.clone()

    size = (1,)* layer._module_to_wrap.weight.dim()
    w_range      = torch.full(size=size,fill_value=layer.param_quantizers['weight'].encoding.max-layer.param_quantizers['weight'].encoding.min, dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
    w_zero_point = torch.full(size=size,fill_value=(layer.param_quantizers['weight'].encoding.offset+qmin)*layer.param_quantizers['weight'].encoding.delta, dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
    # w_range      = torch.tensor([[[[layer.param_quantizers['weight'].encoding.max-layer.param_quantizers['weight'].encoding.min]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
    # w_zero_point = torch.tensor([[[[(layer.param_quantizers['weight'].encoding.offset+qmin)*layer.param_quantizers['weight'].encoding.delta]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)

    w_range_ = w_range.clone()
    w_zero_point_ = w_zero_point.clone()


    if layer.param_quantizers['bias'].encoding is not None:
        size = (1,)* layer._module_to_wrap.bias.dim()
        b_range = torch.full(size=size,fill_value=layer.param_quantizers['bias'].encoding.max-layer.param_quantizers['bias'].encoding.min, dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        b_zero_point = torch.tensor(size=size, fill_value=(layer.param_quantizers['bias'].encoding.offset+qmin)*layer.param_quantizers['bias'].encoding.delta, dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        
        # b_range = torch.tensor([[[[layer.param_quantizers['bias'].encoding.max-layer.param_quantizers['bias'].encoding.min]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        # b_zero_point = torch.tensor([[[[(layer.param_quantizers['bias'].encoding.offset+qmin)*layer.param_quantizers['bias'].encoding.delta]]]], dtype=torch.float32, requires_grad=True, device=layer._module_to_wrap.weight.device)
        
        b_range_ = b_range.clone()
        b_zero_point_ = b_zero_point.clone()
  
    layer.eval()
    cached_inps = cached_inps.to(layer._module_to_wrap.weight.device)
    cached_Qouts= cached_Qouts.to(layer._module_to_wrap.weight.device)
    with torch.no_grad(): 
        cached_outs = layer._module_to_wrap(cached_inps)
    mse_before = F.mse_loss(cached_outs, cached_Qouts)

    opt_w = torch.optim.Adam([layer._module_to_wrap.weight], lr=lr_w)
    if b_range is not None: opt_bias = torch.optim.Adam([layer._module_to_wrap.bias], lr=lr_b)

    opt_qparams_in = torch.optim.Adam([act_range, act_zero_point], lr=lr_qpin)
    opt_qparams_w = torch.optim.Adam([w_range, w_zero_point], lr=lr_qpw)


    layer_ = wrapped_layer(layer, act_quant_first, act_range, act_zero_point, w_range, w_zero_point, b_range, b_zero_point, num_bits, signed)

    weight_ = layer._module_to_wrap.weight.data.clone()
    # for j in (tqdm(range(iters))):
    for j in range(iters):
        idx = torch.randperm(cached_inps.size(0))[:batch_size]

        train_inp = cached_inps[idx]#.cuda()
        train_out = cached_outs[idx]#.cuda() # normal result
        
        qout = layer_(train_inp) # quantized result
        loss = F.mse_loss(qout, train_out)

        opt_w.zero_grad()
        if b_range is not None:  opt_bias.zero_grad()
        opt_qparams_in.zero_grad()
        opt_qparams_w.zero_grad()
        loss.backward()
        opt_w.step()
        if b_range is not None: opt_bias.step()
        opt_qparams_in.step()
        opt_qparams_w.step()  

    act_zero_point = act_zero_point.detach().cpu().numpy().item()
    act_range      = act_range.detach().cpu().numpy().item()

    act_scale      = act_range/(qmax-qmin)
    act_offset     = round(act_zero_point/act_scale - qmin)
    act_max        = (qmax+act_offset)*act_scale
    act_min        = (qmin+act_offset)*act_scale

    w_zero_point = w_zero_point.detach().cpu().numpy().item()
    w_range      = w_range.detach().cpu().numpy().item()

    w_scale      = w_range/(qmax-qmin)
    w_offset     = round(w_zero_point/w_scale - qmin)
    w_max        = (qmax+w_offset)*w_scale
    w_min        = (qmin+w_offset)*w_scale

    if layer.input_quantizers[0].encoding is not None:
        layer.input_quantizers[0].encoding.delta  = act_scale
        layer.input_quantizers[0].encoding.offset = act_offset
        layer.input_quantizers[0].encoding.max    = act_max
        layer.input_quantizers[0].encoding.min    = act_min
    else:
        layer.output_quantizers[0].encoding.delta  = act_scale
        layer.output_quantizers[0].encoding.offset = act_offset
        layer.output_quantizers[0].encoding.max    = act_max
        layer.output_quantizers[0].encoding.min    = act_min

    layer.param_quantizers['weight'].encoding.delta  = w_scale
    layer.param_quantizers['weight'].encoding.offset = w_offset
    layer.param_quantizers['weight'].encoding.max    = w_max
    layer.param_quantizers['weight'].encoding.min    = w_min

    layer.eval()
    with torch.no_grad():mse_after = F.mse_loss(cached_outs, layer(cached_inps))

    return mse_before, mse_after

    



class ImageNetDataPipeline:

    @staticmethod
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for ImageNet dataset and returns it
        """
        data_loader = ImageNetDataLoader(Calibrate_DIR,
                                         image_size=image_net_config.dataset['image_size'],
                                        #  batch_size=image_net_config.evaluation['batch_size'],
                                        batch_size=512,
                                         is_training=False,
                                         num_workers=image_net_config.evaluation['num_workers']).data_loader
        return data_loader

    @staticmethod
    def evaluate(model: torch.nn.Module, use_cuda: bool) -> float:
        """
        Given a torch model, evaluates its Top-1 accuracy on the dataset
        :param model: the model to evaluate
        :param use_cuda: whether or not the GPU should be used.
        """
        evaluator = ImageNetEvaluator(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                        #  batch_size=image_net_config.evaluation['batch_size'],
                                        batch_size=512,
                                      num_workers=image_net_config.evaluation['num_workers'])

        return evaluator.evaluate(model, iterations=None, use_cuda=use_cuda)

    @staticmethod
    def finetune(model: torch.nn.Module, epochs, learning_rate, learning_rate_schedule, use_cuda):
        """
        Given a torch model, finetunes the model to improve its accuracy
        :param model: the model to finetune
        :param epochs: The number of epochs used during the finetuning step.
        :param learning_rate: The learning rate used during the finetuning step.
        :param learning_rate_schedule: The learning rate schedule used during the finetuning step.
        :param use_cuda: whether or not the GPU should be used.
        """
        trainer = ImageNetTrainer(DATASET_DIR, image_size=image_net_config.dataset['image_size'],
                                  batch_size=image_net_config.train['batch_size'],
                                  num_workers=image_net_config.train['num_workers'])

        trainer.train(model, max_epochs=epochs, learning_rate=learning_rate,
                      learning_rate_schedule=learning_rate_schedule, use_cuda=use_cuda)
def pass_calibration_data(sim_model, use_cuda):
    data_loader = ImageNetDataPipeline.get_val_dataloader()
    batch_size = data_loader.batch_size

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    sim_model.eval()
    samples = 1000

    batch_cntr = 0
    with torch.no_grad():
        for input_data, target_data in data_loader:

            inputs_batch = input_data.to(device)
            sim_model(inputs_batch)

            batch_cntr += 1
            if (batch_cntr * batch_size) > samples:
                break

model = mobilenet_v2(pretrained=True)
model = prepare_model(model)
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    model.to(torch.device('cuda'))


accuracy = ImageNetDataPipeline.evaluate(model, use_cuda)

_ = fold_all_batch_norms(model, input_shapes=(1, 3, 224, 224))
dummy_input = torch.rand(1, 3, 224, 224)    # Shape for each ImageNet sample is (3 channels) x (224 height) x (224 width)
if use_cuda: dummy_input = dummy_input.cuda()
sim = QuantizationSimModel(model=model,
                           quant_scheme=QuantScheme.post_training_tf_enhanced,
                           dummy_input=dummy_input,
                           default_output_bw=8,
                           default_param_bw=8)

load_encodings_to_sim(sim,'output/model_after_qat_torch.encodings')    

# dummy_input = dummy_input.cpu()
# sim.compute_encodings(forward_pass_callback=pass_calibration_data, forward_pass_callback_args=use_cuda)
# accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)
# os.makedirs('./output/', exist_ok=True)

# sim.export(path='./output/', filename_prefix='model_after_qat', dummy_input=dummy_input)


# ##########################################################
handlers = []
for name,m in sim.model.named_modules():
    if name =='': continue       
    if not isinstance(m,StaticGridQuantWrapper): continue
    if not (isinstance(m._module_to_wrap, torch.nn.Linear) or isinstance(m._module_to_wrap, torch.nn.Conv2d)): continue    
    QI,QO,QP = Qparam_checker(m)
    if QI|QO == False : continue  
    handlers.append(m.register_forward_hook(partial(hook,name)))
accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)

for handler in handlers: handler.remove()


for i, layer in enumerate(cached_input_output):
    name = cached_input_output[layer][0][2]
    print(name)
    cached_inps  = torch.cat([x[0] for x in cached_input_output[layer]])
    cached_outs = torch.cat([x[1] for x in cached_input_output[layer]])
    mse_before, mse_after = adaquant(layer, cached_inps, cached_outs, iters=100)
    print(f'mse_before:{mse_before}, mse_after:{mse_after}')

accuracy = ImageNetDataPipeline.evaluate(sim.model, use_cuda)