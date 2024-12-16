import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from torch import Tensor
import math

class MIAdam3(Optimizer):
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MIAdam3, self).__init__(params, defaults)
        
        self.buf_exp_avgs = []
        self.buf_exp_avgs2 = []
        self.buf_exp_avgs3 = []

    def __setstate__(self, state):
        super(MIAdam3, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

            
    @staticmethod
    def miadam3(params: List[Tensor],
             grads: List[Tensor],
             exp_avgs: List[Tensor],
             buf_exp_avgs: List[Tensor],
             buf_exp_avgs2: List[Tensor],
             buf_exp_avgs3: List[Tensor],
             exp_avg_sqs: List[Tensor],
             max_exp_avg_sqs: List[Tensor],
             epoch:int,
             kappa:float,
             sub_epoch:int, 
             state_steps: List[int],
             amsgrad: bool,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float,
             ):
        
        for i, param in enumerate(params):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            buf_exp_avg = buf_exp_avgs[i]
            buf_exp_avg2 = buf_exp_avgs2[i]
            buf_exp_avg3 = buf_exp_avgs3[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)


            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            kappa1=kappa
            if epoch<sub_epoch:
                step_size = (lr**4) / bias_correction1
                buf_exp_avg.mul_(kappa1).add_(exp_avg)
                buf_exp_avg2.mul_(kappa1).add_(buf_exp_avg)
                buf_exp_avg3.mul_(kappa1).add_(buf_exp_avg2)
                param.addcdiv_(exp_avg, denom, value=-step_size)

            else :
                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

                

    @torch.no_grad()
    def step(self, closure=None,epoch=0,kappa=0,sub_epoch=0):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['buf_exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # 
                        state['buf_exp_avg2'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # 
                        state['buf_exp_avg3'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # 
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    self.buf_exp_avgs.append(state['buf_exp_avg'])
                    self.buf_exp_avgs2.append(state['buf_exp_avg2'])
                    self.buf_exp_avgs3.append(state['buf_exp_avg3'])

                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            
            MIAdam3.miadam3(params_with_grad,
                               grads,
                               exp_avgs,
                               self.buf_exp_avgs,
                               self.buf_exp_avgs2,
                               self.buf_exp_avgs3,
                               exp_avg_sqs,
                               max_exp_avg_sqs,
                               epoch,
                               kappa,
                               sub_epoch,
                               state_steps,
                               group['amsgrad'],
                               beta1,
                               beta2,
                               group['lr'],
                               group['weight_decay'],
                               group['eps'])
                                
        return loss
