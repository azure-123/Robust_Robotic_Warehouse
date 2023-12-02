import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from marl_utils import onehot_from_logits

def tar_attack(model, epsilon, states, action, tar_action, opt):
    loss_func = nn.CrossEntropyLoss()
    adv_states = [Variable(adv_state, requires_grad=True) for adv_state in states]
    for adv_state in adv_states:
        adv_state.retain_grad()
    onehot_from_logits(model.agents[0].model(adv_states[0]))
    logits = [torch.softmax(model.agents[i].model(adv_states[i])[0], dim=0) for i in range(len(adv_states))]
    # for logit in logits:
    #     logit.retain_grad()
    losses = [-loss_func(logits[i], torch.tensor(tar_action[i])) + loss_func(logits[i], torch.tensor(action[i])) for i in range(len(adv_states))]
    for adv_state in adv_states:
        adv_state.retain_grad()
    opt.zero_grad()
    for loss in losses:
        loss.backward(retain_graph=True)
    eta = [epsilon * adv_states[i].grad.data.sign() for i in range(len(adv_states))]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()

    return adv_states

# def tar_attack_eval(model, epsilon, states, action, tar_action, opt, use_cuda):
    # loss_func = nn.CrossEntropyLoss()
    # adv_states = torch.tensor(np.array(states), requires_grad=True, dtype=torch.float32)
    # if use_cuda:
    #     adv_states = adv_states.cuda()
    # logits = model(adv_states)

    # if use_cuda:
    #     loss = -loss_func(logits, torch.tensor(np.array(tar_action)).long().cuda()) + loss_func(logits, torch.tensor(np.array(action)).long().cuda())
    # else:
    #     loss = -loss_func(logits, torch.tensor(np.array(tar_action)).long()) + loss_func(logits, torch.tensor(np.array(action)).long())
    
    # adv_states.retain_grad()
    # opt.zero_grad()
    # loss.backward()
    

    # eta_0 = epsilon * adv_states.grad.data.sign()
    # adv_states.data = Variable(adv_states.data + eta_0, requires_grad=True)

    # if use_cuda:
    #     eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).cuda().data, -epsilon, epsilon)
    #     adv_states.data = torch.tensor(np.array(states)).cuda().data + eta_0
    # else:
    #     eta_0 = torch.clamp(adv_states.data - torch.tensor(np.array(states)).data, -epsilon, epsilon)
    #     adv_states.data = torch.tensor(np.array(states)).data + eta_0

    # return adv_states.cpu().data.numpy()

def fgsm(model, epsilon, states, action, opt):
    loss_func = nn.CrossEntropyLoss()
    adv_states = [Variable(adv_state, requires_grad=True) for adv_state in states]
    for adv_state in adv_states:
        adv_state.retain_grad()
    onehot_from_logits(model.agents[0].model(adv_states[0]))
    logits = [torch.softmax(model.agents[i].model(adv_states[i])[0], dim=0) for i in range(len(adv_states))]
    # for logit in logits:
    #     logit.retain_grad()
    losses = [loss_func(logits[i], torch.tensor(action[i])) for i in range(len(adv_states))]
    for adv_state in adv_states:
        adv_state.retain_grad()
    opt.zero_grad()
    for loss in losses:
        loss.backward(retain_graph=True)
    eta = [epsilon * adv_states[i].grad.data.sign() for i in range(len(adv_states))]
    for i in range(len(adv_states)):
        adv_states[i].data = Variable(adv_states[i].data + eta[i], requires_grad=True)
        eta[i] = torch.clamp(adv_states[i].data - states[i].data, -epsilon, epsilon)
        adv_states[i].data = states[i].data + eta[i]
        adv_states[i] = adv_states[i].detach()

    return adv_states