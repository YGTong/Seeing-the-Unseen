# from torch.nn import functional as F
# #交叉熵损失
# def ce_loss(score, target):
#     target = target.long()
#     loss = F.cross_entropy(score, target)
#     return loss




from torch.nn import functional as F

def ce_loss(score, target):
    target = target.long()
    loss = F.cross_entropy(score, target)
    return loss




#——————————————————————————————————修改之后——————————————————————————————————————

# from torch.nn import functional as F
# import torch
# import torch.nn as nn


# class OHEM_CELoss(nn.Module):
#     def __init__(self, thresh=0.7, ignore_index=255):
#         super(OHEM_CELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
#         self.ignore_index = ignore_index
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

#     def forward(self, logits, labels):
#         n_min = labels[labels != self.ignore_index].numel() // 16
#         loss = self.criteria(logits, labels).view(-1)
#         loss_hard = loss[loss > self.thresh]
#         if loss_hard.numel() < n_min:
#             loss_hard, _ = loss.topk(n_min)
#         return torch.mean(loss_hard)


# def ce_loss(score, target):
#     target = target.long()
#     loss1 = F.cross_entropy(score, target)

#     # Create an instance of OHEM_CELoss
#     ohem_loss = OHEM_CELoss()
    
#     # Calculate OHEM loss
#     loss2 = ohem_loss(score, target)

#     # Combine the two losses
#     loss = loss1 + loss2

#     return loss