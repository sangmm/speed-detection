import torch

features = torch.load("features.pth")
qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
res = scores.topk(5, dim=1)[1][:,0]
top1correct = gl[res].eq(ql).sum().item()

print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


#提取对应bounding box中的特征, 得到一个固定维度的特征，作为该bounding box的代表，供计算相似度时使用。