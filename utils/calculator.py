from collections import defaultdict
import numpy as np

from utils.utils import guarantee_numpy


# class Calculator:
#     def __init__(self):
#         self.best_acc=0.0
#         self.best_count=0

#     def calculate(self, metrics, loss, y_true, y_pred, numpy=True, split='val', **kwargs):
#         if numpy:
#             y_true = guarantee_numpy(y_true)
#             y_pred = guarantee_numpy(y_pred)

#         history = defaultdict(list)
#         for metric in metrics:
#             history[metric] = getattr(self, f"get_{metric}")(loss=loss, y_true=y_true, y_pred=y_pred, split=split,  **kwargs)
#         return history

#     def get_loss(self, loss, **kwargs):
#         return float(loss)

#     def get_acc(self, y_true, y_pred, argmax=True, acc_count=False, split='val', **kwargs):
#         if argmax:
#             y_pred = np.argmax(y_pred, axis=1)
#         if acc_count:
#             correct_num=sum(y_true == y_pred)
#             if split!='train' and correct_num>self.best_count:
#                 self.best_count=correct_num
#             return correct_num
#         else:
#             acc=sum(y_true == y_pred) / len(y_true)
#             if split!='train' and acc>self.best_acc:
#                 self.best_acc=acc
#             return acc
#     def get_best_acc(self,acc_count=False,split='val',**kwargs):
        
#         if acc_count:
#             return self.best_count
#         else:
#             return self.best_acc



class Calculator:
    def calculate(self, metrics, loss, y_true, y_pred, numpy=True, **kwargs):
        if numpy:
            y_true = guarantee_numpy(y_true)
            y_pred = guarantee_numpy(y_pred)

        history = defaultdict(list)
        for metric in metrics:
            history[metric] = getattr(self, f"get_{metric}")(loss=loss, y_true=y_true, y_pred=y_pred, **kwargs)
        return history

    def get_loss(self, loss, **kwargs):
        return float(loss)

    def get_acc(self, y_true, y_pred, argmax=True, acc_count=False, **kwargs):
        if argmax:
            y_pred = np.argmax(y_pred, axis=1)
        if acc_count:
            return sum(y_true == y_pred)
        else:
            return sum(y_true == y_pred) / len(y_true)
