import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0]
        assert y_pred.dim() == 2
        predicted = torch.argmax(y_pred, dim=1)
        correct = (predicted == y_true).float()
        accuracy = correct.sum() / len(correct)
        return accuracy


class F1Score(nn.Module):
    def __init__(self, num_classes, average="macro"):
        super(F1Score, self).__init__()
        self.num_classes = num_classes
        self.average = average

    def forward(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0]
        assert y_pred.dim() == 2
        # 確率をクラスラベルに変換
        y_pred = torch.argmax(y_pred, dim=1)

        # 混同行列の作成
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        # Precision と Recall の計算
        precision = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-12)
        recall = confusion_matrix.diag() / (confusion_matrix.sum(0) + 1e-12)

        # F1スコアの計算
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        if self.average == "macro":
            return torch.mean(f1)
        elif self.average == "micro":
            total_true = torch.sum(confusion_matrix.sum(1))
            total_pred = torch.sum(confusion_matrix.diag())
            return total_pred / total_true
        else:
            raise ValueError("Average type not recognized. Please choose 'macro' or 'micro'.")


# 使用例
# f1_score = F1Score(num_classes=10, average='macro')
