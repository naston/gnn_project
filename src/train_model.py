from .eval import compute_loss, compute_auc
import torch

def train_link_pred(epochs, model, pred, optimizer, train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g):
    # ----------- 4. training -------------------------------- #
    for e in range(epochs):
        # forward
        h = model(train_g, train_g.ndata["x"])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e+1) % 5 == 0:
            print("In epoch {}, loss: {}".format(e+1, loss))

        # ----------- 5. check results ------------------------ #
        if (e+1) % 100 == 0:
            with torch.no_grad():
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                print("AUC", compute_auc(pos_score, neg_score))