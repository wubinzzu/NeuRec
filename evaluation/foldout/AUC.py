def getAUC(predictions,K):
    auc = []
    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()
    for _ in range(1, K + 1):
        # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
        auc.append(1 - (position / len(neg_predict))) 
        
    return auc