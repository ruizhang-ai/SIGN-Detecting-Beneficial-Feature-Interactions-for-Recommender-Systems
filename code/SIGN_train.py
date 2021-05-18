import numpy as np
import torch
from SIGN_model import L0_SIGN
from sklearn.metrics import roc_auc_score, accuracy_score


def train(args, data_info, show_loss, data_nums):
    train_loader = data_info[0]
    val_loader = data_info[1]
    test_loader = data_info[2]
    num_feature = data_info[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = L0_SIGN(args, num_feature).to(device)
    model = L0_SIGN(args, num_feature, device)
    model = model.to(device)
    #for name, para in model.named_parameters():
    #    print(name, para.size())
    optimizer = torch.optim.Adagrad(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        lr_decay=1e-5,
        #weight_decay=1e-5
    )
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    #crit = torch.nn.BCELoss()
    crit = torch.nn.MSELoss()

    print([i.size() for i in filter(lambda p: p.requires_grad, model.parameters())])
    print('start training...')
    for step in range(args.n_epoch):
        # training
        loss_all = 0
        edge_all = 0
        model.train()
        for data in train_loader:
            data = data.to(device)
            #return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            output, l0_penaty, l2_penaty, num_edges = model(data)
            label = data.y.view(-1,2)
            label = label.to(device)
            baseloss = crit(output, label)
            l0_loss = l0_penaty * args.l0_weight 
            l2_loss = l2_penaty * args.l2_weight
            loss = baseloss + l0_loss + l2_loss 
            loss_all += data.num_graphs * loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        cur_loss = loss_all / data_nums[0]

        # evaluation
        #train_acc, _ = evaluate(model, train_loader, device)
        train_auc = 0
        val_auc,val_acc, _ = evaluate(model, val_loader, device)    
        test_auc, test_acc, test_edges = evaluate(model, test_loader, device)

        print('Epoch: {:03d}, Loss: {:.4f}, Train Auc: {:.4f}, Val Auc: {:.4f}, Acc: {:.4f}, Test Auc: {:.4f}, Acc: {:.4f}, Train edges: {:07d}'.
          format(step, cur_loss, train_auc, val_auc, val_acc, test_auc, test_acc, test_edges))
        #print('epoch %d    train auc: %.4f  acc: %.4f  rmse:%.4f   eval auc: %.4f  acc: %.4f  rmse: %.4f    test auc: %.4f  acc: %.4f  rmse %.4f' 
        #        % (step, train_auc, train_acc, train_rmse, eval_auc, eval_acc, eval_rmse, test_auc, test_acc, test_rmse))


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    labels = []
    edges_all = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, _, _, num_edges = model(data)
            pred = pred.detach().cpu().numpy()
            edges_all += num_edges
            label = data.y.view(-1, 2).detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(np.argmax(labels, 1), np.argmax(predictions, 1))
    return auc, acc, edges_all
