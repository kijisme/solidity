
def train(args, model, train_loader, optimizer, loss_fcn, epoch):
    model.train()
    total_accucracy =  0
    total_macro_f1 = 0
    total_micro_f1 = 0
    total_loss = 0
    circle_lrs = []
    for idx, (batched_graph, labels) in enumerate(train_loader):
        labels = labels.to(args['device'])
        optimizer.zero_grad()
        logits, _ = model(batched_graph)
        loss = loss_fcn(logits, labels)
        train_acc, train_micro_f1, train_macro_f1, train_buggy_f1 = score(labels, logits)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
        optimizer.step()
        total_accucracy += train_acc
        total_micro_f1 += train_micro_f1
        total_macro_f1 += train_macro_f1
        total_loss += loss.item()
        circle_lrs.append(optimizer.param_groups[0]["lr"])
    steps = idx + 1
    return model, total_loss/steps, total_micro_f1/steps, train_macro_f1/steps, total_accucracy/steps, train_buggy_f1/steps, circle_lrs