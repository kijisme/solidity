from torch.utils.tensorboard import SummaryWriter

def visualize_k_folds(log_dir, k_folds, num_epochs, train_results, val_results):
    writer = SummaryWriter(log_dir)
    for fold in range(k_folds):
        for idx in range(num_epochs):
             
            writer.add_scalars('Loss', {f'train_{fold+1}': train_results[fold]['loss'][idx],
                                        f'valid_{fold+1}': val_results[fold]['loss'][idx]}, 
                                        idx)
             
            writer.add_scalars('Accuracy', {f'train_{fold+1}': train_results[fold]['acc'][idx],
                                            f'valid_{fold+1}': val_results[fold]['acc'][idx]}, 
                                            idx)
            
            writer.add_scalars('Micro_f1', {f'train_{fold+1}': train_results[fold]['micro_f1'][idx],
                                            f'valid_{fold+1}': val_results[fold]['micro_f1'][idx]}, 
                                            idx)
            
            writer.add_scalars('Macro_f1', {f'train_{fold+1}': train_results[fold]['macro_f1'][idx],
                                            f'valid_{fold+1}': val_results[fold]['macro_f1'][idx]}, 
                                            idx)
            
            # writer.add_scalars('Train_results', {f'buggy_f1_{fold+1}': train_results[fold]['buggy_f1'][idx],
            #                                      f'macro_f1_{fold+1}': train_results[fold]['macro_f1'][idx]},
            #                                      idx)
            
            # writer.add_scalars('Val_results', {f'buggy_f1_{fold+1}': val_results[fold]['buggy_f1'][idx],
            #                                    f'macro_f1_{fold+1}': val_results[fold]['macro_f1'][idx]}, 
            #                                    idx)
            
    writer.close()
