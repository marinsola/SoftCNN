import time
def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def train(model, optimizer, loader, n_iters=10, criterion=nn.CrossEntropyLoss(),
          device=None, output_batch=False, testloader=None, CNN_mode=False):
    '''CNN_mode : bool, whether or not using matrix input'''
    
    start_time = time.time()
    model.train()
    print('Started training.')
    
    if device is not None:
        
        model.to(device)
        
    for epoch in range(n_iters):
        
        epoch_start_time = time.time()
        epoch_loss = 0
        batchnum = 0
        
        for x,y in loader:
            
            if device is not None:
                
                x, y = x.to(device),y.to(device)
                
            if CNN_mode:
                xn = x
            else:
                xn = torch.flatten(x,2)
                
            y_hat = model(xn)
            loss = criterion(y_hat, y)  
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batchnum += 1
            
            if output_batch:
                
                OPS = '    Batch: {}, batch_loss: {:.4f}, time: {}'
                print(OPS.format(batchnum, loss.item(), format_elapsed(epoch_start_time)))
                
        if testloader:
            
            acc = evaluate(model,testloader,device=device,CNN_mode=CNN_mode)
            
        else:
            
            acc = None
            
        OPS2='''Epoch: {}/{}, epoch_loss: {:.4f}, epoch time: {}, time since start: {},
                validation set accuracy: {:.4f}'''
        print(OPS2.format(epoch+1,n_iters,epoch_loss,format_elapsed(epoch_start_time),
                          format_elapsed(start_time), acc))
        
    print('Training done.')

def evaluate(model,testloader,device='cpu',CNN_mode=False):
    """
    Function for evaluating accuracy of a model on a testset. 
    Expects vector input models. For matrix input set CNN_mode=True. 
    """    

    model.eval()
    model.to(device)
    softmax = nn.Softmax(dim=1)
    acclist = []
    for x,y in testloader:
        
        if device is not None:               
            x, y = x.to(device),y.to(device)
        
        if CNN_mode:
            xn = x
        else:
            xn = torch.flatten(x,2)
            
        y_p = model(xn)        
        accuracy = torch.mean((torch.argmax(softmax(y_p),dim=1)==y).float())
        acclist.append(accuracy)
        
    accuracy = torch.mean(torch.stack(acclist))
    model.train()
    
def plot_weights(model, absolute = True, cmap = 'viridis',
                 blackout = 0.02, figsize = (100,400),
                 parallel=False):
    """
    Function for plotting the v-vectors (of the first block) of a MixBlock model.
    Input parameters should be kept at standard values. If we used data parallelism
    for training the model, set parallel=True.
    Saves the plot as "Plot.jpg".
    """
    
    if parallel:
        model = model.module
    total = model.conv1.out_channels * model.conv1.in_channels
    fig, axs = plt.subplots(total//3, 3, figsize = figsize)
    with torch.no_grad():
        for i in range(total):
            if absolute:
                
                # absolute values of vector entries
                activation = torch.abs(list(model.parameters())[2*i]).to('cpu')
                # entropy of the vector
                probs = activation / torch.sum(activation) 
                activity = - torch.sum(probs*torch.log(probs))

            else:
                activation = (list(model.parameters())[2*i]).to('cpu')
            row = i // 3
            col = i % 3
            pic_res = int(sqrt(model.conv1.in_size))
            axs[row,col].imshow(activation.reshape(pic_res, pic_res),
                                vmin = 0, cmap = cmap)
            axs[row, col].text(0.5, -0.1, "{:.2f}".format(activity.item()), size=40, 
                               ha="center", va = 'bottom', 
                               transform=axs[row, col].transAxes)

    plt.tight_layout()
    plt.savefig("Plot")
    plt.show()