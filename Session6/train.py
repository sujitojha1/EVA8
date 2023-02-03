

def train(net, trainloader, device, criterion, optimizer, scheduler, EPOCHS):
    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 400 == 399:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

        scheduler.step()
        epoch_loss = running_loss/len(trainloader.dataset)
        print(f'epoch {epoch}, loss {epoch_loss}')
    print('Finished Training')