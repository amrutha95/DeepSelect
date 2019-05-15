#Creating adversarial examples

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    s = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*s
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    # Return the perturbed image
    return perturbed_image

def attack(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = Variable(data).type(torch.cuda.FloatTensor), Variable(target).type(torch.cuda.LongTensor)
        oridata = data.squeeze().cpu().numpy()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        model.eval()
        output = model(data)
        worstclass = output.min(1, keepdim=True)[1][0]
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        #loss = -F.nll_loss(output, worstclass)
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        data = fgsm_attack(data, epsilon, data_grad)
        output = model(data)
        final_pred = output.max(1, keepdim=True)[1]
        
        count = 0
        while(final_pred.item() == target.item() and count < 10000):
          data = fgsm_attack(data, epsilon, data_grad)
          output = model(data)
          final_pred = output.max(1, keepdim=True)[1]
          count = count + 1
          
        if len(adv_examples) < 100 and final_pred.item() != target.item():
          if count < 10:
            print('{} done ({}).'.format(len(adv_examples), count))
            adv_ex = data.squeeze().detach().cpu().numpy()
            adv_examples.append( (oridata, init_pred.item(), final_pred.item(), adv_ex) )  #original data, correct prediction, final prediction, adversarial example generated
          else:
            pass
        elif len(adv_examples) >= 100:
          return adv_examples

    # Return the accuracy and an adversarial example
    return adv_examples
  
adv_examples = test(model, loaders['test_loader'], 0.1)
adv_example = adv_examples[3]


plt.imshow(adv_example[0].transpose(1, 2, 0))
plt.title("Prediction: {}".format(adv_example[1]))
plt.figure()
plt.imshow(adv_example[3].transpose(1, 2, 0))
plt.title("Adversarial Prediction: {}".format(adv_example[2]))
