import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

# FGSM attack code
def fgsm_attack(images, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon*sign_data_grad
    return perturbed_images

# restores the tensors to their original scale
def denorm(batch, mean, std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def eval_fgsm(model, device, test_loader, epsilon, mean, std, verbose=False, is_polyact=False):
    model.eval()
    correct = 0

    # Loop over all examples in test set
    for images, target in test_loader:

        # Send the data and label to the device
        images, target = images.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        images.requires_grad = True

        # Forward pass the data through the model
        output = model(images)
        if is_polyact:
          output = output.T

        loss = F.multi_margin_loss(output, target, margin=1e10)

        # print(loss.item())

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = images.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(images, mean, std)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        
        # Reapply normalization THESE VALUES ARE FOR CIFAR-10
        perturbed_data_normalized = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(perturbed_data)

        
        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)
        if is_polyact:
          output = output.T
          
        final_pred = output.argmax(dim=1) 
        correct = correct + (final_pred == target).float().sum()

        del images
        del target
        torch.cuda.empty_cache()
    # Calculate final accuracy for this epsilon
    final_acc = (correct/float(len(test_loader.dataset))).item()

    if verbose:
      print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc 