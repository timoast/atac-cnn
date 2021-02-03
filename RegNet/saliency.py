import torch


def saliency_map(model, inputs, device="cuda"):
    model = model.to(device)
    with torch.set_grad_enabled(True):
        inputs = inputs.to(device)
        inputs.requires_grad_()
        y_hat = model(inputs[None,:,:])
        dummy = torch.ones_like(y_hat).to(device)
        y_hat.backward(dummy)
        gradients = inputs.grad.detach()
        saliency = gradients * inputs.detach()
        saliency = saliency.cpu()
        return saliency


def ism(model, inputs, device = "cuda"):
    with torch.set_grad_enabled(False):
        model = model.to(device)
        inputs = inputs.to(device)
        outputs = model(inputs[None,:,:]).squeeze()
        delta_out = torch.zeros_like(inputs, device=device)
        # change each base in input and re-compute output
        for seq_idx in range(inputs.shape[1]):
            # find which base is non-zero
            base_idx = inputs[:,seq_idx].argmax()
            bases_sum = 0.
            for nt_idx in range(4): # iterate over all the bases
                x_copy = inputs.clone()
                # mutate base
                x_copy[:,seq_idx] = 0.
                x_copy[nt_idx,seq_idx] = 1.
                mutated_output = model(x_copy[None,:,:]).squeeze()
                # record the sum of absolute differences across all classes
                bases_sum += sum(abs(mutated_output - outputs))
            delta_out[base_idx,seq_idx] = bases_sum
        return delta_out.cpu()