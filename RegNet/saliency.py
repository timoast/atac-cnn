import torch
from torch import nn
import torch.nn.functional as F


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
    

# new model for determining filter influence
# based on AI-TAC
class MotifModel(nn.Module):

    def __init__(self, original_model):
        super(MotifModel, self).__init__()
        # extract first layer weights from the model
        self.conv1 = list(original_model.children())[0][0]
        self.process_conv1 = list(original_model.children())[0][1:]
        self.get_outputs = nn.Sequential(*list(original_model.children())[1:])
        # NOTE this is specific to the original model architecture, will need to update if changing models
        
    def forward(self, inputs):
        conv1_outs = self.conv1(inputs)
        
        # save first layer filter activations
        layer1_activations = torch.squeeze(conv1_outs)
        
        # get predicted outputs using all filters
        conv1_outs = self.process_conv1(conv1_outs)
        outs = self.get_outputs(conv1_outs)
        
        # remove each filter one at a time are record output vector
        batch_size = conv1_outs.shape[0]
        # NOTE dimensions here depend on model architecture
        n_filters = 300
        batch_size = conv1_outs.shape[0]
        n_class = 15
        n_bases = 1000
        predictions = torch.zeros(batch_size, n_filters, n_class)
        
        for i in range(n_filters):
            #modify filter i of first layer output
            filter_input = conv1_outs.clone()
            
            # setting filter outputs to zero. AI-TAC sets it to the batch mean value
            filter_input[:,i,:] = filter_input.new_full((batch_size, n_bases), fill_value=0)
            pred_filter_removed = self.get_outputs(filter_input)
            predictions[:,i,:] = outs - pred_filter_removed
        
        return layer1_activations.detach(), outs.detach(), predictions.detach()
    