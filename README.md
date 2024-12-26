# VAE annealing

## What is it?

KLD (Kullback–Leibler divergence) annealing is a technique used in training variational autoencoders (VAE), 
specifically those based on an autoregressive decoder (ex. RNN such as LSTM or GRU). It is used to prevent the KL 
divergence term from vanishing during training. During the initial stages of training, as the encoder has not yet 
learned a meaningful representation of the training data, the autoregressive decoder may become misled by the 
ineffectuality of latent encodings and learn to ignore that information entirely. This results in a highly organized 
latent space but a poor reconstruction efficiency of the trained model and will negatively impact the overall 
performance of the autoencoder.  
  
![Basic annealing shapes](https://github.com/hubertrybka/vae-annealing/blob/main/figures/annealing_example.png)

This repository contains a simple implementation of KLD annealing and a PyTorch VAE loss function.

# Documentation

## Annealer  
 Annealer is a class. An instance is created by passing two parameters:
 * total_steps (int): Number of epochs (steps) to reach full KL divergence weight
 * shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.  
 Annealer can be instantiated with disable=True parameter. This way an instance of Annealer can be still called, but the argument passed to the __call__() method will be returned unchanged.
  
An instance of Annealer is callable.  
Args:
 * kld (torch.tensor): Kullback–Leibler divergence loss (or any other loss object for which multiplication by a scalar is defined).
  
Returns:  
 * torch.tensor: KL divergence loss multiplied by the value of the annealing function


## VAELoss
 VAELoss is a subclass of torch.nn.Module. It is a standard loss function for training VAE (variational autoencoder) neural networks. It is callable and takes four arguments: 
 * x (torch.Tensor): reconstructed input tensor
 * x0 (torch.Tensor): input tensor
 * mu (torch.Tensor): mean of the latent space
 * logvar (torch.Tensor): log variance of the latent space
  
 When called, it returns two instances of single-valued torch.Tensor, which can be passed to an optimizer: 
 * bce (recon loss component)
 * kld (divergence loss component)
  
![Basic annealing shapes](https://github.com/hubertrybka/vae-annealing/blob/main/figures/shapes.png)
 
 ## Usage
 ### Basic annealing
 An instance of the Annealer class stores total_steps and current_steps integer attributes and uses them to calculate the value of the annealing coefficient at each step. Annealer is to be instantiated before the training loop, ex:  
 ```
 from loss import VAELoss
 from annealing import Annealer
  
 annealing_agent = Annealer(total_steps, shape='cosine')  # instantiating annealing agent
 criterion = VAELoss()  # instantiating VAELoss
  
 for epoch in range(100):  # training loop starts here
     x, mu, logvar = model(x0)
     bce, kld = criterion(x, x0, mu, logvar)
     kld = annealing_agent(kld)
     optimizer.zero_grad()
     (bcd + kld).backward
     optimizer.step()
     annealing_agent.step()
 ```  
 After each epoch `annealing_agent.step()` method is to be called. This increases the current_step attribute by 1 until the value matches the total_steps attribute. Based on those values, the slope of an annealing function is calculated. The initial value of current_step is always 0.

 ### Cyclical annealing
 A method proposed by Fu _et al._ in their work [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145)  
   
 ![Cyclical annealing shapes](https://github.com/hubertrybka/vae-annealing/blob/main/figures/cyclical_shapes.png)

 Cyclical annealing is enabled by instantiating the Annealer with cyclical=True parameter. When this is done, every time the step counter (current_step) reaches total_steps, its value is set to zero. The annealing cycle starts over again until it is disabled.  
   
 The cyclical annealing functionality can be disabled (or enabled) during training by passing False (or True) value to the Annealer.set_cyclica() method. An example of cyclical annealing being **disabled at epoch 30** and **enabled again at epoch 70** is illustrated below:
 ```
 annealing_agent = Annealer(total_steps, shape='cosine', cyclical=True)
 # Instantiating annealing agent with cyclical annealing functionality
  
 for epoch in range(100):
     ...  # training script

     if epoch == 30:
         annealing_agent.set_cyclical(False)
     if epoch == 70:
         annealing_agent.set_cyclical(True)
 ```  
 The expected outcome:  
   
 ![Cyclical annealing disable](https://github.com/hubertrybka/vae-annealing/blob/main/figures/enable_disable.png)
