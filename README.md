# VAE annealing
## VAELoss
 VAELoss is a subclass of torch.nn.Module. It is a standard loss function for training VAE (variational autoencoder) neural network. It is callable and takes four argments: 
 * x (torch.Tensor): reconstructed input tensor
 * x0 (torch.Tensor): input tensor
 * mu (torch.Tensor): mean of the latent space
 * logvar (torch.Tensor): log variance of the latent space
  
 When called, it returns two instances of single-valued torch.Tensor, which can be passed to an optimizer: 
 * bce (recon loss component)
 * kld (divergence loss component)
  
## Annealer  
 Annealer is a class. An instance is created by passing two parameters:
 * slope_length (int): Number of epochs (steps) to reach full KL divergence weight
 * shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.  
 Annealer can be instantiated with disable=True parameter. This way an intance of Annealer can be still called, but the argument passed to the __call__() method will be returned unchanged.
  
An instance of Annealer is callable.  
Args:
 * kld (torch.tensor): Kullback–Leibler divergence loss (or any ohter loss object for which multiplication by a scalar is defined).
  
Returns:  
 * torch.tensor: KL divergence loss multiplied by the value of the annealing function
  
![Basic anneling shapes](https://github.com/hubertrybka/vae-annealing/blob/main/shapes.png)
 
 ## Usage
 ### Basic annealing
 An instance of Annealer class stores total_steps and current_steps integer attributes and uses them to calculate the value of annealing coefficient at each step. Annealer is to be instantiated before the training loop, ex:  
 ```
 from loss import VAELoss
 from annealing import Annealer
  
 annealing_agent = Annealer(slope_length, shape='cosine')  # instantiating annealing agent
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
 After each epoch `annealing_agent.step()` method is to be called. This increases the current_step attribute by 1, unitl the value matches total_steps attribute. Based on those values, the slope of an annealing function is calculated. Initial value of current_step is always 0.

 ### Cyclical annealing
 A method proposed by Fu _et al._ in their work [Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing](https://arxiv.org/abs/1903.10145)  
   
 ![Cyclkical anneling shapes](https://github.com/hubertrybka/vae-annealing/blob/main/cyclical_shapes.png)

 Cyclical annealing is enabled by instantiating the Annealer with cyclical=True parameter. When this is done, every time the step counter (current_step) reaches total_steps, it's value is set to zero. The annealing cycle starts over again until it is disabled.  
   
 The cyclical annealing functionality can be disabled (or enabled) during the course of training by passing False (or True) vale to the Annealer.cyclical_setter() method. An example of cyclical annealing being disabled during it's second cycle is illustrated below:
 ```
 annealing_agent = Annealer(slope_length, shape='cosine', cyclical=True)
 # instantiating annealing agent with cyclical annealing functionality
  
 for epoch in range(100):  # training loop starts here
 ... # training script
 if epoch == 30:
     annealing_agent.cyclical_setter = False
 ```  
 The expected outcome:  
 ![Cyclkical anneling disable](https://github.com/hubertrybka/vae-annealing/blob/main/cyclical_disable.png)
