from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import noise_schedule
import unet

class ConditionalDiffusion(nn.Module):
    def __init__(
        self,
        generated_channel=3, # Channels of noised / generated image
        condition_channel=3, # Condition to embed, in our case 3 channels of synthetic low light image.
        schedule="linear",
        timesteps=1000,
        sampler=None,
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.generated_channel = generated_channel
        self.condition_channel = condition_channel
        self.timesteps = timesteps
        #TODO: Implement other option of sampler, ie. DDIM for quicker inference
        self.sampler = sampler
        in_channel = generated_channel + condition_channel # Concat conditin to input image channels
        self.device = device
        self.model = unet.Unet(in_channel, generated_channel).to(device)
        self.schedule = schedule
        self.set_up_noise_schdule()
        
    def set_up_noise_schdule(self):
        # Beta noise schedule
        self.betas = noise_schedule.get_beta_schedule(self.schedule, self.timesteps)

        # Alphas
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
                
    @torch.no_grad()
    def q_sample(self, x_0, t, noise=None):
        """
        Forward noising process to sample noisy image at timestep t,
        Input:
            x_0: Natual input image
            t: Timestep t in shape (batch_size, )
        Returns: Noisy image tensor, and actual noise.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise.to(t.device)
       
        
    def loss(self, x_0, condition, t, noise=None):
        """
        Compute loss by comparing predicted noise from Unet, with actual noise injected into image at
        timestep t.
        Input:
            x_0: Input image at timestep 0 (no noise). Value in range [-1, 1]
            condition: Condition, ie. synthetic low light image. Value in range [-1, 1]
            t: Timestep t
        Returns: Mean sqaure loss between actual noise and predicted noise.
        
        """
        # Actual noisy image at timestep t according to noise schedule
        x_noisy, noise = self.q_sample(x_0, t, noise)
        
        # To apply conditional diffusion, we concat concat condition (ie. synthetic low light image) with
        # noisy image at timestep t.
        model_input=torch.cat([x_noisy, condition], 1).to(x_0.device)
        noise_pred = self.model(model_input, t)
        return F.mse_loss(noise_pred, noise)
        
        
    def forward(self, x_0, condition, noise=None):
        """
        Randomly sample a timestep from [0, T) to compute loss.
        """
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,),
                          device=x_0.device).long()
        
        return self.loss(x_0, condition, t, noise)
    
    
    @torch.no_grad()
    def p_sample(self, x_t, condition, t):
        """
        Inference: Generate image x at timestep t-1, by predicting noise
        injected resulting in x at timestep t.
        Inputs:
            x_t: Noisy image at timestep t
            condition: Conditon to guide diffusion model to generate image. In our case, synthetic
            low light iamge.
            t: Timestep t
        Returns: Generated iamge at t-1 with condition
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Refer to sampling equation in report
        model_input=torch.cat([x_t, condition], 1).to(x_t.device)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * self.model(model_input, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
        
        if t[0].item() == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
        
    @torch.no_grad()
    def p_sample_progressive(self, condition):
        """
        Inference: Full cycle from timestep T to 0, starting from pure noise. 
        Condition will guide image generated to be related to synthetic low light image.
        """
        # Sample image from pure noise at timestep T to timestep 0
        # shape: (B, C, H, W)
        b, c, h, w = condition.shape
        device = next(self.parameters()).device
        condition = condition.to(device)
        
        # Generate pure noise as input for inference
        x = torch.randn([b, self.generated_channel, h, w], device=device)

        images = []
        # Backward denoise with the model to produce image at timestep 0.
        for i in tqdm(range(0, self.timesteps)[::-1]):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, condition, t)
            # Make the image within range of [-1, 1]
            # x = torch.clamp(x, -1.0, 1.0)
            images.append((torch.clamp(x, -1.0, 1.0)).detach().cpu())
        return images
    
    def load_checkpoint(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path))
        
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        
    def get_model_parameters(self):
        return self.model.parameters()
        
# Extract value according to batch size
def extract(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
