import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import uuid
from ..model_utils import requires_grad
import torchvision.transforms as transforms
from raw_data import utils


class StableDiffWrapper(torch.nn.Module):
    def __init__(self, args, device='cuda', dtype=torch.float16):
        super(StableDiffWrapper, self).__init__()
        
        self.latent_only = args.latent_only
        self.steps_train = args.steps_train
        self.latent_steps_train = args.latent_steps_train
        self.latent_dim = (4, 128, 128) # (4, 128, 128) ch, H, W dims in latent space for 1024x1024 images
        self.image_idx = 0 # start

        # Set up generator  
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_8step_unet.safetensors"
        
        # Load UNet model
        self.unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
        self.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        requires_grad(self.unet, False) # Freeze UNet weights

        # Initialize pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=self.unet, torch_dtype=dtype, variant="fp16").to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")

    def forward(self, z, prompt=None, num_inference_steps=8, guidance_scale=0):

        if prompt is None or len(prompt) == 0:
            prompt = utils.read_prompts(size=len(z))                
            
        z = z.half()

        # Initialize an empty tensor to store the images
        B = z.shape[0]
        C, H, W = 3, 1024, 1024  # Output image dimensions
        images = torch.zeros((B, C, H, W), dtype=torch.float32, device=z.device)
        
        # Define a transform to convert PIL images to tensors
        transform = transforms.ToTensor()     
        # Process each latent in the batch
        for i, latent in enumerate(z):
            latent = latent.unsqueeze(0)  # Add a batch dimension
            result = self.pipe(prompt[i], latent=latent, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            pil_image = result.images[0]  # Extract the PIL image
            # pil_image.save(f"{uuid.uuid4().hex}.png")
            tensor_image = transform(pil_image).to(z.device)  # Convert PIL image to tensor and move to the correct device
            images[i] = tensor_image
            
        return images

