import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Union, Optional, Iterable


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: int,
    image_seq_len: int,
    image_token: str,
) -> str:
    
    # Taken from HF
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"    

def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    
    height, width = size
    resized_image = image.resize(
        size=(width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
    image: np.ndarray,
    scale: float = 1.0,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
    image: np.ndarray,
    mean: Optional[Union[float, List[float]]],
    std: Optional[Union[float, List[float]]],
) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    
    normalized_image = (image - mean) / std
    return normalized_image

def process_images(
    images: List[Image.Image],
    size: Tuple[int, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    
    height, width = size
    images = [
        resize(image=image, size=(height, width), resample=resample)
        for image in images
    ]
    
    # convert each image to a numpy array
    images = [np.array(image) for image in images]
    # rescale the pixel values to be between 0 and 1
    images = [rescale(image, scale=rescale_factor) for image in images]
    # normalize the images
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension
    # The model expects [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images

class PaliGemmaProcessor:
    
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        # Tokenizer described here: <link> # TODO: Add link
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        OBJECT_DETECTION_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # These are tokens used for object detection
        
        SEGMENTATION_TOKENS = [
            f"<seg{i:03d}>" for i in range(128)
        ] # These are tokens used for object segmentation
        
        tokenizer.add_tokens(OBJECT_DETECTION_TOKENS + SEGMENTATION_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_eos_tokens = False
        tokenizer.add_bos_tokens = False
        
        self.tokenizer = tokenizer
        
    
    def __call__(
        self,
        text: Union[str, List[str]],
        images: Union[Image.Image, List[Image.Image]],
        padding: str = "longest",
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        
        if isinstance(text, str):
            text = [text]
        if isinstance(images, Image.Image):
            images = [images]
        
        assert len(images) == 1 and len(text) == 1, "Only one image and one text is supported"
        
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0/255,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        
        # convert the list of numpy arrays to a single numpy array with shape [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
        # basically adding the batch dimension
        pixel_values = np.stack(pixel_values, axis=0)
        # convert the numpy array to a torch tensor
        pixel_values = torch.tensor(pixel_values)
        
        # Prepend a `self.image_token_id` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token_id,
                image_seq_len=self.image_seq_length,
                image_token_id=self.IMAGE_TOKEN,
            ) for prompt in text
        ]
        
        # Return the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}
        
        return return_data
    
        
        
        