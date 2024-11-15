import fire
from PIL import Image
from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_model
from typing import Optional

import torch

def move_inputs_to_device(inputs, device):
    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    return model_inputs

def _sample_top_p(logits, top_p=0.9):
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(logits, dim=-1, descending=True)
    prob_sum = torch.cumsum(probs_sort, dim=-1)
    # subtracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking
    mask = prob_sum - probs_sort > top_p
    # zero out all the proabilities below the top_p
    probs_sort[mask] = 0.0
    # re-normalize the probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # sample from the modified probabilities
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # get the token position in the vocab
    next_token = probs_idx.gather(dim=-1, index=next_token)
    return next_token

def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: Optional[str], device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: Optional[str] = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.5,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
        
    kv_cache = KVCache()
    
    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []
        
    for _ in range(int(max_tokens_to_generate)):
        # Get the model outputs
        # TODO: remove the labels
        
        # print("Input IDs shape:", input_ids.shape)
        # print("Attention Mask shape:", attention_mask.shape)
        # print("Pixel Values shape:", pixel_values.shape)
        
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )
        
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        
        # Sample the next token
        if do_sample:
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            next_token_logits = torch.softmax(next_token_logits, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        
        # stop if the stop token is generated
        if next_token.item() == stop_token:
            break
        
        # Append the next token to the input
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )
        
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    # decode the tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("Prompt:", prompt)
    print("Generated:", decoded)

def main(
    model_path: str,
    prompt: str,
    image_file_path: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 0.5,
    top_p: float = 0.9,
    do_sample: bool = False,
    device: str = "cuda",
):
    
    if torch.cuda.is_available():
        device = "cuda" if device == "cuda" else "cpu"
    else:
        if device == "cuda":
            print("WARNING: CUDA is not available. Running on CPU")
            device = "cpu"
    
    print(f"Device: {device}")
    print(f"Loading Model path: {model_path}")
    
    model, tokenizer = load_model(model_path, device)
    model = model.to(device).eval()
    
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    print(f"Number of image tokens: {num_image_tokens}")
    print(f"Image size: {image_size}")
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
    
    print("Running Inference")
    
    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens,
            temperature,
            top_p,
            do_sample,
        )

if __name__ == "__main__":
    fire.Fire(main)
