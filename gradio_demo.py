from typing import Optional, Union, Tuple, List, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gradio as gr
import numpy as np
import ptp_utils
from PIL import Image

from local_blend import LocalBlend
from null_inversion import NullInversion
from attention_control import AttentionStore, AttentionReplace, AttentionRefine, AttentionReweight, EmptyControl, AttentionControlEdit

# -------------------------------------------------------------------------
# Load model and define basic configurations
# -------------------------------------------------------------------------    
print("Initializing and loading StaleDiffsuion model...")
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
device = torch.device('cuda:6') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(device)
tokenizer = ldm_stable.tokenizer
g_cpu = torch.Generator().manual_seed(8888)

# -------------------------------------------------------------------------
# Utils code
# -------------------------------------------------------------------------    
def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, num_ddim_steps: int = 50, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_words, tokenizer, device=device, max_num_words=77) # The maximum number of words that StableDiffusion tokenizer can handle in one forward pass
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device, low_resource=False)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb, tokenizer=tokenizer, device=device, low_resource=False)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"], tokenizer)
        controller = AttentionReweight(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller, tokenizer=tokenizer, device=device, low_resource=False)
    return controller

# -------------------------------------------------------------------------
# Inference code
# -------------------------------------------------------------------------
@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def run_and_display(prompts, controller, 
                    num_ddim_steps: int = 50, guidance_scale: float = 7.5,
                    latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=num_ddim_steps, guidance_scale=guidance_scale, generator=generator, uncond_embeddings=uncond_embeddings, return_type="image")
    if verbose:
        ptp_utils.view_images(images)
    # images[0] is the original reconstructed image
    return images[1], x_t

# -------------------------------------------------------------------------
# Inference code
# -------------------------------------------------------------------------
def pipeline(
    image: np.ndarray, 
    offsets_input: str,
    original_prompt: str,
    edit_prompt: str,
    eq_str_input:str, 
    blend_word_str: str,  
    is_replace_controller: bool = False,
    guidance_scale: float = 7.5,
    num_ddim_steps: int = 50,
    cross_replace_steps_float: float = 0.8,
    self_replace_steps: float = 0.6,
    original_image_latent=None,
):
    # 1. preprocess input 
    offsets, prompts, cross_replace_steps, eq_params_output, blend_word = process_format(
        offsets_input, original_prompt, edit_prompt, cross_replace_steps_float, eq_str_input, blend_word_str
    )

    # For debug
    # print(f"Offsets: {offsets}, type: {type(offsets)}")
    # print(f"Blend words: {blend_word}, type: {type(blend_word)}")
    # print(f"Prompts: {prompts}, type: {type(prompts)}")
    # print(f"Cross replace steps: {cross_replace_steps}, type: {type(cross_replace_steps)}")
    # print(f"Eq params output: {eq_params_output}, type: {type(eq_params_output)}")
    # print(f"is_replace_controller: {is_replace_controller}, type: {type(is_replace_controller)}")
    
    if original_image_latent is not None:
        print("Using original image latent, omitting inversion to speed up the process.")
        x_t = original_image_latent
        uncond_embeddings = None
    else:
        # 2. Initialize NullInversion
        null_inversion = NullInversion(ldm_stable, num_ddim_steps=num_ddim_steps, guidance_scale=guidance_scale, device=device)

        # 3. Proform inversion
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image, original_prompt, offsets=offsets, verbose=True)

    # 4. Initialize controller
    controller = make_controller(prompts, is_replace_controller, cross_replace_steps, self_replace_steps, num_ddim_steps, blend_word, eq_params_output)

    # 5. Generate edited image
    edited_image, _ = run_and_display(prompts, controller, num_ddim_steps=num_ddim_steps, guidance_scale=guidance_scale, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False, generator=g_cpu) 

    print("Editting done.")
    return edited_image

def generate_pic_from_original_prompt(prompt: str, num_ddim_steps: int = 50, guidance_scale: float = 7.5):
    """
    Generate image from original prompt.
    """
    print(f"Generating original image from original prompt: {prompt}")
    controller = AttentionStore() # Placeholder, have no specific function in this case.
    image, latent = text2image_ldm_stable(ldm_stable, [prompt], controller, num_inference_steps=num_ddim_steps, guidance_scale=guidance_scale, return_type='image', start_time=num_ddim_steps)
    
    if image.shape[0] == 1:
        return image[0], latent
    elif image.shape[0] == 2:
        return image[1], latent
    else:
        raise ValueError("Invalid image shape. Expected 1 or 2 images.")

# -------------------------------------------------------------------------
# Preprocess code
# -------------------------------------------------------------------------
def process_format(
    offsets_input: str, # e.g., 0,0,0,1
    original_prompt: str,
    edited_prompt: str,
    cross_replace_steps: float,
    eq_str_input: str,
    blend_word_str: str,
):
    """
    Transforms the input format to be compatible with the model.
    1. 
    Inputs:
    """
    # 1. Parse offsets
    offsets = [int(x) for x in offsets_input.split(",")]

    # 2. Form promts
    prompts = [original_prompt, edited_prompt]

    # 3. Parse cross_replace_steps
    cross_replace_steps = {'default_': cross_replace_steps, }

    # 4. Parse eq_input
    if eq_str_input is not None:
        eq_params_output = process_eq_params_from_text(eq_str_input) # Dict
    else:
        eq_params_output = None

    # 5. Parse blend_tuple_output
    if blend_word_str is not None:
        blend_word = process_blend_tuple_from_text(blend_word_str) # List of tuples
    else:
        blend_word = None
    
    return offsets, prompts, cross_replace_steps, eq_params_output, blend_word

def process_eq_params_from_text(input_text):
    """
    解析用户输入的 `word-weight` 格式，生成 eq_params 字典。
    """
    eq_params = {"words": [], "values": []}
    pairs = [pair.strip() for pair in input_text.split(",") if pair.strip()]
    for pair in pairs:
        try:
            word, weight = pair.split("-")
            eq_params["words"].append(word.strip())
            eq_params["values"].append(float(weight.strip()))
        except ValueError:
            return f"Invalid format: {pair}. Use 'word-weight' format."
    eq_params["words"] = tuple(eq_params["words"])
    eq_params["values"] = tuple(eq_params["values"])
    return eq_params

def toggle_replace_controller(is_replace):
    """
    Returns a message based on the value of `is_replace`.
    """
    if is_replace:
        return "Replace Controller is enabled. This mode will replace cross-attention layers."
    else:
        return "Replace Controller is disabled. This mode will refine cross-attention layers."

def process_blend_tuple_from_text(blend_words: str):
    """
    Parse user input in the `word1-word2` format to generate a blend_tuple.
    """
    blend_tuple = []
    blend_words_from_original = []
    blend_words_from_edit = []
    blend_words = blend_words.split(",")
    for blend_word_pair in blend_words:
        if "-" in blend_word_pair:
            word1, word2 = blend_word_pair.split("-")
            blend_words_from_original.append(word1.strip())
            blend_words_from_edit.append(word2.strip())
        else:
            return f"Invalid format: {blend_word_pair}. Use 'word1-word2' format."
    
    blend_tuple.append(tuple(blend_words_from_original))
    blend_tuple.append(tuple(blend_words_from_edit))

    # Return tuple
    return blend_tuple

# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------
cat_image = "./example_images/gnochi_mirror.jpeg"

# -------------------------------------------------------------------------
# Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Soft()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)
with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    gr.Markdown("# Prompt-to-Prompt Gradio Demo")
    original_image_latent = gr.State()
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(height=384, width=384, label="Input Image", type="numpy", interactive=True, show_download_button=True)
            with gr.Row():
                original_prompt = gr.Textbox(label="Edit prompt", placeholder="Enter your original prompt here")        
                original_generate_button = gr.Button("Generate", elem_id="my_button")


        with gr.Column():
            output_final = gr.Image(height=384, width=384, label="Edited Image", interactive=True, show_download_button=True)
            with gr.Row():
                edited_prompt = gr.Textbox(label="Edit prompt", placeholder="Enter your edit prompt here")        
                edit_generate_button = gr.Button("Generate", elem_id="my_button")

    with gr.Row():
        offsets_input = gr.Textbox(label="Offsets (left,right,top,bottom)", value="0,0,0,0")
        guidance_scale = gr.Number(value=7.5, label='Guidance scale', interactive=True)
        num_ddim_steps = gr.Number(value=50, label='Diffusion steps', interactive=True)
    
    with gr.Row():
        cross_replace_steps_input = gr.Number(value=0.8, label="Cross Replace Steps", interactive=True, minimum=0.0, maximum=1.0, step=0.1)
        self_replace_steps_input = gr.Number(value=0.4, label="Self Replace Steps", interactive=True, minimum=0.0, maximum=1.0, step=0.1)     

    with gr.Row():
        blend_word_str = gr.Textbox(
            label="Blend Words",
            placeholder="Enter blend words in 'word1_in_original-word1_in_edit, word2_in_original-word2_in_edit' format"
        )

    with gr.Tab("Replace/Refine"):   
        gr.Markdown("# Replace/Refine mode configuration")
        is_replace_controller = gr.Checkbox(label="Enable Replace Controller", value=False, interactive=True)
        replace_info = gr.Markdown("Replace Controller is disabled. This mode will refine cross-attention layers.",
                                   elem_classes=["custom-log"])

        
        is_replace_controller.change(
            toggle_replace_controller,
            inputs=[is_replace_controller],
            outputs=[replace_info],
        )

    with gr.Tab("Reweight"):   
        gr.Markdown("# Reweight Parameters Configuration")

        with gr.Row():
            eq_str_input = gr.Textbox(
                label="Equalizer Input",
                placeholder="Enter words and weights in 'word1-weight1, word2-weight2' format",
            )

    original_generate_button.click(
        fn=generate_pic_from_original_prompt,
        inputs=[original_prompt, num_ddim_steps, guidance_scale],
        outputs=[image_input, original_image_latent],
    )
    
    edit_generate_button.click(
    fn=pipeline,
    inputs=[
        image_input, offsets_input, original_prompt, edited_prompt, 
        eq_str_input, blend_word_str, is_replace_controller,
        guidance_scale, num_ddim_steps, cross_replace_steps_input, self_replace_steps_input,
        original_image_latent
    ],
    outputs=[output_final]
    )

    # ---------------------- Examples section ----------------------
    examples = [
        [
            cat_image,
            "0,0,200,0",
            "a cat sitting next to a mirror",
            "a tiger sitting next to a mirror",
            'tiger-2.0',  
            "cat-tiger",
            True,
            7.5,
            50,
            0.8,
            0.5,
        ],
    ]

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            image_input,
            offsets_input,
            original_prompt,
            edited_prompt,
            eq_str_input,
            blend_word_str,
            is_replace_controller,
            guidance_scale,
            num_ddim_steps,
            cross_replace_steps_input,
            self_replace_steps_input,
        ],
        outputs=[
            output_final
        ],
        fn=pipeline,
        cache_examples=False,
        examples_per_page=50,
    )    

if __name__ == "__main__":
    demo.launch(show_error=True, share=True)
