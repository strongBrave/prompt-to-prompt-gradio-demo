# Prompt-to-Prompt with Null Inversion

This project demonstrates the use of **Prompt-to-Prompt editing** with **Null Inversion** for fine-grained image editing using Stable Diffusion. The implementation is wrapped in a Gradio interface for ease of use.

## Note

- This method is less effective for prompts with significant changes compared to the original prompt. For best results, the `edit prompt` should closely resemble the `original prompt`, except for the parts being edited.
- The editing functionality corresponds to the three types of edits described in the original [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) paper: **Replace**, **Refine**, and **Reweight**. Ensure that the `edit prompt` aligns with these editing principles for optimal results.

## TODO
- [ ] Add more examples and detailed usage instructions.
- [ ] Incorporate image-to-prompt generation for better user experience.

## Features

- **Prompt-to-Prompt Editing**: Modify images by editing text prompts while preserving the structure and style of the original image.
- **Null Inversion**: Enables inversion of an image into the latent space of Stable Diffusion for further editing.
- **Attention Control**: Supports Replace, Refine, and Reweight modes for controlling cross-attention layers.
- **Gradio Interface**: User-friendly interface for uploading images, editing prompts, and visualizing results.

## Requirements

- Python 3.8+
- CUDA-enabled GPU
- Required Python libraries:
  - `torch`
  - `diffusers`
  - `transformers`
  - `gradio`
  - `numpy`
  - `Pillow`
  - `tqdm`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd p2p_gradio
   ```

2. Launch the Gradio demo:
   ```bash
   python gradio_demo.py
   ```

3. Open the Gradio interface in your browser. You can:
   - Upload an input image or generate input image using the original prompt.
   - Provide an original prompt and an edited prompt.
   - Configure parameters like guidance scale, diffusion steps, and attention control modes.
   - Generate and download the edited image.

## Key Components

### Null Inversion
Null Inversion is used to invert an input image into the latent space of Stable Diffusion. This allows for precise editing while maintaining the original image's structure.

### Attention Control Modes
> For detailed explanations, refer to the [Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626) paper.
- **Replace**: Replaces the attention score in cross-attention layers to enforce prompt changes.
- **Refine**: Refines cross-attention layers for subtle edits.
- **Reweight**: Adjusts the weights of specific words in the prompt for fine-grained control.

### Gradio Interface
The Gradio interface provides an intuitive way to interact with the model. Users can:
- Upload images.
- Edit prompts.
- Configure advanced parameters like offsets, blend words, and equalizer settings.

## Example

### Input
- **Image**: A cat sitting next to a mirror.
- **Original Prompt**: "a cat sitting next to a mirror"
- **Edited Prompt**: "a tiger sitting next to a mirror"
- **Blend Words**: `cat-tiger`
- **Equalizer Input**: `tiger-2.0`

### Output
An edited image where the cat is replaced by a tiger, while preserving the original structure and style.

## File Structure

- `gradio_demo.py`: Main script for the Gradio interface.
- `ptp_utils.py`: Utility functions for Prompt-to-Prompt editing.
- `null_inversion.py`: Implementation of Null Inversion.
- `attention_control.py`: Attention control mechanisms.
- `local_blend.py`: Local blending for fine-grained edits.
- `example_images/`: Folder containing example input images.

## References

- [Prompt-to-Prompt Editing](https://arxiv.org/abs/2208.01626)
- [Null Text Inversion](https://arxiv.org/abs/2211.09794)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Acknowledgments

This project is built on top of the [prompt-to-prompt](https://github.com/google/prompt-to-prompt) codabase and incorporates demos from its original code.

## License

This project is licensed under the MIT License. See the LICENSE file for details.