from modules import script_callbacks
import gradio as gr

import torch
import os
from transformers import pipeline

def pipe(single_image_file):
    pipe = pipeline("image-classification", model="shadowlilac/aesthetic-shadow-v2", device=0)
    result = pipe(images=[single_image_file])
    prediction_single = result[0]
    score = str(round([p for p in prediction_single if p['label'] == 'hq'][0]['score'], 2))
    return score

def add_tab():

    MARKDOWN = \
    """
    # sd-webui-aesthetic-shadow

    [GitHub](https://github.com/gluttony-10/sd-webui-aesthetic-shadow) | [Bilibili](https://space.bilibili.com/893892)

    Used for [AUTOMATIC1111's stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
    """
        
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            gr.Markdown(MARKDOWN)
        with gr.Row(equal_height=True):
            with gr.Tabs():
                with gr.TabItem(label='Single image'):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(type="filepath", label="Input")
                            run_btn = gr.Button(value="Submit")
                        with gr.Column():
                            output_text = gr.Textbox(label="Output")
                    run_btn.click(pipe, inputs=[input_image], outputs=[output_text])
            
    return [(ui, "aesthetic shadow", "aesthetic-shadow")]




















script_callbacks.on_ui_tabs(add_tab)