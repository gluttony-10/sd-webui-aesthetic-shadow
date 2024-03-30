from modules import script_callbacks
import gradio as gr

import torch
import os
from transformers import pipeline

import json
from glob import glob
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from collections import OrderedDict
from scripts import format
from modules.call_queue import wrap_gradio_gpu_call


# Global variables
current_model = None
current_model_name = None

def unload_model():
    global current_model
    global current_model_name
    # Unload the current model
    if current_model is not None:
        current_model = None
        torch.cuda.empty_cache()
        current_model_name = None
        print("Model unloaded successfully.")
        return ['Model unloaded successfully.']
    else:
        print("No model to unload.")
        return ['No model to unload.']

def load_model(model_select):
    global current_model
    global current_model_name
    if current_model_name == model_select:
        return
    if current_model is not None and current_model_name != model_select:
        unload_model()
    print(f"Loading model {model_select}...")
    current_model = pipeline("image-classification", model=f"shadowlilac/{model_select}", device=0, torch_dtype=torch.float16)
    current_model_name = model_select
    print("Loading completed.")

def pipe(model_select, single_image_file):
    load_model(model_select)
    result = current_model(images=[single_image_file])
    score = str(round([p for p in result[0] if p['label'] == 'hq'][0]['score'], 2))
    return [score, '']

def batch_pipe(model_select, batch_size, batch_input_glob, batch_input_recursive, batch_output_dir, batch_output_action_on_conflict, aesthetic_tags_input, aesthetic_thresholds_input, skip_tags_input, batch_remove_duplicated_tag, batch_output_save_json):
    load_model(model_select)
    
    # Split the tags and thresholds
    skip_tags = skip_tags_input.split(',')
    aesthetic_tags = aesthetic_tags_input.split(',')
    aesthetic_thresholds = list(map(float, aesthetic_thresholds_input.split(',')))

    # Check if the lengths of aesthetic_tags and aesthetic_thresholds are equal
    if len(aesthetic_tags) != len(aesthetic_thresholds):
        raise ValueError("The number of aesthetic tags and aesthetic thresholds must be equal.")

    # Batch process
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()

    if batch_input_glob != '':
        # If there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # Get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # Check the input directory path
        if not os.path.isdir(base_dir):
            return ['input path is not a directory']

        # This line is moved here because some reason
        # PIL.Image.registered_extensions() returns only PNG if you call too early
        supported_extensions = [
            e
            for e, f in Image.registered_extensions().items()
            if f in Image.OPEN
        ]

        paths = [
            Path(p)
            for p in glob(batch_input_glob, recursive=batch_input_recursive)
            if '.' + p.split('.').pop().lower() in supported_extensions
        ]

        print(f'found {len(paths)} image(s)')

        image = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            for path in batch:
                # Guess the output path
                base_dir_last = Path(base_dir).parts[-1]
                base_dir_last_idx = path.parts.index(base_dir_last)
                output_dir = Path(batch_output_dir) if batch_output_dir else Path(base_dir)
                output_path = output_dir / Path(*path.parts[base_dir_last_idx + 1:]).with_suffix('.txt')
                image.append(Image.open(path))

            # Perform classification for the batch
            results = current_model(images=image)
            image = []  # Reset the image list for the next batch

            for idx, result in enumerate(results):
                path = batch[idx]
                
                output_dir.mkdir(0o777, parents=True, exist_ok=True)

                # Format output filename
                format_info = format.Info(path, 'txt')

                batch_output_filename_format = '[name].[output_extension]'
                try:
                    formatted_output_filename = format.pattern.sub(
                        lambda m: format.format(m, format_info),
                        batch_output_filename_format
                    )
                except (TypeError, ValueError) as error:
                    return [str(error)]

                output_path = output_dir.joinpath(
                    formatted_output_filename
                )

                output = []
                existing_tags = []

                if output_path.is_file():
                    file_content = output_path.read_text(errors='ignore').strip()


                    existing_tags = set(file_content.split(','))
                    if batch_output_action_on_conflict == 'ignore' or any(tag in existing_tags for tag in skip_tags):
                        print(f'Skipping {path}')
                        continue

                    output.append(file_content)

                # Process the result
                score = round([p for p in result if p['label'] == 'hq'][0]['score'], 2)
                print(f'Prediction: {score} High Quality from {path}')
                split_tags = next((tag for tag, threshold in zip(aesthetic_tags, aesthetic_thresholds) if score >= threshold), aesthetic_tags[-1])
                processed_tags = [split_tags]

                plain_tags = ', '.join(processed_tags)

                if batch_output_action_on_conflict == 'copy':
                    output = [plain_tags]
                elif batch_output_action_on_conflict == 'prepend':
                    output.insert(0, plain_tags)
                else:
                    output.append(plain_tags)

                if batch_remove_duplicated_tag:
                    output_path.write_text(
                        ', '.join(
                            OrderedDict.fromkeys(
                                map(str.strip, ','.join(output).split(','))
                            )
                        ),
                        encoding='utf-8'
                    )
                else:
                    output_path.write_text(
                        ', '.join(output),
                        encoding='utf-8'
                    )

                if batch_output_save_json:
                    output_path.with_suffix('.json').write_text(
                        json.dumps([plain_tags])
                    )

        print('Processing all done :)')

    return ['']

# UI
def add_tab():

    MARKDOWN = \
    """
    # sd-webui-aesthetic-shadow

    [GitHub](https://github.com/gluttony-10/sd-webui-aesthetic-shadow) | [Bilibili](https://space.bilibili.com/893892)

    Used for [AUTOMATIC1111's stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

    Please wait for 30 seconds for initial submission.
    """
        
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row():
            with gr.Column():
                gr.Markdown(MARKDOWN)
            with gr.Column():
                model_select = gr.Dropdown(
                    label='model select',
                    value='aesthetic-shadow-v2',
                    choices=[
                    'aesthetic-shadow-v2',
                    'aesthetic-shadow-v2-strict',
                    'aesthetic-shadow'
                    ]
                )
                unload_model_button = gr.Button(value="Unload Model", variant="secondary")
                info2 = gr.HTML()
                unload_model_button.click(
                    wrap_gradio_gpu_call(unload_model),
                    inputs=[],
                    outputs=[
                        info2
                    ]
                )
        with gr.Row(equal_height=True):
            with gr.Tabs():
                with gr.TabItem(label='Single Image'):
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(type="filepath", label="Input")
                        with gr.Column():
                            run_btn = gr.Button(value="Submit", variant="primary")
                            output_text = gr.Textbox(label="Output", interactive=False)
                            info1 = gr.HTML()
                    for func in [input_image.change, run_btn.click]:
                        func(
                            wrap_gradio_gpu_call(pipe),
                            inputs=[model_select,input_image], 
                            outputs=[output_text,info1]
                        )
                with gr.TabItem(label='Batch from directory'): 
                    with gr.Row():
                        with gr.Column():
                            batch_size = gr.Slider(
                                minimum=1, 
                                maximum=128, 
                                step=1, 
                                label='Batch size',
                                value=8 
                            )
                            batch_input_glob = gr.Textbox(
                                label='Input directory',
                                placeholder='/path/to/images'
                            )
                            batch_input_recursive = gr.Checkbox(
                                label='Use recursive with glob pattern'
                            )
                            batch_output_dir = gr.Textbox(
                                label='Output directory',
                                placeholder='Leave blank to save images to the same path.'
                            )
                            batch_output_action_on_conflict = gr.Dropdown(
                                label='Action on existing caption',
                                value='append',
                                choices=[
                                    'ignore',
                                    'copy',
                                    'append',
                                    'prepend'
                                ]
                            )
                            aesthetic_tags_input = gr.Textbox(
                                label='Aesthetic tags',
                                placeholder='Enter the aesthetic tags, separated by commas',
                                value='very aesthetic,aesthetic,displeasing,very displeasing'
                            )
                            aesthetic_thresholds_input = gr.Textbox(
                                label='Aesthetic thresholds',
                                placeholder='Enter the aesthetic thresholds, separated by commas',
                                value='0.71,0.45,0.27,0.0'
                            )
                            skip_tags_input = gr.Textbox(
                                label='Skip tags',
                                placeholder='Enter the tags to skip, separated by commas'
                            )
                            batch_remove_duplicated_tag = gr.Checkbox(
                                label='Remove duplicated tag'
                            )
                            batch_output_save_json = gr.Checkbox(
                                label='Save with JSON'
                            )
                        with gr.Column():
                            submit = gr.Button(
                                value='Submit',
                                variant='primary'
                            )
                            info = gr.HTML()
                    submit.click(
                        wrap_gradio_gpu_call(batch_pipe),
                        inputs=[
                            model_select,
                            batch_size,
                            batch_input_glob,
                            batch_input_recursive,
                            batch_output_dir,
                            batch_output_action_on_conflict,
                            aesthetic_tags_input,
                            aesthetic_thresholds_input,
                            skip_tags_input,
                            batch_remove_duplicated_tag,
                            batch_output_save_json,
                        ],
                        outputs=[
                            info
                        ]
                    )
    return [(ui, "aesthetic shadow", "aesthetic-shadow")]


script_callbacks.on_ui_tabs(add_tab)