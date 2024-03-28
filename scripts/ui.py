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


def pipe(model_select, single_image_file):
    print(f"Loading model {model_select}...")
    pipe = pipeline("image-classification", model=f"shadowlilac/{model_select}", device=0, torch_dtype=torch.float16)
    print("Loading completed.")
    result = pipe(images=[single_image_file])
    score = str(round([p for p in result[0] if p['label'] == 'hq'][0]['score'], 2))
    return [score, '']


def batch_pipe(model_select, batch_size, batch_input_glob, batch_input_recursive, batch_output_dir, batch_output_action_on_conflict, aesthetic_tags_input, aesthetic_thresholds_input, batch_remove_duplicated_tag, batch_output_save_json):
    print(f"Loading model {model_select}...")
    pipe = pipeline("image-classification", model=f"shadowlilac/{model_select}", device=0, torch_dtype=torch.float16)
    print("Loading completed.")
    
    aesthetic_tags_input = str(aesthetic_tags_input)
    aesthetic_thresholds_input = str(aesthetic_thresholds_input)
    aesthetic_tags = aesthetic_tags_input.split(',')
    aesthetic_thresholds = list(map(float, aesthetic_thresholds_input.split(',')))

    # batch process
    batch_input_glob = batch_input_glob.strip()
    batch_output_dir = batch_output_dir.strip()

    if batch_input_glob != '':
        # if there is no glob pattern, insert it automatically
        if not batch_input_glob.endswith('*'):
            if not batch_input_glob.endswith(os.sep):
                batch_input_glob += os.sep
            batch_input_glob += '*'

        # get root directory of input glob pattern
        base_dir = batch_input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # check the input directory path
        if not os.path.isdir(base_dir):
            return ['input path is not a directory']

        # this line is moved here because some reason
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

        for i in range(0, len(paths), batch_size):
            image = []
            batch = paths[i:i + batch_size]
            for path in batch:
                image.append(Image.open(path))
            # Perform classification for the batch
            results = pipe(images=image)

            for idx, result in enumerate(results):
                path = batch[idx]
                # guess the output path
                base_dir_last = Path(base_dir).parts[-1]
                base_dir_last_idx = path.parts.index(base_dir_last)
                output_dir = Path(
                    batch_output_dir) if batch_output_dir else Path(base_dir)
                output_dir = output_dir.joinpath(
                    *path.parts[base_dir_last_idx + 1:]).parent

                output_dir.mkdir(0o777, True, True)

                # format output filename
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

                if output_path.is_file():
                    output.append(output_path.read_text(errors='ignore').strip())

                    if batch_output_action_on_conflict == 'ignore':
                        print(f'skipping {path}')
                        continue

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

# ui
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
            gr.Markdown(MARKDOWN)
            model_select = gr.Dropdown(
                                label='model select',
                                value='aesthetic-shadow-v2',
                                choices=[
                                    'aesthetic-shadow-v2',
                                    'aesthetic-shadow-v2-strict',
                                    'aesthetic-shadow'
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
                            batch_remove_duplicated_tag,
                            batch_output_save_json,
                        ],
                        outputs=[
                            info
                        ]
                    )
    return [(ui, "aesthetic shadow", "aesthetic-shadow")]


script_callbacks.on_ui_tabs(add_tab)