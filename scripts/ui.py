from modules import script_callbacks
import gradio as gr

import torch
import os
from transformers import pipeline

def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Row(equal_height=True):
            





















script_callbacks.on_ui_tabs(add_tab)