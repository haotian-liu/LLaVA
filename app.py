import gradio as gr
import predictLLaVa

images = gr.Files(label="Image(s)", file_types=list(".png .jpg .jpeg".split()))
inputText = gr.Textbox(lines=1, label="English Instruction:")
should_translate = gr.Checkbox(label="Translate to German?")

gr.Interface(
    fn=predictLLaVa.process_single_image,
    inputs=[inputText, images, should_translate],
    outputs=gr.Textbox(label="Output", show_copy_button=True),
    title="Fluffyvision",
    description="Generate a text description of an image based on an instruction.",
    allow_flagging="never"
).launch(share=True, auth=predictLLaVa.check_auth, auth_message="Please enter your username and password.")