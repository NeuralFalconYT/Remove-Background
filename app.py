import gradio as gr
from gradio_imageslider import ImageSlider 
from helper import create_transparent_foreground,remove_background_batch_images,remove_background_from_video
from soft_foreground_segmenter import SoftForegroundSegmenter
foreground_model = "foreground-segmentation-model-vitl16_384.onnx"
foreground_segmenter = SoftForegroundSegmenter(onnx_model=foreground_model)

def process_image(image_path):
    original, transparent, output_image_path = create_transparent_foreground(image_path,foreground_segmenter)
    return (original, transparent), output_image_path  

def ui1():
  with gr.Blocks() as demo:
      gr.Markdown("## 🪄 Background Remove From Image")

      with gr.Row():
        with gr.Column():
          image_input = gr.Image(type="filepath", label="Upload Image")
          btn = gr.Button("Remove Background")
        with gr.Column():
          image_slider = ImageSlider(label="Before vs After",position=0.5)
          save_path_box = gr.File(label="Download Transparent Image")

      btn.click(
          fn=process_image,
          inputs=image_input,
          outputs=[image_slider, save_path_box]
      )
      gr.Examples(
        examples=[["./assets/cat.png"],["./assets/girl.jpg"],["./assets/dog.jpg"]],
        inputs=[image_input],
        outputs=[image_slider, save_path_box],
        fn=process_image,
        cache_examples=True,
    )
      
  return demo


def process_uploaded_images(uploaded_images):
    return remove_background_batch_images(uploaded_images,foreground_segmenter)
def ui2():
  with gr.Blocks() as demo:
      gr.Markdown("## 🪄 Background Remover From Bulk Images")
      with gr.Row():
          with gr.Column():
              image_input = gr.File(file_types=["image"], file_count="multiple", label="Upload Multiple Images")
              submit_btn = gr.Button("Remove Backgrounds")
          with gr.Column():
              zip_output = gr.File(label="Download ZIP")

      submit_btn.click(fn=process_uploaded_images, inputs=image_input, outputs=zip_output)
  return demo  



def process_video(video_file):
    output_path = remove_background_from_video(video_file, foreground_segmenter)
    return output_path  # should be absolute or relative path to processed video

def ui3():
  # --- Gradio Interface ---
  with gr.Blocks() as demo:
      gr.Markdown("## 🎥 Remove Background From Video")

      with gr.Row():
          with gr.Column():
              input_video = gr.Video(label="Upload Video (.mp4)")
              run_btn = gr.Button("Remove Background")
          with gr.Column():
              output_video = gr.Video(label="Green Screen Video")

      run_btn.click(fn=process_video, inputs=input_video, outputs=output_video)
    #   gr.Examples(
    #     examples=[["./assets/video.mp4"]],
    #     inputs=[input_video],
    #     outputs=[output_video],
    #     fn=process_video,
    #     cache_examples=True,
    # )
  return demo
demo1=ui1()
demo2=ui2()
demo3=ui3()
demo = gr.TabbedInterface([demo1, demo2,demo3],["Background Remove From Image","Background Remover From Bulk Images","Remove Background From Video"],title="Microsoft DAViD Background Remove")
demo.queue().launch(debug=True, share=True)
