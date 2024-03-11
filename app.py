import cv2
import gradio as gr
from infer import Model


def hex_to_rgb(hex_color, div=255):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / div
    g = int(hex_color[2:4], 16) / div
    b = int(hex_color[4:6], 16) / div

    return r, g, b


def draw_circle(image, color, x, y, radius, thickness):

    circle_color = hex_to_rgb(color, 1)

    overlay = image.copy()
    overlay = cv2.circle(overlay, (x, y), radius, circle_color, -1)
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    image = cv2.circle(image, (x, y), radius, circle_color, thickness)

    return image


model = Model()

with gr.Blocks() as app:

    with gr.Row():

        original_image = gr.Image(visible=False, type="numpy")
        input_image = gr.Image(label="Input Image",
                               sources="upload",
                               type="numpy",
                               interactive=True)
        with gr.Tab("Output Image"):
            output_image = gr.Image(type="numpy", interactive=False)
        with gr.Tab("Output Depth Image"):
            depth_image = gr.Image(type="numpy", interactive=False)
    with gr.Column():
        pos_x = gr.Slider(0.0,
                          1.0,
                          value=0.5,
                          step=0.1,
                          label="X",
                          interactive=True)
        pos_y = gr.Slider(0.0,
                          1.0,
                          value=0.5,
                          step=0.1,
                          label="Y",
                          interactive=True)
        pos_z = gr.Slider(-1.0,
                          1.0,
                          value=-0.7,
                          step=0.05,
                          label="Z",
                          interactive=True)
        light = gr.Slider(0.0,
                          2.0,
                          value=0.7,
                          step=0.1,
                          label="Light Intensity",
                          interactive=True)
        color = gr.ColorPicker(value="#FFFFFF",
                               label="Light Color",
                               interactive=True)

    def on_upload_image(image, x, y, z, power, color):
        r, g, b = hex_to_rgb(color)
        x_ = (x - 0.5) * 2
        y_ = (-y + 0.5) * 2

        d_img, o_img = model.inference_image(image, x_, y_, z, r, g, b, power)
        return d_img, o_img, image

    input_image.upload(on_upload_image,
                       [input_image, pos_x, pos_y, pos_z, light, color],
                       [depth_image, output_image, original_image])

    def on_change_image(image, x, y, color):
        x_ = int(x * image.shape[1])
        y_ = int(y * image.shape[0])

        r = int(max(min(image.shape[0], image.shape[1]) / 10, 10))

        return draw_circle(image, color, x_, y_, r, r // 10)

    original_image.change(on_change_image,
                          [original_image, pos_x, pos_y, color], [input_image])

    def get_point(image, color, z, power, evt: gr.SelectData):
        x_ = evt.index[0]
        y_ = evt.index[1]
        x = (x_ / image.shape[1])
        y = (y_ / image.shape[0])

        r = int(max(min(image.shape[0], image.shape[1]) / 10, 10))

        input_image = draw_circle(image, color, x_, y_, r, r // 10)
        output_image = on_change_parameter(x, y, z, power, color)
        return output_image, input_image, x, y

    input_image.select(
        get_point,
        [original_image, color, pos_z, light],
        [output_image, input_image, pos_x, pos_y],
        show_progress="hidden",
    )

    def on_change_parameter_and_update_image(image, x, y, z, power, color):

        x_ = int(image.shape[1] * x)
        y_ = int(image.shape[0] * y)
        r = int(max(min(image.shape[0], image.shape[1]) / 10, 10))

        input_image = draw_circle(image, color, x_, y_, r, r // 10)
        output_image = on_change_parameter(x, y, z, power, color)

        return input_image, output_image

    def on_change_parameter(x, y, z, power, color):
        r, g, b = hex_to_rgb(color)
        xx = (x - 0.5) * 2
        yy = (-y + 0.5) * 2

        return model.add_light(xx, yy, z, r, g, b, power)

    pos_x.input(on_change_parameter_and_update_image,
                [original_image, pos_x, pos_y, pos_z, light, color],
                [input_image, output_image],
                show_progress="hidden",
                trigger_mode="always_last")
    pos_y.input(on_change_parameter_and_update_image,
                [original_image, pos_x, pos_y, pos_z, light, color],
                [input_image, output_image],
                show_progress="hidden",
                trigger_mode="always_last")
    pos_z.input(on_change_parameter, [pos_x, pos_y, pos_z, light, color],
                [output_image],
                show_progress="hidden",
                trigger_mode="always_last")
    light.input(on_change_parameter, [pos_x, pos_y, pos_z, light, color],
                [output_image],
                show_progress="hidden",
                trigger_mode="always_last")
    color.input(on_change_parameter, [pos_x, pos_y, pos_z, light, color],
                [output_image],
                show_progress="hidden",
                trigger_mode="always_last")

app.queue().launch()
