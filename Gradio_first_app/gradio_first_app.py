
import os
import gradio as gr


def calculate(num1, num2, operation):
    try:
        if operation == "Add (+)":
            return num1 + num2
        elif operation == "Subtract (−)":
            return num1 - num2
        elif operation == "Multiply (×)":
            return num1 * num2
        elif operation == "Divide (÷)":
            if num2 == 0:
                return "Error: division by zero"
            return num1 / num2
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Simple Calculator") as demo:
    gr.Markdown("# 🧮 Simple Calculator")
    gr.Markdown("Four basic arithmetic operations")

    with gr.Row():
        num1 = gr.Number(label="First number", value=0)
        num2 = gr.Number(label="Second number", value=0)

    operation = gr.Radio(
        choices=[
            "Add (+)",
            "Subtract (−)",
            "Multiply (×)",
            "Divide (÷)",
        ],
        value="Add (+)",
        label="Operation",
    )

    btn = gr.Button("Calculate", variant="primary")
    result = gr.Textbox(label="Result", interactive=False)

    btn.click(
        fn=calculate,
        inputs=[num1, num2, operation],
        outputs=result,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
