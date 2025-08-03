import gradio as gr
import torch
import traceback
import sys
import os
import signal

print("Starting application...")

# Add signal handler for graceful shutdown
def signal_handler(signum, frame):
    print("Received interrupt signal, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from model import load_model, load_tokenizer
    print("Imported model functions successfully")
    
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully")
    
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    print("Tokenizer loaded successfully")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)

def generate_response(prompt, history=[]):
    try:
        if not prompt.strip():
            return history, history
            
        print(f"Generating response for: {prompt}")
        input_ids = torch.tensor([tokenizer.encode(prompt, out_type=int)]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=0,
                eos_token_id=1
            )
        
        response = tokenizer.decode(output_ids[0].tolist())
        print(f"Generated response: {response}")
        
        history.append((prompt, response))
        return history, history
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        history.append((prompt, f"Error: {error_msg}"))
        return history, history

with gr.Blocks(title="MinAI Proof of Concept Alpha Model", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🇲🇲 MinAI Proof of Concept Alpha Model")
    
    gr.Markdown("""
    **⚠️ Important Notice:**
    This is a **proof of concept** and **alpha version** model. It is not expected to be useful or produce good quality outputs. 
    The model is in early development stage and serves as a demonstration of the technical approach rather than a production-ready solution.
    
    **About this model:**
    - Experimental Burmese language model
    - Limited training data and capabilities
    - Responses may be inaccurate, incomplete, or nonsensical
    - Use for research and testing purposes only
    """)
    
    gr.Markdown("### Quick Test Examples - Click to try:")
    
    with gr.Row():
        example_buttons = []
        examples = [
            "မင်္ဂလာပါ",  # Hello
            "နေကောင်းလား",  # How are you?
            "ကျေးဇူးတင်ပါတယ်",  # Thank you
            "နာမည်ဘာလဲ",  # What's your name?
            "ရန်ကုန်မြို့"  # Yangon city
        ]
        
        for example in examples:
            btn = gr.Button(example, scale=1, size="sm")
            example_buttons.append(btn)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=400,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type a message in Burmese...",
                    container=False,
                    scale=4
                )
                send = gr.Button("Send", scale=1, variant="primary")
                clear = gr.Button("Clear", scale=1)

    def user_input(message, chat_history):
        try:
            if not message.strip():
                return message, chat_history
            return "", generate_response(message, chat_history)[0]
        except Exception as e:
            print(f"Error in user_input: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            error_history = chat_history + [(message, f"Error: {str(e)}")]
            return "", error_history

    # Event handlers
    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=True)
    send.click(user_input, [msg, chatbot], [msg, chatbot], queue=True)
    clear.click(lambda: [], None, chatbot, queue=False)
    
    # Example button handlers
    def set_example(example_text):
        return example_text
    
    for i, btn in enumerate(example_buttons):
        btn.click(
            fn=lambda example=examples[i]: example,
            inputs=None,
            outputs=msg,
            queue=False
        )

if __name__ == "__main__":
    try:
        print("Launching Gradio interface...")
        demo.launch(
            share=False, 
            debug=True,
            server_name="127.0.0.1",  # Explicit localhost
            server_port=7860,         # Explicit port
            inbrowser=False,          # Don't auto-open browser
            quiet=False               # Show startup messages
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Try alternative launch configuration
        try:
            print("Trying alternative launch configuration...")
            demo.launch(
                share=False,
                debug=False,
                server_name="0.0.0.0",
                server_port=7861,
                inbrowser=False
            )
        except Exception as e2:
            print(f"Alternative launch also failed: {e2}")
            print("Please check if port 7860 or 7861 is already in use")
