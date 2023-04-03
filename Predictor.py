import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import os
from utils.LeNet5_model import LeNet5


selected_image_path = ""

# Define function to perform the "guess"
def perform_guess(image_path):
    global selected_image_path
    # This is where you would call your function to perform the guess
    # Replace the line below with your actual function call
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28
    img = cv2.resize(img, (28, 28))

    # Convert the image to a PyTorch tensor
    img = torch.from_numpy(np.array([img])).float()

    # Normalize the image
    img /= 255

    # Load the pretrained model
    model = LeNet5()
    model.load_state_dict(torch.load('utils/lenet5_mnist.pt'))

    # Set the model to evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = output.argmax(dim=1).item()

    result_text.set(pred)

def update_number_dropdown(*args):
    global selected_image_path
    digit_folder = digit_combobox.get()
    number_folder = os.path.join("test_images", digit_folder)
    number_files = os.listdir(number_folder)
    number_combobox['values'] = number_files
    selected_image_path = os.path.join(number_folder, number_files[0])
    update_image()

def update_image(*args):
    global selected_image_path
    number_file = number_combobox.get()
    selected_image_path = os.path.join("test_images", digit_combobox.get(), number_file)
    img = Image.open(selected_image_path)
    img = img.resize((200, 200), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    image_viewer.config(image=photo)
    image_viewer.image = photo
    
# Create the main window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")

# Create the left column for the image viewer and dropdown menus
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Add the combobox to select the digit
digit_options = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
digit_combobox = ttk.Combobox(left_frame, values=digit_options, state="readonly")
digit_combobox.set(digit_options[0])
digit_combobox.bind("<<ComboboxSelected>>", update_number_dropdown)
digit_combobox.pack(pady=10)

# Add the combobox to select the number file
number_files = os.listdir(os.path.join("test_images", digit_options[0]))
number_combobox = ttk.Combobox(left_frame, values=number_files, state="readonly")
number_combobox.set(number_files[0])
number_combobox.bind("<<ComboboxSelected>>", update_image)
number_combobox.pack(pady=10)

# Add the image viewer
img = Image.open(os.path.join("test_images", digit_options[0], number_files[0]))
img = img.resize((200, 200), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(img)
image_viewer = tk.Label(left_frame, image=photo)
image_viewer.image = photo
image_viewer.pack(pady=10)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
guess_button = ttk.Button(right_frame, text="Guess", command=lambda: perform_guess(selected_image_path))
guess_button.pack(pady=10)
result_text = tk.StringVar()
result_label = ttk.Label(right_frame, textvariable=result_text, font=("Helvetica", 24))
result_label.pack(pady=10)

update_number_dropdown()
update_image()

root.mainloop()



