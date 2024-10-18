import streamlit as st
import numpy as np
from PIL import Image, ImageSequence
from ultralytics import YOLO
import tempfile

# Load the YOLO model
ModelPath = r"./models/train2bst.pt"
model = YOLO(ModelPath)

# Streamlit configuration
st.set_page_config(
    page_title="Sentiment Detection",
    page_icon=":smiley:",
)

# Add custom CSS
custom_css = """
<style>
body {
    background-color: white;
    color: #333;
    font-family: 'Arial', sans-serif;
}

.ea3mdgi8{
background-color: white;
}

h1 {
    text-align: center;
    color: #007BFF;   
}

#457c40bd{
outline: 2px solid black;}

h2 {
    color: #007BFF;
}

.ef3psqc16 {
    background-color: white;
    color: black;
}

.ef3psqc16:hover {
    background-color: blue;  /* Darker shade for hover effect */
    filter: drop-shadow(0 0 10px #fffdef);
}

.download-button {
    background-color: #28a745;
    color: white;
}


.image-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

.image-container img {
    margin: 10px;
    border: 2px solid #007BFF;
    border-radius: 5px;
}

.stAppHeader {
background-color: grey;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Initialize session state variables
if 'detected_frames' not in st.session_state:
    st.session_state.detected_frames = []
if 'output_gif_path' not in st.session_state:
    st.session_state.output_gif_path = None
if 'output_image' not in st.session_state:
    st.session_state.output_image = None

st.title("Sentiment Detection 😀😢😠")

# File uploader
uploaded_files = st.file_uploader("Choose a GIF or Image file", type=["gif", "jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Add a button to proceed with detection
    if st.button("Proceed to Detect"):
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == "gif":
                # Open the GIF
                gif_image = Image.open(uploaded_file)

                # Extract frames from GIF
                gif_frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif_image)]

                # Process each frame with YOLO
                detected_frames = []
                for frame in gif_frames:
                    frame_np = np.array(frame)

                    # YOLO detection
                    result = model.predict(frame_np)

                    # Draw detections on the frame
                    if result:
                        detected_frame = result[0].plot()
                        detected_frames.append(Image.fromarray(detected_frame))
                    else:
                        detected_frames.append(frame)  # If no detections, use original frame

                # Save the detected frames as a new GIF
                output_gif_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gif").name
                detected_frames[0].save(output_gif_path, save_all=True, append_images=detected_frames[1:], loop=0, duration=gif_image.info['duration'])
                st.session_state.output_gif_path = output_gif_path

            elif file_extension in ["jpg", "jpeg", "png"]:
                # Open the image
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)

                # YOLO detection
                result = model.predict(image_np)

                # Draw detections on the image
                if result:
                    output_image = Image.fromarray(result[0].plot())
                else:
                    output_image = image  # If no detections, use original image
                st.session_state.output_image = output_image

        # Display the processed GIF or image
        if st.session_state.output_gif_path:
            st.subheader("Processed GIF with Detection:")
            st.image(st.session_state.output_gif_path)
            # Provide a download button for the output GIF
            with open(st.session_state.output_gif_path, "rb") as file:
                st.download_button(
                    label="Download Processed GIF",
                    data=file,
                    file_name="detected_output.gif",
                    mime="image/gif",
                    key="download_gif"
                )
        
        if st.session_state.output_image is not None:
            st.subheader("Processed Image with Detection:")
            st.image(st.session_state.output_image, caption="Detected Image")
            # Provide a download button for the output image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                st.session_state.output_image.save(temp_image.name)
                with open(temp_image.name, "rb") as file:
                    st.download_button(
                        label="Download Processed Image",
                        data=file,
                        file_name="detected_output.png",
                        mime="image/png",
                        key="download_image"
                    )

# Clear Screen button
if st.button("Clear Screen"):
    st.session_state.detected_frames = []
    st.session_state.output_gif_path = None
    st.session_state.output_image = None
    # Clear the displayed images or GIFs
    
