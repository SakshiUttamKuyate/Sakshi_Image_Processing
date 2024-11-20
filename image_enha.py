import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Function to display images side by side for comparison
def show_images_side_by_side(original, modified, title_original="Original Image", title_modified="Modified Image"):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    modified_rgb = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_rgb)
    ax[0].set_title(title_original)
    ax[0].axis('off')

    ax[1].imshow(modified_rgb)
    ax[1].set_title(title_modified)
    ax[1].axis('off')

    st.pyplot(fig)

# Function to adjust contrast and brightness
def adjust_contrast_brightness(image, contrast, brightness):
    enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    show_images_side_by_side(image, enhanced_image, "Original Image", "Contrast & Brightness Adjusted")
    return enhanced_image

# Function to smooth the image
def smooth_image(image, kernel_size):
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    show_images_side_by_side(image, smoothed_image, "Original Image", "Smoothed Image")
    return smoothed_image

# Function to sharpen the image
def sharpen_image(image):
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel_sharpening)
    show_images_side_by_side(image, sharpened_image, "Original Image", "Sharpened Image")
    return sharpened_image

# Function to apply a mask
def apply_mask(image, x1, y1, x2, y2):
    height, width = image.shape[:2]
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height or x1 >= x2 or y1 >= y2:
        st.error("Invalid coordinates!")
        return

    mask = np.zeros_like(image)
    mask[y1:y2, x1:x2] = (255, 255, 255)  # White rectangle
    masked_image = cv2.bitwise_and(image, mask)
    show_images_side_by_side(image, masked_image, "Original Image", "Masked Image")
    return masked_image

# Streamlit App
def main():
    st.title("📷 Image Processing Application")

    # Tabs for better organization
    tabs = st.tabs(["Home", "Processing", "About"])

    # Home tab
    with tabs[0]:
        st.header("Welcome to the Image Processing App!")
        st.write("""
        This application allows you to perform various image processing tasks, including:
        - Adjusting contrast and brightness
        - Smoothing the image
        - Sharpening the image
        - Applying a mask to the image
        """)

        st.image("https://placekitten.com/800/400", caption="Image Processing Demo", use_column_width=True)

    # Image Processing tab
    with tabs[1]:
        st.header("Upload and Process Your Image")

        # Upload image
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Convert the uploaded image to OpenCV format
            image = Image.open(uploaded_image)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Sidebar for options
            with st.sidebar:
                st.subheader("⚙️ Processing Options")
                processing_type = st.selectbox("Select Processing Type", 
                                               ["None", "Adjust Contrast & Brightness", "Smooth Image", "Sharpen Image", "Apply Mask"])

            # Perform processing based on user selection
            if processing_type == "Adjust Contrast & Brightness":
                st.subheader("Adjust Contrast & Brightness")
                contrast = st.slider("Contrast", 1.0, 3.0, 1.0)
                brightness = st.slider("Brightness", -100, 100, 0)
                adjust_contrast_brightness(image, contrast, brightness)
            
            elif processing_type == "Smooth Image":
                st.subheader("Smooth Image")
                kernel_size = st.slider("Kernel Size", 3, 11, 5, step=2)
                smooth_image(image, kernel_size)
            
            elif processing_type == "Sharpen Image":
                st.subheader("Sharpen Image")
                sharpen_image(image)
            
            elif processing_type == "Apply Mask":
                st.subheader("Apply Mask")
                x1 = st.number_input("Top-left corner X", 0, image.shape[1] - 1, 0)
                y1 = st.number_input("Top-left corner Y", 0, image.shape[0] - 1, 0)
                x2 = st.number_input("Bottom-right corner X", 0, image.shape[1], image.shape[1])
                y2 = st.number_input("Bottom-right corner Y", 0, image.shape[0], image.shape[0])
                apply_mask(image, x1, y1, x2, y2)

    # About tab
    with tabs[2]:
        st.header("About this App")
        st.write("""
        This app is powered by OpenCV, NumPy, and Streamlit. 
        It's designed to demonstrate basic image processing techniques.
        """)
        st.write("**Developer:** Your Name")
        st.write("**GitHub:** [Your GitHub Link](https://github.com/your-profile)")

if __name__ == "__main__":
    main()
