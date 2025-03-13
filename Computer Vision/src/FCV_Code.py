import cv2
import numpy as np
import streamlit as st
import glob
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Image Contrast Enhancement", page_icon="✨", layout="centered")

# ---- Sidebar for Gamma Value and Image Upload ----
st.sidebar.title("🔧 Settings")
# Input for gamma value
gamma_value = st.sidebar.slider("Gamma Value", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
st.sidebar.write("Adjust the gamma value to control brightness.")

# Image upload
st.sidebar.write("Upload a new image to see the enhancement.")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ---- Check and load input image ----
if uploaded_file is not None:
    # Read the uploaded image file as an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
else:
    st.error("Please upload an image to proceed.")
    st.stop()

# ---- Define Image Processing Methods ----
# Method 1: CLAHE
def apply_clahe(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((clahe_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# Method 2: Linear Contrast Stretching
def linear_contrast_stretch(image):
    stretched_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return stretched_image

# Method 3: Luminance Adjustment using Template Matching with SIFT
def get_best_match_image(input_image):
    dataset_folder = '/Users/virajedlabadkar/Downloads/FCV Lab/flowers_dataset/*.jpg'
    well_lit_image_paths = glob.glob(dataset_folder)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Compute keypoints and descriptors for the input image
    input_keypoints, input_descriptors = sift.detectAndCompute(input_image, None)
    
    best_match_image = None
    max_matches = 0  # Track the maximum number of matches
    
    # Loop through all images in the dataset to find the best match
    for image_path in well_lit_image_paths:
        well_lit_image = cv2.imread(image_path)
        if well_lit_image is not None:
            well_lit_keypoints, well_lit_descriptors = sift.detectAndCompute(well_lit_image, None)
            if well_lit_descriptors is not None:
                matches = bf.match(input_descriptors, well_lit_descriptors)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    best_match_image = well_lit_image
                    
    return best_match_image if best_match_image is not None else input_image

best_match_image = get_best_match_image(input_image)

# Method 4: Luminance Adjustment
def adjust_luminance(input_img, reference_img):
    input_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    reference_lab = cv2.cvtColor(reference_img, cv2.COLOR_BGR2LAB)
    input_l, input_a, input_b = cv2.split(input_lab)
    ref_l, ref_a, ref_b = cv2.split(reference_lab)
    input_mean, input_std = input_l.mean(), input_l.std()
    ref_mean, ref_std = ref_l.mean(), ref_l.std()
    adjusted_l = (input_l - input_mean) * (ref_std / input_std) + ref_mean
    adjusted_l = np.clip(adjusted_l, 0, 255).astype(np.uint8)
    adjusted_lab = cv2.merge((adjusted_l, input_a, input_b))
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

luminance_adjusted_image = adjust_luminance(input_image, best_match_image)

# Method 5: Gamma Correction
def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma_corrected_image = apply_gamma_correction(input_image, gamma_value)

# ---- Display Images in Streamlit ----
st.title("✨ Image Contrast Enhancement Comparative Analysis ✨")
st.write("Compare different image enhancement methods side-by-side:")

# Display images in rows of two columns
# Row 1: Original and CLAHE
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

with col2:
    st.subheader("CLAHE Enhanced Image")
    clahe_image = apply_clahe(input_image)
    st.image(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

# Row 2: Linear Contrast Stretching and Luminance Adjustment
col3, col4 = st.columns(2)
with col3:
    st.subheader("Linear Contrast Stretched Image")
    stretched_image = linear_contrast_stretch(input_image)
    st.image(cv2.cvtColor(stretched_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

with col4:
    st.subheader("Luminance Adjusted Image")
    st.image(cv2.cvtColor(luminance_adjusted_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

# Row 3: Gamma Correction and Gamma Curve Plot
col5, col6 = st.columns(2)
with col5:
    st.subheader("Gamma Corrected Image")
    st.write(f"Gamma: {gamma_value}")
    st.image(cv2.cvtColor(gamma_corrected_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

with col6:
    st.subheader("Gamma Curve")
    # Plot the gamma curve
    x = np.arange(256)
    y = np.array([((i / 255.0) ** (1.0 / gamma_value)) * 255 for i in x])
    fig, ax = plt.subplots()
    ax.plot(x, y, color="blue")
    ax.set_title("Output Intensity vs Input Intensity")
    ax.set_xlabel("Input Intensity")
    ax.set_ylabel("Output Intensity")
    ax.grid(True)
    st.pyplot(fig)
