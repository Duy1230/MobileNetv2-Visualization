import streamlit as st
from app import *
import matplotlib.pyplot as plt
import torch

# Set the page configuration
st.set_page_config(
    page_title="📊 CNN Visualization Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header Section
st.markdown("# 📊 Convolutional Neural Network (CNN) Visualization")
st.markdown("""
Welcome to the **CNN Visualization Tool**! This application allows you to:
- **Upload an image** of your choice.
- **Select a CNN layer** to visualize its filters and output feature maps.
- **Understand** how different layers of a CNN process and interpret images.
""")

# Sidebar for user inputs
st.sidebar.header("Upload and Selection")


# def read_image(path):
#     image = Image.open(path).convert('RGB')
#     return image


def show_image(image):
    st.image(image, caption='Uploaded Image', use_column_width=True)


def load_model():
    return load_mobileNet()


def visualize_filters(conv_layer):
    filters = conv_layer.weight.data.clone()
    filters = normalize_filter(filters)
    n_filters = filters.shape[0]
    n_cols = 8
    n_rows = n_filters // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n_filters):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        filter = filters[i].permute(1, 2, 0).numpy()
        ax.imshow(filter, cmap='gray')
        ax.axis('off')
    # Remove empty subplots
    for i in range(n_filters, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])
    st.pyplot(fig)


def visualize_outputs(model, conv_layer_name, input_image):

    input_tensor = preprocess_image(input_image)

    intermediate_outputs, pred = get_multiple_intermediate_outputs(
        model, input_tensor, [conv_layer_name])

    if intermediate_outputs:
        output = intermediate_outputs[conv_layer_name].squeeze(0)
        n_features = output.shape[0]
        n_cols = 8
        n_rows = n_features // n_cols + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            feature_map = output[i].cpu().numpy()
            ax.imshow(feature_map, cmap='gray')
            ax.axis('off')
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])
        st.pyplot(fig)
        return torch.nn.functional.softmax(pred[0], dim=0)
    else:
        st.write("No output captured for this layer.")


def show_prediction(prediction):
    with open("imagenet_classes.txt", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    top_5_preds = torch.topk(prediction, 5)
    top_5_preds = [(classes[top_5_preds[1][i]], top_5_preds[0][i].item())
                   for i in range(top_5_preds[0].size(0))]
    return top_5_preds


st.sidebar.header("🔧 Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "📂 **Upload an Image**", type=["png", "jpg", "jpeg"]
)

selected_layer = st.sidebar.selectbox(
    "🖥️ **Select a CNN Layer to Visualize**", [
        "features.0.0",
        "features.2.conv.1.0",
        "features.3.conv.1.0",
        "features.4.conv.1.0",
        "features.5.conv.1.0",
        "features.6.conv.1.0",
        "features.7.conv.1.0",
        "features.8.conv.1.0",
        "features.9.conv.1.0",
        "features.10.conv.1.0",
        "features.11.conv.1.0",
        "features.12.conv.1.0",
        "features.13.conv.1.0",
        "features.14.conv.1.0",
        "features.15.conv.1.0",
        "features.16.conv.1.0",
        "features.17.conv.1.0",
        "features.18.0",
    ]
)

# Main Content
if uploaded_file is not None:
    # Display the uploaded image
    image = read_image(uploaded_file)
    show_image(image)

    # Load the CNN model
    model = load_model()

    # Retrieve the selected convolutional layer
    conv_layer = get_layer_by_name(model, selected_layer)

    # Section Header
    st.header(f"🔍 **Visualization for Layer: `{selected_layer}`**")

    # Filters Visualization
    st.subheader("🎨 **Filters Visualization**")
    if not "18" in selected_layer:
        visualize_filters(conv_layer)
    else:
        st.info("This layer does not contain convolutional filters to display.")

    # Output Feature Maps Visualization
    st.subheader("🗺️ **Output Feature Maps**")
    with st.spinner("Generating feature maps..."):
        prediction = visualize_outputs(model, selected_layer, image)

    # Display Predictions
    st.subheader("🏆 **Top 5 Predictions**")
    top_5_predictions = show_prediction(prediction)
    st.table(data=top_5_predictions)

    # Additional Information
    st.markdown("""
    ---
    **About This Tool:**
    - **Filters Visualization:** Shows the patterns that the CNN layer is looking for in the input image.
    - **Output Feature Maps:** Visualizes the activation maps showing where and how strongly the filters are responding.
    - **Top 5 Predictions:** Displays the most probable classifications based on the input image.

    **Made with ❤️ using Streamlit and PyTorch**
    """)
else:
    st.write("🛑 **Please upload an image to get started.**")
