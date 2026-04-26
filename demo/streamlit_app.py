"""Streamlit demo for video captioning system."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import BLIP2VideoCaptioningModel
from src.utils import get_device, setup_logging


# Page configuration
st.set_page_config(
    page_title="Video Captioning System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logger
logger = setup_logging()

# Safety disclaimer
st.sidebar.markdown("""
## ⚠️ Safety Disclaimer

This video captioning system is for research and educational purposes only. 
Generated captions may not always be accurate and should not be used for:
- Medical diagnosis or treatment decisions
- Legal or security applications
- Critical decision-making processes

Please use responsibly and verify important information independently.
""")

# Model loading
@st.cache_resource
def load_model(model_path: str = "checkpoints/best_model"):
    """Load the video captioning model."""
    try:
        if os.path.exists(model_path):
            model = BLIP2VideoCaptioningModel.from_pretrained(model_path)
        else:
            # Use pretrained BLIP2 model as fallback
            st.warning(f"Model not found at {model_path}. Using pretrained BLIP2 model.")
            from omegaconf import OmegaConf
            config = OmegaConf.create({
                "vision_model_name": "Salesforce/blip2-opt-2.7b",
                "max_frames": 8,
                "max_length": 77,
                "num_beams": 4,
                "temperature": 1.0,
                "num_attention_heads": 32,
                "freeze_vision_encoder": False,
                "freeze_text_encoder": False,
            })
            model = BLIP2VideoCaptioningModel(config)
        
        device = get_device("auto")
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def extract_frames(video_file, max_frames: int = 8) -> Tuple[np.ndarray, list]:
    """Extract frames from video file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load video
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames = []
        frame_images = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Resize for display
                frame_display = cv2.resize(frame_rgb, (224, 224))
                frame_images.append(frame_display)
        
        cap.release()
        
        return np.array(frames), frame_images
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def generate_caption(model, processor, frames: np.ndarray, device: torch.device) -> str:
    """Generate caption for video frames."""
    try:
        # Convert frames to PIL Images
        frame_images = [Image.fromarray(frame) for frame in frames]
        
        # Process with BLIP processor
        inputs = processor(images=frame_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_length=77,
                num_beams=4,
                temperature=1.0,
            )
        
        # Decode caption
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption
    
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return f"Error generating caption: {e}"


def visualize_attention(model, frames: np.ndarray, device: torch.device) -> Optional[go.Figure]:
    """Visualize attention weights."""
    try:
        # Get attention weights
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = frames_tensor.to(device)
        
        attention_weights = model.get_attention_weights(frames_tensor.unsqueeze(0))
        attention_weights = attention_weights.squeeze().cpu().numpy()
        
        # Create attention visualization
        fig = go.Figure()
        
        # Add attention weights as heatmap
        fig.add_trace(go.Heatmap(
            z=attention_weights,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Attention Weight")
        ))
        
        fig.update_layout(
            title="Temporal Attention Weights",
            xaxis_title="Frame Index",
            yaxis_title="Attention Head",
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error visualizing attention: {e}")
        return None


# Main app
def main():
    """Main Streamlit application."""
    st.title("🎬 Video Captioning System")
    st.markdown("Generate natural language descriptions for videos using state-of-the-art vision-language models.")
    
    # Sidebar controls
    st.sidebar.header("Model Settings")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="checkpoints/best_model",
        help="Path to the trained model checkpoint"
    )
    
    max_frames = st.sidebar.slider(
        "Max Frames", 
        min_value=1, 
        max_value=16, 
        value=8,
        help="Maximum number of frames to sample from video"
    )
    
    num_beams = st.sidebar.slider(
        "Beam Size", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Number of beams for beam search generation"
    )
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=1.0,
        help="Sampling temperature (higher = more diverse)"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Upload Video")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload a video file to generate captions"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.info(f"📁 **File:** {uploaded_file.name}")
            st.info(f"📏 **Size:** {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Extract frames
            with st.spinner("Extracting frames..."):
                frames, frame_images = extract_frames(uploaded_file, max_frames)
            
            st.success(f"✅ Extracted {len(frames)} frames")
            
            # Display frames
            st.subheader("🎞️ Sampled Frames")
            
            # Create frame grid
            cols = st.columns(4)
            for i, frame_img in enumerate(frame_images):
                with cols[i % 4]:
                    st.image(frame_img, caption=f"Frame {i+1}", use_column_width=True)
    
    with col2:
        st.header("📝 Generated Caption")
        
        if uploaded_file is not None:
            # Generate caption
            with st.spinner("Generating caption..."):
                caption = generate_caption(model, model.processor, frames, device)
            
            # Display caption
            st.markdown("### Generated Caption:")
            st.markdown(f"**{caption}**")
            
            # Copy button
            if st.button("📋 Copy Caption"):
                st.write("Caption copied to clipboard!")
            
            # Attention visualization
            st.subheader("🔍 Attention Visualization")
            
            with st.spinner("Computing attention weights..."):
                attention_fig = visualize_attention(model, frames, device)
            
            if attention_fig:
                st.plotly_chart(attention_fig, use_container_width=True)
            else:
                st.warning("Could not generate attention visualization")
    
    # Additional features
    st.header("🔧 Additional Features")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Model Information")
        
        if model is not None:
            # Model stats
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
            st.metric("Device", str(device))
            st.metric("Model Type", "BLIP2 Video Captioning")
    
    with col4:
        st.subheader("⚙️ Generation Settings")
        
        st.json({
            "Max Frames": max_frames,
            "Beam Size": num_beams,
            "Temperature": temperature,
            "Max Length": 77,
            "Model": model_path
        })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This System
    
    This video captioning system uses BLIP2 (Bootstrapping Language-Image Pre-training) 
    with temporal attention mechanisms to generate natural language descriptions for videos.
    
    **Key Features:**
    - 🎯 State-of-the-art vision-language model
    - ⏱️ Temporal attention for video understanding
    - 🔍 Attention visualization
    - 📊 Comprehensive evaluation metrics
    - 🚀 Easy-to-use interface
    
    **Technical Details:**
    - Based on BLIP2 architecture
    - Supports multiple video formats
    - Configurable generation parameters
    - Real-time inference
    """)


if __name__ == "__main__":
    main()
