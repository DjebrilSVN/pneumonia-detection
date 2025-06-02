import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import model_loader
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme & CSS ---
st.markdown(
    """
    <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Overall styling */
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main background */
        .main .block-container {
            background-color: #f0f4f8;
            padding: 2rem;
            border-radius: 20px;
        }
        
        /* Sidebar styling */
        .css-1d391kg, [data-testid="stSidebar"] {
            background-color: #1e3a8a;
            color: #FFFFFF;
            padding: 20px;
            border-radius: 0 20px 20px 0;
        }
        
        /* Sidebar title */
        [data-testid="stSidebar"] .sidebar-content h1 {
            color: white;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] .sidebar-content p {
            color: #e2e8f0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Sidebar card */
        [data-testid="stSidebar"] .card {
            background-color: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        
        [data-testid="stSidebar"] .card h3 {
            color: #ffffff;
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        
        [data-testid="stSidebar"] .card p {
            color: #e2e8f0;
            font-size: 0.9rem;
        }
        
        /* Page title */
        h1 {
            font-size: 2.5rem !important;
            color: #1e3a8a;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            font-size: 1.8rem !important;
            color: #1e3a8a;
            font-weight: 600;
        }
        
        h3 {
            font-size: 1.4rem !important;
            color: #1e3a8a;
            font-weight: 500;
        }
        
        /* Card containers - Neumorphic style */
        .card {
            background-color: #f0f4f8;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 8px 8px 16px #d1d9e6, -8px -8px 16px #ffffff;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 10px 10px 20px #d1d9e6, -10px -10px 20px #ffffff;
            transform: translateY(-5px);
        }
        
        /* Bordered info box - Neumorphic style */
        .bordered {
            background-color: #f0f4f8;
            border-radius: 20px;
            padding: 20px;
            box-shadow: inset 4px 4px 8px #d1d9e6, inset -4px -4px 8px #ffffff;
        }
        
        .bordered .section {
            margin-bottom: 16px;
        }
        
        .bordered .section h4 {
            margin: 0;
            font-size: 1.1rem;
            color: #1e3a8a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 6px;
            font-weight: 500;
        }
        
        .bordered .section .value {
            font-size: 1.8rem !important;
            color: #1e293b;
            margin-top: 8px;
            font-weight: 600;
        }
        
        /* Uploader area */
        [data-testid="stFileUploader"] {
            background-color: #f0f4f8;
            border: 2px dashed #94a3b8;
            border-radius: 12px;
            padding: 16px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #1e3a8a;
            background-color: #f8fafc;
        }
        
        /* Heatmap sizing */
        .heatmap-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            max-width: 500px;
            margin: 0 auto;
        }
        
        .heatmap-container img {
            max-width: 100%;
            height: auto;
            object-fit: contain;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #f0f4f8;
            color: #1e3a8a;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            box-shadow: 4px 4px 8px #d1d9e6, -4px -4px 8px #ffffff;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            box-shadow: 6px 6px 12px #d1d9e6, -6px -6px 12px #ffffff;
            transform: translateY(-2px);
        }
        
        /* Navigation bar */
        .nav-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background-color: #f0f4f8;
            border-radius: 12px;
            box-shadow: 4px 4px 8px #d1d9e6, -4px -4px 8px #ffffff;
            margin-bottom: 1.5rem;
        }
        
        .nav-item {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            color: #1e3a8a;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .nav-item:hover {
            background-color: #e2e8f0;
            box-shadow: inset 2px 2px 5px #d1d9e6, inset -2px -2px 5px #ffffff;
        }
        
        .nav-item.active {
            background-color: #1e3a8a;
            color: white;
            box-shadow: inset 2px 2px 5px #0f1c45, inset -2px -2px 5px #2d58cf;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1rem;
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 2rem;
            border-top: 1px solid #e2e8f0;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f0f4f8;
            border-radius: 10px;
            box-shadow: 4px 4px 8px #d1d9e6, -4px -4px 8px #ffffff;
            padding: 0.75rem 1rem;
            color: #1e3a8a;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            box-shadow: 6px 6px 12px #d1d9e6, -6px -6px 12px #ffffff;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #1e3a8a;
        }
        
        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #1e3a8a;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 4px 4px 8px rgba(0,0,0,0.2);
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True
)

# --- Navigation Bar ---
def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script>
            function nav_page(page_name, timeout_secs) {
                var timer = setTimeout(function() {
                    window.location.href = page_name
                }, timeout_secs*1000);
            }
            nav_page('%s', %d);
        </script>
    """ % (page_name, timeout_secs)
    st.markdown(nav_script, unsafe_allow_html=True)

# App state management
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Navigation bar buttons
def render_nav_bar():
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button('Home', key='home_btn'):
            st.session_state.current_page = 'Home'
            st.rerun()

    
    with col2:
        if st.button('About', key='about_btn'):
            st.session_state.current_page = 'About'
            st.rerun()

    
    with col3:
        if st.button('How It Works', key='how_btn'):
            st.session_state.current_page = 'How It Works'
            st.rerun()

    
    with col4:
        if st.button('FAQ', key='faq_btn'):
            st.session_state.current_page = 'FAQ'
            st.rerun()

    
    with col5:
        if st.button('Contact', key='contact_btn'):
            st.session_state.current_page = 'Contact'
            st.rerun()

    
    # Highlight current page
    st.markdown(
        f"""
        <script>
            document.querySelector('button[kind="secondary"]:contains("{st.session_state.current_page}")').classList.add('active');
        </script>
        """,
        unsafe_allow_html=True
    )

# --- Sidebar ---
st.sidebar.markdown(
    """
    <h1>🩺 Pneumonia Classifier</h1>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <p><strong>Instructions</strong></p>
    <p>1. Upload a chest X-ray (PNG/JPG)</p>
    <p>2. Review AI Diagnosis & Confidence</p>
    <p>3. See heatmap explanation</p>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# Add sidebar sections AI Tool
st.sidebar.markdown(
    """
    <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 20px;">
        <h3 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 10px;">About This Tool</h3>
        <p style="color: #e2e8f0; font-size: 0.9rem;">
            This AI-powered tool helps detect pneumonia from chest X-rays using deep learning technology.
            It provides instant analysis with visual explanations.
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Model Loading ---
@st.cache_resource
def load_model():
    path = "pneumonia deploy.pth"
    if not os.path.exists(path):
        st.sidebar.error("Model file 'pneumonia deploy.pth' not found.")
        return None
    m = model_loader.load_model(path)
    m.eval()
    return m

# --- Preprocessing, Prediction, Explanation ---
def preprocess(img: Image.Image) -> torch.Tensor:
    tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tr(img)

def predict(m, tensor: torch.Tensor):
    with torch.no_grad():
        out = m(tensor.unsqueeze(0))
        p = torch.sigmoid(out).item()
        lbl = "Pneumonia" if p >= 0.5 else "Normal"
    return lbl, p

def explain(m, tensor: torch.Tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device)
    ig = IntegratedGradients(m)
    attr = ig.attribute(tensor.unsqueeze(0).to(device), target=0)
    return attr.squeeze().cpu().detach().numpy().transpose(1,2,0)

# --- Main UI ---
render_nav_bar()

# Display different content based on current page
if st.session_state.current_page == 'Home':
    st.title("🩺 Pneumonia Detection from Chest X-ray")
    st.markdown("""
        <p style="font-size: 1.1rem; color: #475569; margin-bottom: 1rem;">
            Upload a chest X-ray to see AI Diagnosis, Confidence, and an explainable heatmap.
            This tool helps medical professionals and patients understand potential pneumonia cases.
        </p>
    """, unsafe_allow_html=True)

    # File uploader 
    st.subheader("Upload X-ray Image")
    file = st.file_uploader("📤 Drop or Select an Image", type=["png","jpg","jpeg"])
    st.markdown("""
        <p style="font-size: 0.9rem; color: #64748b; font-style: italic;">
            Supported formats: PNG, JPG, JPEG
        </p>
    """, unsafe_allow_html=True)

    model = load_model()

    if file and model:
        img = Image.open(file).convert("RGB")
        proc = preprocess(img)
        label, conf = predict(model, proc)
        
        # Progress indicator
        with st.spinner("Analyzing X-ray..."):
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulating processing time
                import time
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        
        st.success("Analysis complete!")
        
        # Results section 
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns([1.5, 1])
        
        # Show uploaded image 
        with col1:
            st.markdown("<h3>Uploaded X-ray</h3>", unsafe_allow_html=True)
            st.image(img, width=400)
        
        # Bordered result & confidence 
        with col2:
            
            # Diagnosis section
            result_color = "#10b981" if label == "Normal" else "#ef4444"
            st.markdown(
                f"""
                <div class='bordered'>
                <div class='section'>
                <h4>AI Diagnosis</h4>
                <div class='value' style="color: {result_color};">{label}</div>
                </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Confidence section with progress bar visualization
            confidence_percentage = conf*100 if label == "Pneumonia" else (1-conf)*100
            st.markdown(
                f"""
                <div class='section'>
                    <h4>Confidence</h4>
                    <div class='value'>{confidence_percentage:.1f}%</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # visual confidence meter
            st.progress(confidence_percentage/100)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add tooltip explanation
            st.markdown(
                """
                <div class="tooltip" style="margin-top: 10px;">
                    ℹ️ What does this mean?
                    <span class="tooltiptext">
                        The AI model analyzes patterns in the X-ray to detect signs of pneumonia.
                        Higher confidence indicates stronger evidence in the image.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Heatmap explanation 
        heat = explain(model, proc)
        orig_np = np.array(img.resize((224,224)))
        
        # Configure the heatmap visualization 
        fig, _ = viz.visualize_image_attr(
            heat, orig_np, method="blended_heat_map",
            sign="all", show_colorbar=True, title=""
        )
        
        # Set a smaller fixed size for the figure
        fig.set_size_inches(5, 5)
        fig.tight_layout()
        
        st.subheader("📊 Heatmap Explanation")
        
        st.markdown("""
            <p style="font-size: 1rem; color: #475569; margin-bottom: 1rem;">
                The heatmap highlights areas the AI focused on to make its diagnosis.
                Red areas indicate features associated with pneumonia, while blue areas
                indicate features associated with normal lungs.
            </p>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='heatmap-container'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Technical details 
        with st.expander("ℹ️ Technical Details"):
            st.markdown(
                """
                <div style="padding: 10px;">
                    <h4 style="color: #1e3a8a; margin-bottom: 10px;">Model Information</h4>
                    <ul style="list-style-type: none; padding-left: 0;">
                        <li style="margin-bottom: 8px; padding-left: 20px; position: relative;">
                            <span style="position: absolute; left: 0; color: #1e3a8a;">•</span>
                            <strong>Architecture:</strong> DenseNet-161 (custom weights)
                        </li>
                        <li style="margin-bottom: 8px; padding-left: 20px; position: relative;">
                            <span style="position: absolute; left: 0; color: #1e3a8a;">•</span>
                            <strong>Input:</strong> 224×224 grayscale
                        </li>
                        <li style="margin-bottom: 8px; padding-left: 20px; position: relative;">
                            <span style="position: absolute; left: 0; color: #1e3a8a;">•</span>
                            <strong>Output:</strong> Pneumonia / Normal
                        </li>
                        <li style="margin-bottom: 8px; padding-left: 20px; position: relative;">
                            <span style="position: absolute; left: 0; color: #1e3a8a;">•</span>
                            <strong>Explainability:</strong> Integrated Gradients (Captum)
                        </li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Add "How It Works" section (works only when no file is uploaded) 
    if not file:
        st.subheader("How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                """
                <div style="text-align: center; padding: 10px;">
                    <h3 style="font-size: 1.2rem;">1. Upload</h3>
                    <p style="font-size: 0.9rem;">
                        Upload a chest X-ray image in PNG or JPG format.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div style="text-align: center; padding: 10px;">
                    <h3 style="font-size: 1.2rem;">2. Analyze</h3>
                    <p style="font-size: 0.9rem;">
                        Our AI model analyzes the image for signs of pneumonia.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                """
                <div style="text-align: center; padding: 10px;">
                    <h3 style="font-size: 1.2rem;">3. Interpret</h3>
                    <p style="font-size: 0.9rem;">
                        View the diagnosis, confidence level, and visual explanation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == 'About':
    st.title("About")
    st.markdown("""
        <p style="font-size: 1.1rem; color: #475569;">
            The Pneumonia Classifier is an AI-powered tool designed to assist medical professionals and patients
            in detecting pneumonia from chest X-ray images. Using advanced deep learning techniques, our system
            analyzes X-ray images and provides instant results with visual explanations.
        </p>
        <p style="font-size: 1.1rem; color: #475569; margin-top: 1rem;">
            Our mission is to make pneumonia detection more accessible and understandable, helping to improve
            early diagnosis and treatment outcomes.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == 'How It Works':
    st.title("How It Works")
    
    st.markdown("""
        <p style="font-size: 1.1rem; color: #475569;">
            Our pneumonia detection system uses a deep learning model called DenseNet-161, which has been
            trained on thousands of labeled chest X-ray images. Here's how the process works:
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>

    </p>
    <h3>1. Image Upload</h3>
    <p style="font-size: 1rem; color: #475569;">
    You upload a chest X-ray image through our secure interface.</p>
    <h3>2. Preprocessing</h3>
    <p style="font-size: 1rem; color: #475569;">
    The image is automatically resized, normalized, and prepared for analysis.</p>
    <h3>3. AI Analysis</h3>
    <p style="font-size: 1rem; color: #475569;">
    Our deep learning model analyzes the image, looking for patterns associated with pneumonia.</p>
    <h3>4. Results Generation</h3>
    <p style="font-size: 1rem; color: #475569;">
    The system provides a diagnosis (Pneumonia or Normal) along with a confidence score.</p>
    <h3>5. Visual Explanation</h3>
    <p style="font-size: 1rem; color: #475569;">
    A heatmap is generated to show which areas of the X-ray influenced the AI's decision,
    making the diagnosis more transparent and understandable.</p>

            
        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == 'FAQ':
    st.title("Frequently Asked Questions")
    
    with st.expander("How accurate is the AI diagnosis?"):
        st.markdown("""
            <p style="font-size: 1rem; color: #475569;">
                Our model achieves approximately 90-95% accuracy on validated test sets. However, it should be used
                as a supportive tool for healthcare professionals, not as a replacement for clinical diagnosis.
            </p>
        """, unsafe_allow_html=True)
    
    with st.expander("What types of X-ray images work best?"):
        st.markdown("""
            <p style="font-size: 1rem; color: #475569;">
                The system works best with frontal (PA or AP) chest X-rays that are clearly focused and properly exposed.
                Digital X-rays in PNG or JPG format provide the best results.
            </p>
        """, unsafe_allow_html=True)
    
    with st.expander("Is my data secure and private?"):
        st.markdown("""
            <p style="font-size: 1rem; color: #475569;">
                Yes, we prioritize data security and privacy. Uploaded images are processed in real-time and are not
                stored permanently on our servers. All processing is done securely.
            </p>
        """, unsafe_allow_html=True)
    
    with st.expander("What does the heatmap show?"):
        st.markdown("""
            <p style="font-size: 1rem; color: #475569;">
                The heatmap highlights regions of the X-ray that influenced the AI's decision. Red areas indicate
                features associated with pneumonia, while blue areas indicate features associated with normal lungs.
                This visualization helps make the AI's decision-making process more transparent.
            </p>
        """, unsafe_allow_html=True)
    
    with st.expander("Can this tool detect COVID-19?"):
        st.markdown("""
            <p style="font-size: 1rem; color: #475569;">
                This specific model is trained to detect bacterial and viral pneumonia in general, not COVID-19
                specifically. While COVID-19 can cause pneumonia visible on X-rays, this tool is not validated
                for COVID-19 diagnosis.
            </p>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.current_page == 'Contact':
    st.markdown("<h1>Contact Us</h1>", unsafe_allow_html=True)

    st.markdown("""
        <p style="font-size: 1.1rem; color: #475569; margin-bottom: 2rem;">
            We'd love to hear from you! If you have questions, feedback, or need support,
            please reach out using the form below or contact us directly.
        </p>
    """, unsafe_allow_html=True)


    # Optional: Add a separator or heading here
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

    # Contact info section
    st.markdown("""
        <div style="background-color: #f0f4f8; padding: 20px; border-radius: 12px; box-shadow: inset 4px 4px 8px #d1d9e6, inset -4px -4px 8px #ffffff;">
            <h3 style="color: #1e3a8a; margin-bottom: 15px;">Contact Information</h3>
            <p style="margin-bottom: 10px;"><strong>Email:</strong> support@PFE.com</p>
            <p style="margin-bottom: 10px;"><strong>Phone:</strong> +1 (555) 123-4567</p>
            <p style="margin-bottom: 10px;"><strong>Address:</strong> USTO</p>
            <p style="margin-bottom: 10px;"><strong>Hours:</strong> Monday–Friday, 9am–5pm </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown(
    """
    <p>Built with ❤️ by Chennoufi Djebril & Meghazi oussama | Streamlit • PyTorch • Captum</p>
    <p style="font-size: 0.8rem; margin-top: 5px;">© 2025 Pneumonia Classifier. All rights reserved.</p>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
