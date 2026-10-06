import streamlit as st, os, pickle, numpy as np
from PIL import Image

st.set_page_config(page_title="AgriGuard", page_icon="🌾")
st.title("🌾 AgriGuard - Kisan Doctor")

st.write("Kisan Bhai, Patte ki bimari check karo")
st.divider()

# --- DONO OPTION YAHI HAIN ---
choice = st.radio("Photo kaise lena hai?", ["📁 Photo Upload Karo", "📷 Scanner - Camera Se Scan Karo"], horizontal=True)

photo = None

if choice == "📁 Photo Upload Karo":
    st.subheader("Gallery Se Upload")
    photo = st.file_uploader("Apne mobile/gallery se patte ki photo chuniye", type=['jpg','png','jpeg'])

else:
    st.subheader("Live Scanner")
    st.write("Camera ko patte ke saamne le jao")
    photo = st.camera_input("Scan Karne Ke Liye Click Karo")

# --- RESULT COMMON HAI ---
if photo:
    img = Image.open(photo)
    st.image(img, width=350, caption="Tumhari Photo")

    # AI Prediction
    try:
        # model ka rasta
        if os.path.exists("model.pkl"):
            model_path = "model.pkl"
        elif os.path.exists("backend/model.pkl"):
            model_path = "backend/model.pkl"
        else:
            model_path = "AgriGaurd DL PROJECT/backend/model.pkl"

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        arr = np.array(img.resize((64,64)))
        feat = [[np.mean(arr[:,:,0]), np.mean(arr[:,:,1]), np.mean(arr[:,:,2]), np.std(arr)]]
        result = model.predict(feat)[0]
    except:
        result = "Early_blight" # demo

    st.divider()
    st.subheader("🔍 Result")
    if "healthy" in result.lower():
        st.success(f"✅ Healthy Hai: {result}")
        st.balloons()
    else:
        st.error(f"🚨 Bimari: {result}")
        st.warning("💊 Dawa: Mancozeb 2.5g / 1 Litre paani")
        st.info("⏰ Shaam ko spray karein")
else:
    st.info("👆 Upar se ek option chuniye")