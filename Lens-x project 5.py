import streamlit as st
import joblib
import pandas as pd
import requests
import io # Import io module for file handling
import base64 # Import base64 module

# ---------------------------- PAGE CONFIG ---------------------------- #
st.set_page_config(page_title="AI & NLP Combo Suite", layout="wide", page_icon="🤖")

# ---------------------------- MODEL LOADING ---------------------------- #
# Movie Recommender
try:
    movie_df = joblib.load("movies.pkl")
    movie_vectors = joblib.load("vectors.pkl")
    movie_model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Movie recommender models (movies.pkl, vectors.pkl, model.pkl) not found. Please ensure they are in the correct directory.")
    st.stop()


# NLP Models
try:
    spam_model = joblib.load("spam_classifier.pkl")
    language_model = joblib.load("lang_det.pkl")
    news_model = joblib.load("news_cat.pkl")
    review_model = joblib.load("review.pkl")
except FileNotFoundError:
    st.error("One or more NLP models not found. Please ensure spam_classifier.pkl, lang_det.pkl, news_cat.pkl, review.pkl are in the correct directory.")
    st.stop()


# ---------------------------- HELPER FUNCTIONS ---------------------------- #
API_KEY = "7a5423f6"
def fetch_poster(imdb_id):
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        poster = data.get("Poster")
        if poster == "N/A" or not poster:
            return "https://via.placeholder.com/300x450?text=No+Image"
        return poster
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

def recommend_movies(movie_name):
    index = movie_df[movie_df.name == movie_name].index[0]
    distances, indexes = movie_model.kneighbors([movie_vectors[index]], n_neighbors=6)
    recommendations = []
    for i in indexes[0][1:]:
        name = movie_df.loc[i]['name']
        imdb_id = movie_df.loc[i]['movie_id']
        poster_url = fetch_poster(imdb_id)
        recommendations.append((name, imdb_id, poster_url))
    return recommendations

# Function to generate a custom HTML download link with base64 encoding
def get_download_link(file_path, link_text, button_color="#4CAF50"):
    with open(file_path, "rb") as file:
        data = file.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}"><button style="background-color:{button_color}; color:white; font-size:18px; padding:10px 20px; border:none; border-radius:8px;">{link_text}</button></a>'
    return href

# ---------------------------- SIDEBAR NAVIGATION ---------------------------- #
# Custom CSS for sidebar background color
st.markdown(
    """
    <style>
    .stSidebar {
        background-color: #add8e6; /* Light Blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("# 🚀 Navigation Panel")

# Using st.sidebar.radio for navigation
option = st.sidebar.radio(
    "Go to",
    ("🏠 Home", "ℹ️ About Me", "📝 Projects", "📞 Contact") # Updated navigation options
)

# ---------------------------- PAGE CONTENT BASED ON NAVIGATION ---------------------------- #

if option == "🏠 Home":
    st.markdown("""
        <h1 style='color:#ffa500;'>I'm Praveen Kumar Tripathi </h1>
        <h3>Aspiring AI/ML Engineer</h3>
        <p>I'm a motivated and dedicated student with a strong passion for Artificial Intelligence (AI) and Machine Learning (ML).</p>
        <p>My interests span across Natural Language Processing, Computer Vision, and Data-Driven Problem Solving. I enjoy building intelligent systems that solve real-world challenges and thrive in collaborative environments where innovation happens.</p>
        <h3>Data Analytics</h3>
        <p>I'm passionate about transforming raw data into actionable insights that drive real business outcomes. My goal is to leverage analytical techniques to solve complex problems and contribute directly to strategic decision-making.</p>        
        <br>
    """, unsafe_allow_html=True)
    
    # Use the custom function to generate the colored download button
    try:
        button_html = get_download_link("Praveen Kumar Tripathi.pdf", "📄 Download CV", "#4CAF50") # Green color
        st.markdown(button_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CV file 'Praveen Kumar Tripathi.pdf' not found. Please ensure the file is in the same directory.")
        
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712107.png", width=150)

elif option == "ℹ️ About Me":
    st.header("About Me")
    about_me_tab = st.tabs(["Education", "Skills", "Certifications", "Internship"])

    with about_me_tab[0]: # Education Tab
        st.subheader("Education")
        education_data = {
            "Qualification": ["MBA", "B.Sc", "12th", "10th"],
            "Stream": ["Human Resource & Marketing", "Zoology/Botany/Chemistry", "Science", "-"],
            "Passout Year": [2024, 2022, 2019, 2017],
            "Institute": ["AKTU", "VBSPU", "UP Board", "UP Board"],
            "City/State": ["Ghazipur, Uttar Pradesh", "Ghazipur, Uttar Pradesh", "Ghazipur, Uttar Pradesh", "Ghazipur, Uttar Pradesh"],
            "Score": ["6.91 CGPA", "64.6%", "73.6%", "80%"]
        }
        st.table(pd.DataFrame(education_data))

    with about_me_tab[1]: # Skills Tab
        st.subheader("Skills & Tools")
        skills_data = [
            "Python", "MYSQL", "Statistics", "Power BI", "Tableau", "Adv.Excel", "NumPy",  "Pandas",
            "Matplotlib", "Seaborn", "Machine Learning",  "NLP", "Streamlit", "Scikit-learn",
            "Ms.PowerPoint", "GitHub"
        ]
        # Display skills in a grid-like format for better visual appeal
        cols = st.columns(6) # Adjust number of columns as needed
        for i, skill in enumerate(skills_data):
            cols[i % 6].markdown(f"**{skill}**")

    with about_me_tab[2]: # Certifications Tab
        st.subheader("Certifications")
        st.image("Data Analytics Certificate.png",width=350, caption="Data Analytics Certificate", use_column_width=False)
        
        # New certificates added here
        st.image("Power point certificate.png", width=350, caption="PowerPoint Certificate", use_container_width=False)
        st.image("Rubicon Employability certificate.jpg", width=350, caption="Rubicon Employability Certificate", use_container_width=False)
        st.image("Excel Certificate.png", width=350, caption="Excel Certificate", use_container_width=False)
        st.image("MBA Certificate.jpg", width=350, caption="MBA Certificate", use_container_width=False)
        
        st.markdown("""
        - **Certified Data Analytics Specialist** — Proficient in a full data stack including Python, SQL (MySQL, MongoDB), NumPy, Pandas, Statistics, Power BI, and Tableau.
        - **Advanced Tool Proficiency** — Possess advanced skills in Microsoft Excel and PowerPoint for detailed analysis and impactful data presentations.
        - **End-to-End Data Workflow** — Capable of transforming raw data into actionable insights and presenting them effectively.
        - **Driving Data-Driven Decisions** — My skills are focused on helping organizations make informed decisions based on solid data analysis.
        - **Created Interactive dashboards and data storytelling reports to guide strategic decision-making in a simulated business context.**
        - **My MBA from a reputable university like Dr. A.P.J. Abdul Kalam Technical University has equipped me with a strong foundation in business administration, which I am eager to apply to real-world challenges.** 
        - **Employability Skills Training** – Rubicon LifeSkills Program (Sept 2023) 
          – Covered Communication, Teamwork, Interview Skills & Corporate Readiness.             
        - **GitHub Repository (Certificates Folder): [View All Certificates]**(https://github.com/PraveenTripathi020/My_Certificate.git)
        """)

    with about_me_tab[3]: # Internship Tab
        st.subheader("Internship")
        st.image("Digital marketing internship certificate.jpg",width=350, caption="Digital Marketing Internship Certificate", use_container_width=False)
        st.markdown("""
        - **Internship Experience**: "I successfully completed a 2-month digital marketing internship at G-TECH INFO INDIA PRIVATE LIMITED, from August 25, 2023, to October 25, 2023."
        - **Skills Acquired**: During my internship, I gained hands-on experience in various aspects of digital marketing, including
          - **Search Engine Optimization (SEO)**
          - **Social Media Marketing**
          - **Email Marketing**
          - **Content Marketing**
          - **Pay-Per-Click Advertising (PPC)**
          - **Analytics and Reporting**
        - **Contributions**: "I demonstrated a strong commitment to learning and actively contributed to the success of the digital marketing initiatives during my internship."
        - **Supervision**: "I worked under the guidance of my supervisor, Sushma Yadav, who holds the title of SEO Manager."
        - **GitHub Repository**: [View Intership Certificate]: (https://github.com/PraveenTripathi020/Internship_Certificate.git)          
        """)

elif option == "📝 Projects":
    st.header("My Projects")
    project_tabs = st.tabs(["LensX (NLP Suite)", "Data Dashboards", "Banking Automation System", "Restaurant Billing System", "Resume Builder"])

    with project_tabs[0]: # LensX (NLP Suite)
        st.subheader("Project 1: Lens-X Project")
        st.write("""
        - **Description**: This project is a comprehensive suite of AI and NLP models, including a movie recommender, spam classifier, food sentiment analyzer, language detector, and news classifier.
        - **Technologies Used**: Python, Streamlit, Scikit-learn, nltk, Numpy,  Pandas,  Requests
        - **Link**: [GitHub Repository](https://github.com/PraveenTripathi020/Lens-expert.git)
        """)
        st.subheader("🤖 AI/NLP Models Suite")
        model_choice = st.selectbox(
            "Select an AI/NLP Model",
            ["🎥 Movie Recommendation", "📩 Spam Classifier", "⭐ Food Sentiment", "🌍 Language Detection", "📢 News Classification"],
            index=0 # Default selected model
        )

        if model_choice == "🎥 Movie Recommendation":
            st.header("🎬 Movie Recommender")
            selected_movie = st.selectbox("Search or Select a Movie", sorted(movie_df['name'].unique()), index=None, placeholder="Search here")
            if st.button("🍿 Recommend"):
                if not selected_movie:
                    st.warning("Please select a movie first!")
                else:
                    with st.spinner("Fetching recommendations..."):
                        recommendations = recommend_movies(selected_movie)
                    st.subheader("⭐ Top 5 Recommendations")
                    cols = st.columns(5)
                    for col, (name, imdb_id, poster) in zip(cols, recommendations):
                        with col:
                            st.image(poster, use_column_width=True, caption=name)
                            st.markdown(f"[🔗 IMDB](https://www.imdb.com/title/{imdb_id})")

        elif model_choice == "📩 Spam Classifier":
            st.header("🚫 Spam Detector")
            st.write("### Single Message Prediction")
            msg = st.text_input("Enter message")
            if st.button("Detect Single Message"):
                pred = spam_model.predict([msg])
                st.success("📩 Not Spam" if pred[0] == 1 else "🚫 Spam")

            st.write("### Bulk Prediction (Upload CSV/TXT)")
            uploaded_file_spam = st.file_uploader("Upload a file (CSV/TXT) for Spam Detection", type=["csv", "txt"], key="spam_uploader")
            if uploaded_file_spam is not None:
                try:
                    # Read the file content
                    if uploaded_file_spam.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file_spam)
                    else: # Assuming .txt, read as lines
                        string_data = io.StringIO(uploaded_file_spam.getvalue().decode("utf-8")).read()
                        df = pd.DataFrame(string_data.splitlines(), columns=["text"])

                    if not df.empty and 'text' in df.columns: # Ensure 'text' column exists
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        if st.button("Run Bulk Spam Detection"):
                            with st.spinner("Processing..."):
                                df['Prediction'] = spam_model.predict(df['text'])
                                df['Result'] = df['Prediction'].apply(lambda x: "Not Spam 📩" if x == 1 else "Spam 🚫")
                                st.subheader("Bulk Spam Detection Results")
                                st.dataframe(df[['text', 'Result']])
                    else:
                        st.error("The uploaded file must contain a column named 'text' for processing, or be a plain text file with one message per line.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        elif model_choice == "⭐ Food Sentiment":
            st.header("🍽️ Food Sentiment Analyzer")
            st.write("### Single Review Prediction")
            review = st.text_input("Enter review")
            if st.button("Analyze Single Review"):
                pred = review_model.predict([review])
                st.success("👍 Liked Food 😃" if pred[0] == 1 else "👎 Disliked Food 😔")

            st.write("### Bulk Prediction (Upload CSV/TXT)")
            uploaded_file_sentiment = st.file_uploader("Upload a file (CSV/TXT) for Sentiment Analysis", type=["csv", "txt"], key="sentiment_uploader")
            if uploaded_file_sentiment is not None:
                try:
                    if uploaded_file_sentiment.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file_sentiment)
                    else:
                        string_data = io.StringIO(uploaded_file_sentiment.getvalue().decode("utf-8")).read()
                        df = pd.DataFrame(string_data.splitlines(), columns=["text"])

                    if not df.empty and 'text' in df.columns:
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        if st.button("Run Bulk Sentiment Analysis"):
                            with st.spinner("Processing..."):
                                df['Prediction'] = review_model.predict(df['text'])
                                df['Result'] = df['Prediction'].apply(lambda x: "Liked Food 👍😃" if x == 1 else "Disliked Food 👎😔")
                                st.subheader("Bulk Sentiment Analysis Results")
                                st.dataframe(df[['text', 'Result']])
                    else:
                        st.error("The uploaded file must contain a column named 'text' for processing, or be a plain text file with one review per line.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        elif model_choice == "🌍 Language Detection":
            st.header("🌐 Language Detector")
            st.write("### Single Sentence Prediction")
            sentence = st.text_input("Enter a sentence")
            if st.button("Detect Single Language"):
                pred = language_model.predict([sentence])
                st.info(f"Detected Language: 🌍 {pred[0]}")

            st.write("### Bulk Prediction (Upload CSV/TXT)")
            uploaded_file_lang = st.file_uploader("Upload a file (CSV/TXT) for Language Detection", type=["csv", "txt"], key="lang_uploader")
            if uploaded_file_lang is not None:
                try:
                    if uploaded_file_lang.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file_lang)
                    else:
                        string_data = io.StringIO(uploaded_file_lang.getvalue().decode("utf-8")).read()
                        df = pd.DataFrame(string_data.splitlines(), columns=["text"])

                    if not df.empty and 'text' in df.columns:
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        if st.button("Run Bulk Language Detection"):
                            with st.spinner("Processing..."):
                                df['Prediction'] = language_model.predict(df['text'])
                                df['Result'] = df['Prediction'].apply(lambda x: f"🌍 {x}")
                                st.subheader("Bulk Language Detection Results")
                                st.dataframe(df[['text', 'Result']])
                    else:
                        st.error("The uploaded file must contain a column named 'text' for processing, or be a plain text file with one sentence per line.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        elif model_choice == "📢 News Classification":
            st.header("🗞️ News Category Detector")
            st.write("### Single News Prediction")
            news = st.text_input("Enter news content")
            if st.button("Classify Single News"):
                pred = news_model.predict([news])
                st.success(f"📢 News Category: {pred[0]}")

            st.write("### Bulk Prediction (Upload CSV/TXT)")
            uploaded_file_news = st.file_uploader("Upload a file (CSV/TXT) for News Classification", type=["csv", "txt"], key="news_uploader")
            if uploaded_file_news is not None:
                try:
                    if uploaded_file_news.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file_news)
                    else:
                        string_data = io.StringIO(uploaded_file_news.getvalue().decode("utf-8")).read()
                        df = pd.DataFrame(string_data.splitlines(), columns=["text"])

                    if not df.empty and 'text' in df.columns:
                        st.write("Preview of uploaded data:")
                        st.dataframe(df.head())
                        if st.button("Run Bulk News Classification"):
                            with st.spinner("Processing..."):
                                df['Prediction'] = news_model.predict(df['text'])
                                df['Result'] = df['Prediction'].apply(lambda x: f"📢 {x}")
                                st.subheader("Bulk News Classification Results")
                                st.dataframe(df[['text', 'Result']])
                    else:
                        st.error("The uploaded file must contain a column named 'text' for processing, or be a plain text file with one news entry per line.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")


    with project_tabs[1]: # Data Dashboards
        st.subheader("Data Dashboards")
        
        # IPL Dashboard
        st.markdown("---")
        st.subheader("1. IPL Dashboard (2008-2024)")
        st.image("IPL Dashboard (2).png", caption="IPL Dashboard (2008-2024)", use_container_width=True)
        st.markdown("""
        यह डैशबोर्ड IPL (इंडियन प्रीमियर लीग) के 2008 से 2024 तक के डेटा का गहन विश्लेषण प्रस्तुत करता है।
        इसमें टीमों के प्रदर्शन, खिलाड़ियों के आँकड़े और विभिन्न सीज़न के विजेताओं की जानकारी शामिल है।
        यह क्रिकेट प्रेमियों और विश्लेषकों के लिए एक मूल्यवान संसाधन है।
        
        [GitHub Repository](https://github.com/PraveenTripathi020/IPL_Dashboard.git)
        """)

        # HR Analytics Dashboard
        st.markdown("---")
        st.subheader("2. HR Analytics Dashboard")
        st.image("HR Analytics Dashboard-1.png", caption="HR Analytics Dashboard", use_container_width=True)
        st.markdown("""
        यह HR एनालिटिक्स डैशबोर्ड मानव संसाधन डेटा का व्यापक अवलोकन प्रदान करता है।
        यह कर्मचारियों की संख्या, औसत वेतन, विभाग-वार विश्लेषण और कर्मचारी जनसांख्यिकी जैसे महत्वपूर्ण HR मेट्रिक्स को ट्रैक करने में मदद करता है।
        यह प्रबंधन को बेहतर निर्णय लेने में सहायता करता है।
        
        [GitHub Repository](https://github.com/PraveenTripathi020/HR_Analytics_Dashboard.git)
        """)

        # Sachin ODI Dashboard
        st.markdown("---")
        st.subheader("3. Sachin ODI Dashboard")
        st.image("Sachin Dashboard.png", caption="Sachin ODI Dashboard", use_container_width=True)
        st.markdown("""
        यह डैशबोर्ड महान क्रिकेटर सचिन तेंदुलकर के वनडे अंतर्राष्ट्रीय (ODI) करियर के आँकड़ों पर केंद्रित है।
        इसमें उनके रन, शतक, मैच, जीत-हार का रिकॉर्ड और विभिन्न मैदानों पर उनके प्रदर्शन का विस्तृत विश्लेषण शामिल है।
        यह सचिन के प्रशंसकों और क्रिकेट सांख्यिकीविदों के लिए एक शानदार संसाधन है।
        
        [GitHub Repository](https://github.com/PraveenTripathi020/Sachin_ODI_Dashboard.git)
        """)

        # Covid-19 Dashboard
        st.markdown("---")
        st.subheader("4. Covid-19 Dashboard")
        st.image("Covid-19 Dashboard.png", caption="Covid-19 Dashboard", use_container_width=True)
        st.markdown("""
        यह COVID-19 डैशबोर्ड भारत में महामारी की स्थिति को दर्शाता है।
        इसमें पुष्ट मामले, सक्रिय मामले, ठीक हुए मामले, मृत्यु और रिकवरी दर जैसे महत्वपूर्ण डेटा शामिल हैं।
        यह सार्वजनिक स्वास्थ्य अधिकारियों और आम जनता को स्थिति को समझने में मदद करता है।
        
        [GitHub Repository](https://github.com/PraveenTripathi020/Covid_19_Dashboard.git)
        """)

        # Olympic Sports Dashboard
        st.markdown("---")
        st.subheader("5. Olympic Sports Dashboard (1896-2016)")
        st.image("Olympic sports Dashboard.png", caption="Olympic Sports Dashboard (1896-2016)", use_container_width=True)
        st.markdown("""
        यह ओलंपिक स्पोर्ट्स डैशबोर्ड 1896 से 2016 तक के ओलंपिक खेलों के डेटा का विश्लेषण करता है।
        इसमें कुल खेल, कुल प्रतिभागी, लिंग के आधार पर पदक की संख्या और शीर्ष पदक जीतने वाले देशों की जानकारी शामिल है।
        यह खेल प्रेमियों और शोधकर्ताओं के लिए ओलंपिक इतिहास को समझने का एक उत्कृष्ट उपकरण है।
        
        [GitHub Repository](https://github.com/PraveenTripathi020/Olympic_Sports_Dashboard.git)
        """)

        # Superstore  Dashboard
        st.markdown("---")
        st.subheader("6. Superstore Dashboard With Tableau")
        st.image("Superstore dashboard.png", caption="Superstore Dashboard", use_container_width=True)
        st.markdown("""
        यह डैशबोर्ड सुपरस्टोर के प्रदर्शन का एक संक्षिप्त अवलोकन प्रदान करता है। इसमें कुल ऑर्डर, बिक्री और लाभ के साथ-साथ साल-दर-साल बिक्री और लाभ का रुझान (trend) दिखाया गया है। 
        इसके अलावा, यह विभिन्न उप-श्रेणियों (sub-categories) के प्रदर्शन को भी उजागर करता है ताकि यह पता चल सके कि कौन से उत्पाद सबसे अधिक लाभदायक या बिक्री वाले हैं।
        
        [Tableau Public](https://public.tableau.com/views/Book215april/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
        """)
        
        # Interactive Dynamic Sales Report Dashboard
        st.markdown("---")
        st.subheader("7. Interactive Dynamic Sales Report Dashboard With Excel")
        st.image("Excel Dashboard.png", caption="Interactive Dynamic Sales Report Dashboard", use_container_width=True)
        st.markdown("""
        यह एक्सेल डैशबोर्ड बिक्री, ऑर्डर, मात्रा और लाभ जैसे प्रमुख प्रदर्शन संकेतकों (KPIs) को दर्शाता है। इसमें तिमाही और श्रेणी के अनुसार बिक्री को विभिन्न चार्ट के माध्यम से प्रस्तुत किया गया है।
        दाईं ओर दिए गए फ़िल्टर इसे इंटरैक्टिव बनाते हैं, जिससे उपयोगकर्ता अपनी पसंद के डेटा का आसानी से विश्लेषण कर सकते हैं।
        यह डैशबोर्ड बिक्री के प्रदर्शन का एक व्यापक, फिर भी संक्षिप्त अवलोकन प्रदान करता है।
                    
        [GitHub Repository](https://github.com/PraveenTripathi020/Excel-Dashboard.git)
        """)

        # MS-Powerpoint Project
        st.markdown("---")
        st.subheader("8. Food Delivery App Sales Presentation With Ms-Powerpoint")
        st.image("Ms-power point project.png", caption="Food Delivery App Sales Presentation", use_container_width=True)
        st.markdown("""              
        यह पावरपॉइंट स्लाइड एक फ़ूड डिलीवरी ऐप की बिक्री (Sales) का ग्राफ़ दिखाती है। यहाँ कुछ मुख्य बिंदु दिए गए हैं:
        यह स्लाइड 2020 से 2025 तक एक फ़ूड डिलीवरी ऐप की बिक्री का प्रदर्शन करती है। ग्राफ़ में 2023 तक बिक्री में लगातार वृद्धि देखी गई,
        जिसके बाद 2024 में थोड़ी गिरावट आई और 2025 में फिर से सुधार हुआ। यह प्रस्तुति कंपनी के पिछले और अनुमानित बिक्री प्रदर्शन को दर्शाती है
                    
        [GitHub Repository](https://github.com/PraveenTripathi020/Ms-power-point-project.git)
        """)

    with project_tabs[2]: # Banking Automation System
        st.subheader("Banking Automation System")
        st.image("Banking Automation.png", caption="Banking Automation System Interface", use_container_width=True)
        st.markdown("""
        A desktop-based Python application that simulates core banking operations.
        This project focuses on automation of basic financial tasks with a clean interface.

        **Key Features:**
        * Account creation & management
        * Deposit & withdrawal functionality
        * Transaction history & balance check
        * Secure login system

        **Tech Stack:**
        * Python - Tkinter - File Handling

        [View Source on GitHub](https://github.com/PraveenTripathi020/Online_Banking.git)
        """)
         
       # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
    with project_tabs[3]: # Restaurant Billing System (NEW TAB)
        st.subheader("Restaurant Billing System")
        st.image("Restaurant Billing System.png", caption="Restaurant Billing System Interface", use_container_width=True) # Assuming the image is in the same directory as the script.
        st.markdown("""
        A user-friendly desktop application designed to streamline billing operations for restaurants.
        This system aims to simplify order management, bill generation, and payment processing, enhancing efficiency in daily restaurant operations.

        **Key Features:**
        * Menu item selection and quantity management
        * Automatic bill calculation including taxes and discounts (if applicable)
        * Order tracking and management
        * Simple and intuitive graphical user interface

        **Tech Stack:**
        * Python - Tkinter - File Handling (or database for larger scale)
        

        [View Source on GitHub](https://github.com/PraveenTripathi020/My_Restaurant_bill.git)
        """)
   
    with project_tabs[4]: # Resume Builder
        st.subheader("Resume Builder")
        st.write("Description and details about your Resume Builder project will go here.")
        st.info("Coming Soon!")

elif option == "📞 Contact":
    st.header("Contact Me")
    st.write("Feel free to reach out!")
    st.markdown("""
    - **Email**: rishaloopandit020@gmail.com
    - **LinkedIn**: [Your LinkedIn Profile](your_linkedin_profile_link)
    - **GitHub**: [My GitHub Profile](https://github.com/PraveenTripathi020)
    """)

# ---------------------------- FOOTER ---------------------------- #
st.markdown("""
---
<div style='text-align: center;'>
🤖 <b>AI & NLP Combo Suite</b> | Built with ❤️ by Praveen Kumar Tripathi | © 2025
</div>
""", unsafe_allow_html=True)


