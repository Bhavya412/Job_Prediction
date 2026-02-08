# Full Streamlit App with Sidebar + Resume-Based Charts
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import io
import pdfplumber
import docx

from preprocessing.data_cleaning import clean_data
from preprocessing.feature_engineering import prepare_input

st.set_page_config(layout="wide", page_title="NYC Job Dashboard")
st.markdown("<h1 style='text-align: center;'>Jobs Analysis and Prediction Dashboard</h1>", unsafe_allow_html=True)

# Load and clean data
df = pd.read_csv("data/Jobs_NYC_Postings.csv", low_memory=False)
df.columns = df.columns.str.strip()
df = clean_data(df)

# Load model
model = joblib.load("models/salary_predictor.pkl")

# Sidebar filters
st.sidebar.header("\U0001F50D Search Filters")
job_roles = sorted(df['Business Title'].dropna().unique().tolist())
selected_job = st.sidebar.selectbox("Select Job Role", ["All"] + job_roles)

skills_input = st.sidebar.text_input("Enter Skills (e.g., Python, Excel)")

locations = sorted(df['Work Location'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Select Location", ["All"] + locations)

# Resume Matching
st.header("\U0001F4C4 Resume-Based Job Matching")
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf", "docx", "txt"])

def extract_text_from_resume(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return ' '.join(page.extract_text() or '' for page in pdf.pages)
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return ' '.join(p.text for p in doc.paragraphs)
    elif file.name.endswith('.txt'):
        return file.read().decode('utf-8')
    return ''

matched_jobs = pd.DataFrame()
if uploaded_file:
    resume_text = extract_text_from_resume(uploaded_file).lower()
    st.success("✅ Resume uploaded and text extracted!")

    def match_score(skills_text):
        if pd.isna(skills_text): return 0
        job_text = skills_text.lower()
        return sum(1 for word in resume_text.split() if word in job_text)

    df['Match Score'] = df['Preferred Skills'].apply(match_score)
    matched_jobs = df.sort_values(by='Match Score', ascending=False).head(5)

    st.subheader("\U0001F3AF Top 5 Matched Jobs Based on Resume")
    for _, row in matched_jobs.iterrows():
        st.markdown(f"**{row['Business Title']} @ {row['Agency']}**")
        st.markdown(f"- \U0001F4CD {row['Work Location']}")
        st.markdown(f"- \U0001F4B0 ${row['Salary Range From']} - ${row['Salary Range To']} ({row['Salary Frequency']})")
        st.markdown(f"- ⭐ Match Score: {row['Match Score']}")
        st.markdown("---")

# Filter data based on sidebar
filtered_df = df.copy()
if selected_job != "All":
    filtered_df = filtered_df[filtered_df['Business Title'] == selected_job]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['Work Location'] == selected_location]
if skills_input:
    skills_keywords = [skill.strip().lower() for skill in skills_input.split(',')]
    filtered_df = filtered_df[filtered_df['Preferred Skills'].str.lower().fillna('').apply(
        lambda text: any(skill in text for skill in skills_keywords))]

# Show job results (Top 5)
st.subheader(f"\U0001F3AF Job Listings: Showing {min(5, len(filtered_df))} of {len(filtered_df)}")
for _, row in filtered_df.head(5).iterrows():
    st.markdown(f"### {row.get('Business Title', 'N/A')} @ {row.get('Agency', 'N/A')}")
    st.markdown(f"**Location:** {row.get('Work Location', 'N/A')}")
    st.markdown(f"**Salary:** ${row.get('Salary Range From', 0)} - ${row.get('Salary Range To', 0)} ({row.get('Salary Frequency', '')})")
    st.markdown(f"**Career Level:** {row.get('Career Level', 'N/A')}")
    description = row.get('Job Description', '')
    short_desc = description[:300] + ('...' if len(description) > 300 else '')
    st.markdown(f"**Job Description:** {short_desc}")
    st.markdown("---")

# Download
excel_buffer = io.BytesIO()
filtered_df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
excel_buffer.seek(0)
st.download_button(
    label="⬇️ Download Job Matches as Excel",
    data=excel_buffer,
    file_name="job_matches.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Visualizations Section
st.header("📊 Visual Insights")
if not filtered_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        top_agencies = filtered_df['Agency'].value_counts().nlargest(10).reset_index()
        top_agencies.columns = ['Agency', 'Postings']
        fig = px.bar(top_agencies, x='Postings', y='Agency', orientation='h', color='Agency', title='Top Hiring Agencies')
        st.plotly_chart(fig)

    with col2:
        category_counts = filtered_df['Job Category'].value_counts().nlargest(6).reset_index()
        category_counts.columns = ['Job Category', 'Count']
        fig = px.pie(category_counts, names='Job Category', values='Count', title='Top Job Categories')
        st.plotly_chart(fig)

    st.subheader("💰 Salary Range Histogram")
    if 'Salary Range From' in filtered_df and 'Salary Range To' in filtered_df:
        filtered_df['Average Salary'] = (filtered_df['Salary Range From'] + filtered_df['Salary Range To']) / 2
        fig = px.histogram(filtered_df, x='Average Salary', nbins=30, title="Salary Distribution")
        st.plotly_chart(fig)

    st.subheader("📊 Career Level Bubble Chart by Job Category")
    if 'Career Level' in filtered_df and 'Job Category' in filtered_df:
        bubble_df = filtered_df.groupby(['Career Level', 'Job Category']).size().reset_index(name='Count')
        fig = px.scatter(bubble_df, x='Career Level', y='Job Category', size='Count', color='Count', title='Career Level vs. Job Category (Bubble Chart)')
        st.plotly_chart(fig)

    st.subheader("🗺️ Job Locations Treemap")
    location_counts = filtered_df['Work Location'].value_counts().nlargest(15).reset_index()
    location_counts.columns = ['Location', 'Count']
    fig = px.treemap(location_counts, path=['Location'], values='Count', title='Job Distribution by Location')
    st.plotly_chart(fig)

    if 'Preferred Skills' in filtered_df:
        st.subheader("☁️ Word Cloud: Preferred Skills")
        text = ' '.join(filtered_df['Preferred Skills'].dropna().astype(str))
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
else:
    st.info("No data for sidebar filters.")

# Salary Prediction
st.header("\U0001F916 Predict Expected Salary")
st.write("Enter job details to get an estimated salary based on historical data.")

with st.form("prediction_form"):
    title = st.selectbox("Job Title", sorted(df['Business Title'].dropna().unique()))
    agency = st.selectbox("Agency", sorted(df['Agency'].dropna().unique()))
    location = st.selectbox("Location", sorted(df['Work Location'].dropna().unique()))
    job_category = st.selectbox("Job Category", sorted(df['Job Category'].dropna().unique()))
    experience = st.slider("Years of Experience", 0, 30, 3)
    submitted = st.form_submit_button("Predict Salary")

    if submitted:
        try:
            user_input = prepare_input(title, agency, location, experience, job_category)
            prediction = model.predict(user_input)[0]
            st.success(f"\U0001F911 Estimated Annual Salary: ${int(prediction):,}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
