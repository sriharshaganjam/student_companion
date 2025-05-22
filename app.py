import streamlit as st
import pandas as pd
import random
import requests

# ----------------- SETUP -----------------

st.set_page_config(page_title=" AI powered Student Companion", layout="wide")
st.title("Student X's - Curriculum Performance Dashboard")

# ----------------- LOAD CURRICULUM -----------------

@st.cache_data
def load_curriculum(csv_path="curriculum.csv"):
    df = pd.read_csv(csv_path)

    # Extract both original and target chapters
    original = df[['Semester', 'Subject', 'Chapter']].rename(columns={'Chapter': 'Chapter_Name'})
    target = df[['Target Semester', 'Target Subject', 'Target Chapter']].rename(
        columns={'Target Semester': 'Semester', 'Target Subject': 'Subject', 'Target Chapter': 'Chapter_Name'}
    )
    curriculum_df = pd.concat([original, target], ignore_index=True)
    curriculum_df.drop_duplicates(inplace=True)

    # Simulate scores (only Semester 1 has real values for the initial load)
    def simulate_score(sem): return random.randint(0, 100) if sem == 1 else 0
    curriculum_df["Score"] = curriculum_df["Semester"].apply(simulate_score)

    # Sort cleanly
    return curriculum_df.sort_values(by=["Semester", "Subject", "Chapter_Name"]).reset_index(drop=True)

curriculum_df = load_curriculum()

# ----------------- BUILD DEPENDENCY MAP -----------------

@st.cache_data
def build_dependency_map(csv_path="curriculum.csv"):
    df = pd.read_csv(csv_path)
    dependency_map = {}

    for _, row in df.iterrows():
        # Skip rows with missing dependency fields
        if pd.isna(row["Related Type"]) or pd.isna(row["Target Chapter"]) or pd.isna(row["Target Semester"]):
            continue

        key = (row['Semester'], row['Subject'], row['Chapter'])
        info = {
            "relation": row["Related Type"],
            "target": row["Target Chapter"],
            "target_semester": int(row["Target Semester"]),
            "target_subject": row["Target Subject"] 
        }
        dependency_map.setdefault(key, []).append(info)

    return dependency_map

dependency_map = build_dependency_map()

# ----------------- SIMULATE SCORES -----------------

def simulate_scores():
    chapters = curriculum_df[['Semester', 'Subject', 'Chapter_Name']].rename(columns={'Chapter_Name': 'Chapter'})
    # Ensure no NaNs before iterating and converting to int for 'Semester'
    chapters = chapters.dropna(subset=['Semester', 'Subject', 'Chapter']).drop_duplicates()
    data = []

    for _, row in chapters.iterrows():
        sem = int(row['Semester'])
        score = random.randint(0, 100) if sem == 1 else 0
        data.append({
            'Semester': sem,
            'Subject': row['Subject'],
            'Chapter': row['Chapter'],
            'Score': score
        })

    return pd.DataFrame(data).sort_values(by=['Semester', 'Subject', 'Chapter'])

# ----------------- MISTRAL SETUP -----------------

mistral_api_key = st.secrets.get("mistral", {}).get("api_key")
mistral_api_base_url = st.secrets.get("mistral", {}).get("base_url", "https://api.mistral.ai/v1")

if not mistral_api_key:
    st.error("🚨 Mistral API key not found. Please add it in .streamlit/secrets.toml.")
    st.stop()

# ----------------- FEEDBACK GENERATION -----------------

def generate_feedback(chapter, score, semester, subject):
    if score == 0:
        return "🕐 No score yet. Feedback will be generated once this chapter is attempted.", ""

    key = (semester, subject, chapter)
    future_links = dependency_map.get(key, [])

    # Define the common style for the overall usage note
    formatted_dependency_text_style = "font-size: 1.2em; font-weight: bold;" 

    # Style for just the 'Related Type' value
    underline_style = "text-decoration: underline;"

    raw_usage_note_content = "" # This will hold the plain text of the dependency note
    if not future_links:
        raw_usage_note_content = "This chapter is a foundational concept for future topics."
    else:
        # Construct grammatically correct sentences with specific formatting
        formatted_links = []
        for link in future_links:
            formatted_links.append(
                f"It is <span style='{underline_style}'>{link['relation'].replace('_', ' ').lower()}</span> for '{link['target']}' "
                f"in Semester {int(link['target_semester'])} ({link['target_subject']})"
            )
        
        if len(formatted_links) == 1:
            raw_usage_note_content = f"This chapter is important because {formatted_links[0]}."
        elif len(formatted_links) == 2:
            raw_usage_note_content = f"This chapter is important because {formatted_links[0]} and {formatted_links[1]}."
        else:
            last_link = formatted_links.pop()
            raw_usage_note_content = f"This chapter is important because {', '.join(formatted_links)}, and {last_link}."


    # This is the HTML-formatted string that will be displayed by st.markdown separately.
    html_formatted_usage_note = f"<br><br><span style='{formatted_dependency_text_style}'>{raw_usage_note_content}</span>"


    if score <= 40:
        tone = "a clear and firm nudge"
    elif score <= 60:
        tone = "a constructive and helpful note"
    else:
        tone = "an encouraging and positive message"

    prompt = (
        f"As a friendly student companion app, give {tone} to a student who scored {score}% in the chapter '{chapter}'. "
        f"Be concise (under 200 words). "
        "Keep the tone supportive and helpful—not like a teacher, but like a friend."
    )

    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a friendly AI companion that helps students understand how their current learning connects to future success. Generate general feedback for the student based on their score."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(f"{mistral_api_base_url}/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            mistral_response_content = response.json()['choices'][0]['message']['content'].strip()
            
            # Add the placeholder text based on score
            placeholder_text = ""
            if score <= 40:
                placeholder_text = "Let me know if you want to practice this chapter once again before retaking the test to improve your scores."
            elif score >= 41:
                placeholder_text = "Do you want to improve your scores by revising the chapter once again or retaking the test?"
            
            # Append placeholder text to Mistral's response with a line break
            if placeholder_text:
                mistral_response_content += f"<br><br>{placeholder_text}"

            return mistral_response_content, html_formatted_usage_note
        else:
            return f"⚠️ Mistral API Error (status code {response.status_code})", ""
    except Exception as e:
        return f"⚠️ Exception occurred: {e}", ""

# ----------------- SCORE SIMULATION UI -----------------

if st.button("🎲 Simulate New Scores"):
    st.session_state['scores_df'] = simulate_scores()

if 'scores_df' not in st.session_state:
    st.session_state['scores_df'] = simulate_scores()

scores_df = st.session_state['scores_df']
scores_df['Score (%)'] = scores_df['Score'].astype(str) + "%"

st.subheader("📘 Full Semester-wise Chapter Scores")
st.dataframe(scores_df.drop(columns=['Score']), use_container_width=True)

# ----------------- FEEDBACK DISPLAY -----------------

st.subheader("💬 Auto-Generated Feedback")
for _, row in scores_df.iterrows():
    chapter = row['Chapter']
    score = row['Score']
    semester = int(row['Semester'])
    subject = row['Subject']
    
    # MODIFIED: Increased font size of chapter name using HTML span and inline style
    st.markdown(f"<span style='font-size: 1.2em;'>**📖 {chapter}** — *Score: {score}%*</span>", unsafe_allow_html=True)

    # Generate feedback, which now returns two parts
    mistral_feedback_text, dependency_html_string = generate_feedback(chapter, score, semester, subject)

    # Use a container to group the info box and the markdown if you want them visually connected.
    # This creates a single blue box around both parts.
    with st.container(border=True):
        # Display the Mistral feedback (plain text, now potentially including placeholder text)
        st.write(mistral_feedback_text, unsafe_allow_html=True) 

        # Only display the dependency note if it's not empty
        if dependency_html_string:
            st.markdown(dependency_html_string, unsafe_allow_html=True)