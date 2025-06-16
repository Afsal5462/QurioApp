import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import io
import json
import re
from datetime import datetime
import plotly.express as px
import pandas as pd
#cod
# Set your API Key  
genai.configure(api_key="AIzaSyD9RsCzKBYRUm8RK6deeeQQAqkP7GP2Mag")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

st.set_page_config(page_title="Smart Document Quiz Generator", page_icon="🧠", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {"mcq": [], "short_answer": [], "true_false": []}
if 'doc_text' not in st.session_state:
    st.session_state.doc_text = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    return text

def generate_enhanced_quiz(doc_text, difficulty, num_mcq, num_short, num_tf, focus_areas):
    focus_instruction = f"Focus on: {', '.join(focus_areas)}." if focus_areas else ""
    
    prompt = f"""
    Given the following document, generate a comprehensive quiz for self-learning:
    ---
    {doc_text}
    ---
    
    Create a {difficulty.lower()} level quiz with:
    - {num_mcq} multiple choice questions (4 options each, indicate correct answer with explanation)
    - {num_short} short answer questions (with detailed answers and key points)
    - {num_tf} true/false questions (with explanations)
    
    {focus_instruction}
    
    For each question, also provide:
    - Learning objective
    - Difficulty rating (1-5)
    - Explanation for the correct answer
    - Common mistakes to avoid
    
    Format as JSON:
    {{
        "mcq": [{{
            "question": "...", 
            "options": ["A", "B", "C", "D"], 
            "answer": "A",
            "explanation": "...",
            "learning_objective": "...",
            "difficulty": 3,
            "common_mistakes": "..."
        }}],
        "short_answer": [{{
            "question": "...", 
            "answer": "...",
            "key_points": ["point1", "point2"],
            "learning_objective": "...",
            "difficulty": 3
        }}],
        "true_false": [{{
            "statement": "...",
            "answer": true,
            "explanation": "...",
            "learning_objective": "...",
            "difficulty": 2
        }}]
    }}
    """
    
    response = model.generate_content(prompt)
    return response.text

def reset_quiz():
    st.session_state.page = 'upload'
    st.session_state.current_quiz = None
    st.session_state.user_answers = {"mcq": [], "short_answer": [], "true_false": []}
    st.session_state.doc_text = ""
    st.session_state.uploaded_file_name = ""

# Navigation
def show_navigation():
    pages = {
        'upload': '📄 Upload Document',
        'quiz': '📝 Take Quiz', 
        'results': '📊 Results & Analysis',
        'history': '📈 Learning Progress'
    }
    
    cols = st.columns(len(pages))
    for i, (page_key, page_name) in enumerate(pages.items()):
        with cols[i]:
            if page_key == st.session_state.page:
                st.markdown(f"**🔹 {page_name}**")
            else:
                if st.button(page_name, key=f"nav_{page_key}", disabled=(
                    (page_key == 'quiz' and not st.session_state.current_quiz) or
                    (page_key == 'results' and not st.session_state.get('quiz_completed', False))
                )):
                    st.session_state.page = page_key
                    st.rerun()

# PAGE 1: DOCUMENT UPLOAD
def page_upload():
    st.markdown("<h1 style='text-align:center; color:#38bdf8;'>🧠 Smart Document Quiz Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b; font-size:1.2em;'>Upload your document and customize your learning experience</p>", unsafe_allow_html=True)
    
    # Configuration sidebar
    st.sidebar.header("⚙️ Quiz Configuration")
    difficulty_level = st.sidebar.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
    num_mcq = st.sidebar.slider("Multiple Choice Questions", 0, 10, 5)
    num_short = st.sidebar.slider("Short Answer Questions", 0, 8, 3)
    num_tf = st.sidebar.slider("True/False Questions", 0, 5, 2)
    question_types = st.sidebar.multiselect("Focus Areas", 
        ["Key Concepts", "Definitions", "Examples", "Applications", "Analysis"], 
        default=["Key Concepts", "Definitions"])

    # File Upload Section
    st.markdown("## 📄 Document Upload")
    uploaded_file = st.file_uploader("Choose your document", type=["pdf", "docx", "txt"], 
                                   help="Upload a PDF, Word document, or text file to generate a quiz from")

    if uploaded_file:
        # Extract and store document text
        doc_text = extract_text(uploaded_file)
        st.session_state.doc_text = doc_text
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Document preview and stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📑 Document Preview")
            with st.expander("View Document Content", expanded=False):
                st.write(doc_text[:2000] + ("..." if len(doc_text) > 2000 else ""))
        
        with col2:
            st.markdown("### 📊 Document Analysis")
            word_count = len(doc_text.split())
            reading_time = max(1, word_count // 200)
            st.metric("Reading Time", f"{reading_time} min")
            st.metric("Word Count", f"{word_count:,}")
            st.metric("Characters", f"{len(doc_text):,}")

        # Generate Quiz Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Generate Smart Quiz", type="primary", use_container_width=True, key="generate_quiz"):
                with st.spinner("🔄 Creating your personalized quiz... This may take a moment."):
                    try:
                        response_text = generate_enhanced_quiz(
                            doc_text, difficulty_level, num_mcq, num_short, num_tf, question_types
                        )
                        cleaned_text = re.sub(r"```json|```", "", response_text).strip()
                        quiz_data = json.loads(cleaned_text)
                        
                        st.session_state.current_quiz = {
                            'data': quiz_data,
                            'difficulty': difficulty_level,
                            'timestamp': datetime.now(),
                            'document_name': uploaded_file.name,
                            'config': {
                                'num_mcq': num_mcq,
                                'num_short': num_short,
                                'num_tf': num_tf,
                                'focus_areas': question_types
                            }
                        }
                        
                        # Reset user answers for new quiz
                        st.session_state.user_answers = {"mcq": [], "short_answer": [], "true_false": []}
                        
                        st.success("🎉 Quiz generated successfully!")
                        st.balloons()
                        
                        # Auto-navigate to quiz page
                        st.session_state.page = 'quiz'
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error generating quiz: {str(e)}")
                        with st.expander("Debug Information"):
                            st.code(response_text, language="json")

    # Study Tips
    st.markdown("---")
    with st.expander("💡 Study Tips for Better Learning"):
        st.markdown("""
        **Before Taking the Quiz:**
        - 📖 Read through the document carefully
        - ✏️ Take notes on key concepts
        - 🎯 Pay attention to the focus areas you selected
        
        **During the Quiz:**
        - 🤔 Read each question thoroughly
        - ⏰ Don't rush - understanding is key
        - 📝 For short answers, include key terms and concepts
        
        **After the Quiz:**
        - 📊 Review your results carefully
        - 🔄 Retake quizzes to reinforce learning
        - 📚 Focus extra study time on missed concepts
        """)

# PAGE 2: TAKE QUIZ
def page_quiz():
    if not st.session_state.current_quiz:
        st.error("No quiz available. Please upload a document and generate a quiz first.")
        if st.button("← Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    quiz_data = st.session_state.current_quiz['data']
    
    st.markdown("<h1 style='text-align:center; color:#38bdf8;'>📝 Take Your Quiz</h1>", unsafe_allow_html=True)
    
    # Quiz info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document", st.session_state.current_quiz['document_name'])
    with col2:
        st.metric("Difficulty", st.session_state.current_quiz['difficulty'])
    with col3:
        total_questions = len(quiz_data.get('mcq', [])) + len(quiz_data.get('short_answer', [])) + len(quiz_data.get('true_false', []))
        st.metric("Total Questions", total_questions)
    with col4:
        est_time = total_questions * 2  # 2 minutes per question estimate
        st.metric("Est. Time", f"{est_time} min")

    st.markdown("---")
    
    # Initialize user answers if not already done
    if not st.session_state.user_answers["mcq"] and 'mcq' in quiz_data:
        st.session_state.user_answers["mcq"] = [""] * len(quiz_data["mcq"])
    if not st.session_state.user_answers["true_false"] and 'true_false' in quiz_data:
        st.session_state.user_answers["true_false"] = [None] * len(quiz_data["true_false"])
    if not st.session_state.user_answers["short_answer"] and 'short_answer' in quiz_data:
        st.session_state.user_answers["short_answer"] = [""] * len(quiz_data["short_answer"])

    # Multiple Choice Questions
    if 'mcq' in quiz_data and quiz_data['mcq']:
        st.markdown("## 🔘 Multiple Choice Questions")
        for idx, q in enumerate(quiz_data["mcq"]):
            st.markdown(f"### Question {idx+1}")
            st.markdown(f"**Difficulty:** {'⭐' * q.get('difficulty', 3)}")
            st.write(q['question'])
            
            user_choice = st.radio(
                "Select your answer:",
                q["options"],
                key=f"mcq_{idx}",
                index=q["options"].index(st.session_state.user_answers["mcq"][idx]) if st.session_state.user_answers["mcq"][idx] in q["options"] else 0
            )
            st.session_state.user_answers["mcq"][idx] = user_choice
            
            if q.get('learning_objective'):
                st.caption(f"🎯 Learning Objective: {q['learning_objective']}")
            st.markdown("---")

    # True/False Questions
    if 'true_false' in quiz_data and quiz_data['true_false']:
        st.markdown("## ✅ True/False Questions")
        for idx, q in enumerate(quiz_data["true_false"]):
            st.markdown(f"### Statement {idx+1}")
            st.markdown(f"**Difficulty:** {'⭐' * q.get('difficulty', 2)}")
            st.write(q['statement'])
            
            user_tf = st.radio(
                "Is this statement true or false?",
                ["True", "False"],
                key=f"tf_{idx}",
                index=0 if st.session_state.user_answers["true_false"][idx] is None else (0 if st.session_state.user_answers["true_false"][idx] else 1)
            )
            st.session_state.user_answers["true_false"][idx] = (user_tf == "True")
            
            if q.get('learning_objective'):
                st.caption(f"🎯 Learning Objective: {q['learning_objective']}")
            st.markdown("---")

    # Short Answer Questions
    if 'short_answer' in quiz_data and quiz_data['short_answer']:
        st.markdown("## ✍️ Short Answer Questions")
        for idx, q in enumerate(quiz_data["short_answer"]):
            st.markdown(f"### Question {idx+1}")
            st.markdown(f"**Difficulty:** {'⭐' * q.get('difficulty', 3)}")
            st.write(q['question'])
            
            user_input = st.text_area(
                "Your answer:",
                key=f"sa_{idx}",
                height=100,
                value=st.session_state.user_answers["short_answer"][idx]
            )
            st.session_state.user_answers["short_answer"][idx] = user_input
            
            if q.get('learning_objective'):
                st.caption(f"🎯 Learning Objective: {q['learning_objective']}")
            st.markdown("---")

    # Submit Quiz
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Submit Quiz & Get Results", type="primary", use_container_width=True):
            st.session_state.quiz_completed = True
            st.session_state.page = 'results'
            st.rerun()

# PAGE 3: RESULTS & ANALYSIS
def page_results():
    if not st.session_state.current_quiz or not st.session_state.get('quiz_completed', False):
        st.error("No quiz results available.")
        return
    
    quiz_data = st.session_state.current_quiz['data']
    user_answers = st.session_state.user_answers
    
    st.markdown("<h1 style='text-align:center; color:#38bdf8;'>📊 Quiz Results & Analysis</h1>", unsafe_allow_html=True)
    
    # Calculate Scores
    with st.spinner("Analyzing your performance..."):
        scores = {"mcq": 0, "true_false": 0, "short_answer": 0}
        total_questions = {"mcq": 0, "true_false": 0, "short_answer": 0}
        
        # Grade MCQ
        if 'mcq' in quiz_data and quiz_data['mcq']:
            total_questions["mcq"] = len(quiz_data["mcq"])
            for idx, q in enumerate(quiz_data["mcq"]):
                if idx < len(user_answers["mcq"]):
                    user_choice = user_answers["mcq"][idx].strip()
                    correct_answer = q["answer"].strip()
                    
                    if (user_choice.upper() == correct_answer.upper() or 
                        user_choice == correct_answer or
                        (len(user_choice) > 1 and user_choice in q["options"] and 
                         q["options"].index(user_choice) == ord(correct_answer.upper()) - ord('A'))):
                        scores["mcq"] += 1
        
        # Grade True/False
        if 'true_false' in quiz_data and quiz_data['true_false']:
            total_questions["true_false"] = len(quiz_data["true_false"])
            for idx, q in enumerate(quiz_data["true_false"]):
                if idx < len(user_answers["true_false"]) and user_answers["true_false"][idx] == q["answer"]:
                    scores["true_false"] += 1
        
        # Grade Short Answers
        if 'short_answer' in quiz_data and quiz_data['short_answer']:
            total_questions["short_answer"] = len(quiz_data["short_answer"])
            for idx, q in enumerate(quiz_data["short_answer"]):
                if idx < len(user_answers["short_answer"]):
                    user_ans = user_answers["short_answer"][idx].lower()
                    correct_ans = q["answer"].lower()
                    if any(word in user_ans for word in correct_ans.split()[:3]):
                        scores["short_answer"] += 0.5
    
    total_score = sum(scores.values())
    total_possible = sum(total_questions.values())
    percentage = (total_score / total_possible * 100) if total_possible > 0 else 0
    
    # Display Overall Results
    st.markdown("## 🎯 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{percentage:.1f}%", delta=f"{total_score:.1f}/{total_possible}")
    with col2:
        st.metric("MCQ Score", f"{scores['mcq']}/{total_questions['mcq']}")
    with col3:
        st.metric("T/F Score", f"{scores['true_false']}/{total_questions['true_false']}")
    with col4:
        st.metric("Short Answer", f"{scores['short_answer']:.1f}/{total_questions['short_answer']}")
    
    # Performance Message
    if percentage >= 80:
        st.success("🎉 Excellent work! You have a strong understanding of the material.")
    elif percentage >= 60:
        st.info("👍 Good job! Review the areas you missed for better understanding.")
    else:
        st.warning("📚 Keep studying! Focus on the concepts you found challenging.")
    
    # Detailed Answer Review
    st.markdown("---")
    st.markdown("### 📋 Detailed Answer Review")
    
    # MCQ Review
    if 'mcq' in quiz_data and quiz_data['mcq']:
        st.subheader("🔘 Multiple Choice Review")
        for idx, q in enumerate(quiz_data["mcq"]):
            user_ans = user_answers["mcq"][idx] if idx < len(user_answers["mcq"]) else "No answer"
            
            user_choice = user_ans.strip()
            correct_answer = q["answer"].strip()
            is_correct = (user_choice.upper() == correct_answer.upper() or 
                        user_choice == correct_answer or
                        (len(user_choice) > 1 and user_choice in q["options"] and 
                         q["options"].index(user_choice) == ord(correct_answer.upper()) - ord('A')))
            
            with st.expander(f"Question {idx+1}: {'✅' if is_correct else '❌'}"):
                st.write(f"**Question:** {q['question']}")
                st.write(f"**Your answer:** {user_ans}")
                st.write(f"**Correct answer:** {q['answer']} - {q['options'][ord(q['answer'].upper()) - ord('A')] if q['answer'].upper() in 'ABCD' else 'N/A'}")
                if q.get('explanation'):
                    st.write(f"**Explanation:** {q['explanation']}")
                if q.get('common_mistakes'):
                    st.write(f"**Common mistakes:** {q['common_mistakes']}")
    
    # True/False Review
    if 'true_false' in quiz_data and quiz_data['true_false']:
        st.subheader("✅ True/False Review")
        for idx, q in enumerate(quiz_data["true_false"]):
            user_ans = user_answers["true_false"][idx] if idx < len(user_answers["true_false"]) else None
            is_correct = user_ans == q["answer"]
            
            with st.expander(f"Statement {idx+1}: {'✅' if is_correct else '❌'}"):
                st.write(f"**Statement:** {q['statement']}")
                st.write(f"**Your answer:** {'True' if user_ans else 'False'}")
                st.write(f"**Correct answer:** {'True' if q['answer'] else 'False'}")
                if q.get('explanation'):
                    st.write(f"**Explanation:** {q['explanation']}")
                if q.get('learning_objective'):
                    st.write(f"**Learning Objective:** {q['learning_objective']}")
    
    # Short Answer Review
    if 'short_answer' in quiz_data and quiz_data['short_answer']:
        st.subheader("✍️ Short Answer Review")
        for idx, q in enumerate(quiz_data["short_answer"]):
            user_ans = user_answers["short_answer"][idx] if idx < len(user_answers["short_answer"]) else "No answer"
            
            user_ans_lower = user_ans.lower()
            correct_ans_lower = q["answer"].lower()
            has_keywords = any(word in user_ans_lower for word in correct_ans_lower.split()[:3])
            
            with st.expander(f"Question {idx+1}: {'✅' if has_keywords else '❌'}"):
                st.write(f"**Question:** {q['question']}")
                st.write(f"**Your answer:** {user_ans}")
                st.write(f"**Model answer:** {q['answer']}")
                
                if q.get('key_points'):
                    st.write("**Key points to include:**")
                    for point in q['key_points']:
                        st.write(f"• {point}")
                
                if q.get('learning_objective'):
                    st.write(f"**Learning Objective:** {q['learning_objective']}")
                
                if has_keywords:
                    st.success("Good! Your answer contains key concepts.")
                else:
                    st.warning("Consider including more key concepts from the model answer.")
    
    # Save to history
    quiz_result = {
        'timestamp': datetime.now(),
        'document': st.session_state.current_quiz['document_name'],
        'score': percentage,
        'difficulty': st.session_state.current_quiz['difficulty'],
        'details': scores,
        'total_questions': total_questions
    }
    
    # Avoid duplicate entries
    if not any(result['timestamp'] == quiz_result['timestamp'] for result in st.session_state.quiz_history):
        st.session_state.quiz_history.append(quiz_result)
    
    # Action Buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.user_answers = {"mcq": [], "short_answer": [], "true_false": []}
            st.session_state.quiz_completed = False
            st.session_state.page = 'quiz'
            st.rerun()
    
    with col2:
        if st.button("📈 View Progress", use_container_width=True):
            st.session_state.page = 'history'
            st.rerun()
    
    with col3:
        if st.button("📄 New Document", use_container_width=True):
            reset_quiz()
            st.rerun()

# PAGE 4: LEARNING PROGRESS
def page_history():
    st.markdown("<h1 style='text-align:center; color:#38bdf8;'>📈 Learning Progress</h1>", unsafe_allow_html=True)
    
    if not st.session_state.quiz_history:
        st.info("No quiz history yet. Take some quizzes to see your progress!")
        if st.button("📄 Upload Document"):
            st.session_state.page = 'upload'
            st.rerun()
        return
    
    df = pd.DataFrame(st.session_state.quiz_history)
    
    # Overall Stats
    st.markdown("## 📊 Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Quizzes", len(df))
    with col2:
        st.metric("Average Score", f"{df['score'].mean():.1f}%")
    with col3:
        st.metric("Best Score", f"{df['score'].max():.1f}%")
    with col4:
        st.metric("Improvement", f"{df['score'].iloc[-1] - df['score'].iloc[0]:.1f}%" if len(df) > 1 else "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Score over time
        fig = px.line(df, x='timestamp', y='score', title='Score Progress Over Time',
                     markers=True, line_shape='spline')
        fig.update_layout(xaxis_title="Date", yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Score by difficulty
        if 'difficulty' in df.columns:
            avg_by_difficulty = df.groupby('difficulty')['score'].mean().reset_index()
            fig = px.bar(avg_by_difficulty, x='difficulty', y='score', 
                        title='Average Score by Difficulty Level')
            fig.update_layout(xaxis_title="Difficulty", yaxis_title="Average Score (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent History
    st.markdown("## 📚 Recent Quiz History")
    for i, quiz in enumerate(reversed(st.session_state.quiz_history[-10:])):
        with st.expander(f"Quiz {len(st.session_state.quiz_history)-i}: {quiz['document']} - {quiz['score']:.1f}%"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Date:** {quiz['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Difficulty:** {quiz['difficulty']}")
                st.write(f"**Overall Score:** {quiz['score']:.1f}%")
            with col2:
                st.write("**Question Type Breakdown:**")
                for q_type, score in quiz['details'].items():
                    total = quiz['total_questions'][q_type]
                    st.write(f"• {q_type.replace('_', ' ').title()}: {score}/{total}")
    
    # Clear History
    st.markdown("---")
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.quiz_history = []
        st.success("History cleared!")
        st.rerun()

# MAIN APP LOGIC
def main():
    # Show navigation
    show_navigation()
    st.markdown("---")
    
    # Route to appropriate page
    if st.session_state.page == 'upload':
        page_upload()
    elif st.session_state.page == 'quiz':
        page_quiz()
    elif st.session_state.page == 'results':
        page_results()
    elif st.session_state.page == 'history':
        page_history()

if __name__ == "__main__":
    main()
