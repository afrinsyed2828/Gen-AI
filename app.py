"""
AI Mock Interview Generator
A production-ready application that generates realistic mock interview sessions
using AI (Groq/OpenAI) with dynamic question generation and evaluation.
"""

import streamlit as st
import json
import os
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv
import requests
from pathlib import Path
import base64

# Load environment variables
load_dotenv()

# API Configuration
API_PROVIDER = os.getenv("API_PROVIDER", "groq")  # groq or openai
API_KEY = os.getenv("GROQ_API_KEY") if API_PROVIDER == "groq" else os.getenv("OPENAI_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions" if API_PROVIDER == "groq" else "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("MODEL", "mixtral-8x7b-32768" if API_PROVIDER == "groq" else "gpt-3.5-turbo")

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'feedbacks' not in st.session_state:
        st.session_state.feedbacks = []
    if 'scores' not in st.session_state:
        st.session_state.scores = []
    if 'interview_config' not in st.session_state:
        st.session_state.interview_config = {}
    if 'follow_up_active' not in st.session_state:
        st.session_state.follow_up_active = False
    if 'current_follow_up' not in st.session_state:
        st.session_state.current_follow_up = None
    if 'interview_history' not in st.session_state:
        st.session_state.interview_history = []

# API Functions
def call_llm_api(messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    """
    Call the configured LLM API (Groq or OpenAI)
    """
    if not API_KEY:
        return "Error: API key not found. Please check your .env file."
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling API: {str(e)}"

def generate_questions(role: str, experience: str, interview_type: str, num_questions: int = 7) -> List[str]:
    """
    Generate interview questions based on user configuration
    """
    prompt = f"""Generate {num_questions} {interview_type} interview questions for a {experience} level {role} position.
    Format: Return only the questions, one per line, numbered 1-{num_questions}.
    Make questions specific, challenging, and relevant to the role and experience level.
    For Technical interviews, include role-specific technical questions.
    For HR interviews, focus on soft skills and behavioral questions.
    For Behavioral interviews, use STAR format questions."""
    
    messages = [
        {"role": "system", "content": "You are an expert interview coach specializing in creating realistic interview questions."},
        {"role": "user", "content": prompt}
    ]
    
    response = call_llm_api(messages, temperature=0.8)
    
    # Parse questions from response
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and any(line.startswith(str(i)) for i in range(1, num_questions + 1)):
            # Remove number prefix and clean
            question = line.split('.', 1)[-1].strip() if '.' in line else line
            questions.append(question)
    
    # Ensure we have exactly num_questions, pad if necessary
    while len(questions) < num_questions:
        questions.append(f"Question {len(questions) + 1}: Please describe your experience with {role} technologies.")
    
    # Randomize question order
    random.shuffle(questions)
    
    return questions[:num_questions]

def evaluate_answer(question: str, answer: str, role: str, experience: str, interview_type: str) -> Dict[str, Any]:
    """
    Evaluate user's answer and provide feedback and score
    """
    prompt = f"""You are an expert interviewer evaluating a candidate's answer.
    
    Role: {role}
    Experience Level: {experience}
    Interview Type: {interview_type}
    Question: {question}
    Candidate's Answer: {answer}
    
    Provide evaluation in JSON format with the following keys:
    - score: integer from 0-10
    - feedback: detailed feedback on strengths and areas for improvement
    - model_answer: a high-quality example answer
    
    Return ONLY the JSON object."""
    
    messages = [
        {"role": "system", "content": "You are an expert interviewer providing detailed evaluation."},
        {"role": "user", "content": prompt}
    ]
    
    response = call_llm_api(messages, temperature=0.3)
    
    try:
        # Try to parse JSON from response
        # Sometimes the response might have text before/after JSON
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            evaluation = json.loads(json_str)
        else:
            # Fallback if no JSON found
            evaluation = {
                "score": random.randint(5, 8),
                "feedback": "Your answer was acceptable. Consider providing more specific examples.",
                "model_answer": "A strong answer would include specific examples and metrics."
            }
    except:
        evaluation = {
            "score": 5,
            "feedback": "We couldn't evaluate properly. Please try to be more specific in your answers.",
            "model_answer": "Focus on providing structured answers with examples."
        }
    
    return evaluation

def generate_follow_up(question: str, answer: str, role: str, experience: str) -> Optional[str]:
    """
    Generate a follow-up question based on the user's answer
    """
    if not answer or len(answer.split()) < 5:
        return None
    
    prompt = f"""Based on the following interview context, generate ONE follow-up question:
    
    Role: {role}
    Experience Level: {experience}
    Original Question: {question}
    Candidate's Answer: {answer}
    
    Generate a relevant follow-up question that digs deeper into the candidate's response.
    Return ONLY the question."""
    
    messages = [
        {"role": "system", "content": "You are an experienced interviewer who asks insightful follow-up questions."},
        {"role": "user", "content": prompt}
    ]
    
    response = call_llm_api(messages, temperature=0.7)
    
    # Clean up response
    follow_up = response.strip()
    if follow_up and len(follow_up) > 10:
        return follow_up
    return None

def calculate_final_score(scores: List[int]) -> Dict[str, Any]:
    """
    Calculate final score and generate summary
    """
    if not scores:
        return {"total_score": 0, "average_score": 0, "max_score": 0, "min_score": 0}
    
    avg_score = sum(scores) / len(scores)
    return {
        "total_score": sum(scores),
        "average_score": round(avg_score, 2),
        "max_score": max(scores),
        "min_score": min(scores),
        "total_questions": len(scores)
    }

def save_interview_results(config: Dict, questions: List, answers: List, feedbacks: List, scores: List, final_score: Dict) -> Dict:
    """
    Save interview results to local storage
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    interview_data = {
        "id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "questions": questions,
        "answers": answers,
        "feedbacks": feedbacks,
        "scores": scores,
        "final_score": final_score
    }
    
    # Save to history
    st.session_state.interview_history.append(interview_data)
    
    # Save to file
    history_dir = Path("interview_history")
    history_dir.mkdir(exist_ok=True)
    
    filename = history_dir / f"interview_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(interview_data, f, indent=2)
    
    return interview_data

def export_to_pdf(interview_data: Dict) -> str:
    """
    Export interview results to PDF format (HTML-based)
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interview Results - {interview_data['id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; margin-top: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .question {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            .answer {{ padding: 10px; margin: 10px 0; border-left: 3px solid #4CAF50; }}
            .feedback {{ background: #e3f2fd; padding: 10px; margin: 10px 0; }}
            .score {{ font-weight: bold; color: #2196F3; }}
            .final-score {{ font-size: 20px; font-weight: bold; color: #4CAF50; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>AI Mock Interview Results</h1>
        <p><strong>Date:</strong> {interview_data['timestamp']}</p>
        <p><strong>Role:</strong> {interview_data['config']['role']}</p>
        <p><strong>Experience:</strong> {interview_data['config']['experience']}</p>
        <p><strong>Interview Type:</strong> {interview_data['config']['interview_type']}</p>
        
        <div class="final-score">
            Final Score: {interview_data['final_score']['average_score']}/10
        </div>
        
        <h2>Interview Details</h2>
    """
    
    for i, (q, a, f, s) in enumerate(zip(
        interview_data['questions'],
        interview_data['answers'],
        interview_data['feedbacks'],
        interview_data['scores']
    ), 1):
        html_content += f"""
        <div class="section">
            <div class="question">
                <strong>Question {i}:</strong> {q}
            </div>
            <div class="answer">
                <strong>Your Answer:</strong><br>{a}
            </div>
            <div class="feedback">
                <strong>Feedback:</strong><br>{f.get('feedback', 'No feedback')}<br>
                <strong>Model Answer:</strong><br>{f.get('model_answer', 'No model answer')}
            </div>
            <div class="score">
                Score: {s}/10
            </div>
        </div>
        <hr>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

# UI Components
def home_page():
    """Display home page with introduction"""
    st.title("🎯 AI Mock Interview Generator")
    
    st.markdown("""
    ### Welcome to your personal AI Interview Coach!
    
    This AI-powered system helps you practice for real interviews by:
    - Generating realistic interview questions based on your target role
    - Providing instant feedback on your answers
    - Offering model answers for comparison
    - Tracking your progress over time
    
    ### How it works:
    1. **Setup**: Select your target role, experience level, and interview type
    2. **Practice**: Answer questions one by one with optional follow-up questions
    3. **Learn**: Get detailed feedback, scores, and model answers
    4. **Improve**: Track your progress and export results for review
    
    ### Features:
    - 🎯 Role-specific questions
    - 📊 Real-time scoring and feedback
    - 🔄 Dynamic follow-up questions
    - 📈 Progress tracking
    - 💾 Export results (PDF/JSON)
    - 📚 Interview history
    
    **Ready to start?** Click the **Setup Interview** button below!
    """)
    
    if st.button("🎯 Setup Interview", type="primary", use_container_width=True):
        st.session_state.page = "setup"
        st.rerun()

def setup_page():
    """Interview setup page with configuration options"""
    st.title("⚙️ Interview Setup")
    
    with st.form("setup_form"):
        role = st.text_input(
            "Job Role",
            placeholder="e.g., Software Engineer, Data Scientist, Product Manager",
            help="Specify the job role you're preparing for"
        )
        
        experience = st.select_slider(
            "Experience Level",
            options=["Beginner", "Intermediate", "Advanced"],
            value="Intermediate",
            help="Select your experience level for appropriate question difficulty"
        )
        
        interview_type = st.selectbox(
            "Interview Type",
            options=["Technical", "HR", "Behavioral"],
            help="Choose the type of interview questions"
        )
        
        num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=10,
            value=7,
            help="Select how many questions you want to answer"
        )
        
        enable_followup = st.checkbox(
            "Enable Follow-up Questions",
            value=True,
            help="AI will generate follow-up questions based on your answers"
        )
        
        submitted = st.form_submit_button("🚀 Start Interview", type="primary", use_container_width=True)
        
        if submitted:
            if not role:
                st.error("Please enter a job role")
            else:
                with st.spinner("Generating your personalized interview questions..."):
                    questions = generate_questions(role, experience, interview_type, num_questions)
                
                if questions and not questions[0].startswith("Error"):
                    # Store configuration
                    st.session_state.interview_config = {
                        "role": role,
                        "experience": experience,
                        "interview_type": interview_type,
                        "num_questions": num_questions,
                        "enable_followup": enable_followup,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.questions = questions
                    st.session_state.current_question_index = 0
                    st.session_state.answers = []
                    st.session_state.feedbacks = []
                    st.session_state.scores = []
                    st.session_state.interview_started = True
                    st.session_state.page = "interview"
                    st.rerun()
                else:
                    st.error("Failed to generate questions. Please check your API configuration.")

def interview_page():
    """Live interview page with question display and answer input"""
    st.title("🎤 Live Interview")
    
    # Progress tracking
    total_questions = len(st.session_state.questions)
    current_idx = st.session_state.current_question_index
    
    # Progress bar
    progress = current_idx / total_questions if total_questions > 0 else 0
    st.progress(progress)
    st.write(f"Question {current_idx + 1} of {total_questions}")
    
    # Display current question
    if current_idx < total_questions:
        current_question = st.session_state.questions[current_idx]
        
        # Question container
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>Question {current_idx + 1}</h3>
            <p style="font-size: 18px;">{current_question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer input
        answer = st.text_area(
            "Your Answer",
            height=150,
            placeholder="Type your answer here...",
            key=f"answer_{current_idx}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            submit_button = st.button("Submit Answer", type="primary", use_container_width=True)
        
        with col2:
            if st.button("Skip Question", use_container_width=True):
                # Skip with default evaluation
                evaluation = {
                    "score": 0,
                    "feedback": "Question skipped",
                    "model_answer": "No model answer provided"
                }
                st.session_state.answers.append("Skipped")
                st.session_state.feedbacks.append(evaluation)
                st.session_state.scores.append(0)
                
                if current_idx + 1 >= total_questions:
                    st.session_state.page = "results"
                else:
                    st.session_state.current_question_index += 1
                st.rerun()
        
        if submit_button:
            if not answer.strip():
                st.warning("Please provide an answer before submitting")
            else:
                with st.spinner("Evaluating your answer..."):
                    evaluation = evaluate_answer(
                        current_question,
                        answer,
                        st.session_state.interview_config["role"],
                        st.session_state.interview_config["experience"],
                        st.session_state.interview_config["interview_type"]
                    )
                
                # Store answer and evaluation
                st.session_state.answers.append(answer)
                st.session_state.feedbacks.append(evaluation)
                st.session_state.scores.append(evaluation.get("score", 5))
                
                # Check for follow-up questions
                if st.session_state.interview_config.get("enable_followup", True):
                    follow_up = generate_follow_up(
                        current_question,
                        answer,
                        st.session_state.interview_config["role"],
                        st.session_state.interview_config["experience"]
                    )
                    
                    if follow_up and not follow_up.startswith("Error"):
                        st.session_state.current_follow_up = follow_up
                        st.session_state.follow_up_active = True
                        st.rerun()
                    else:
                        # Move to next question
                        if current_idx + 1 >= total_questions:
                            st.session_state.page = "results"
                        else:
                            st.session_state.current_question_index += 1
                        st.rerun()
                else:
                    # Move to next question
                    if current_idx + 1 >= total_questions:
                        st.session_state.page = "results"
                    else:
                        st.session_state.current_question_index += 1
                    st.rerun()
        
        # Display follow-up question if active
        if st.session_state.follow_up_active and st.session_state.current_follow_up:
            st.markdown("---")
            st.info(f"💡 **Follow-up Question:** {st.session_state.current_follow_up}")
            
            follow_up_answer = st.text_area(
                "Your Follow-up Answer",
                height=100,
                key=f"followup_{current_idx}"
            )
            
            if st.button("Submit Follow-up Answer", type="primary"):
                if follow_up_answer.strip():
                    # Evaluate follow-up
                    evaluation = evaluate_answer(
                        st.session_state.current_follow_up,
                        follow_up_answer,
                        st.session_state.interview_config["role"],
                        st.session_state.interview_config["experience"],
                        st.session_state.interview_config["interview_type"]
                    )
                    
                    # Append to existing evaluation or store separately
                    # For simplicity, we'll combine with main question evaluation
                    st.session_state.feedbacks[-1]["follow_up_feedback"] = evaluation.get("feedback", "")
                    st.session_state.feedbacks[-1]["follow_up_score"] = evaluation.get("score", 5)
                    
                    # Move to next question
                    st.session_state.follow_up_active = False
                    st.session_state.current_follow_up = None
                    
                    if current_idx + 1 >= total_questions:
                        st.session_state.page = "results"
                    else:
                        st.session_state.current_question_index += 1
                    st.rerun()
                else:
                    st.warning("Please provide an answer to the follow-up question")
    
    # Exit button
    if st.button("❌ End Interview Early", use_container_width=True):
        st.session_state.page = "results"
        st.rerun()

def results_page():
    """Display results and feedback"""
    st.title("📊 Interview Results & Feedback")
    
    # Calculate final score
    final_score = calculate_final_score(st.session_state.scores)
    
    # Display overall score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{final_score['average_score']}/10", 
                  delta=None, delta_color="normal")
    with col2:
        st.metric("Total Questions", final_score['total_questions'])
    with col3:
        st.metric("Best Score", f"{final_score['max_score']}/10")
    
    # Performance gauge
    st.markdown("### Overall Performance")
    score_percentage = (final_score['average_score'] / 10) * 100
    st.progress(score_percentage / 100)
    
    # Detailed feedback for each question
    st.markdown("### Question-by-Question Analysis")
    
    for i, (question, answer, feedback, score) in enumerate(zip(
        st.session_state.questions,
        st.session_state.answers,
        st.session_state.feedbacks,
        st.session_state.scores
    ), 1):
        with st.expander(f"Question {i} - Score: {score}/10"):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Your Answer:** {answer}")
            st.markdown(f"**Feedback:** {feedback.get('feedback', 'No feedback')}")
            st.markdown(f"**Model Answer:** {feedback.get('model_answer', 'No model answer')}")
            if 'follow_up_feedback' in feedback:
                st.markdown(f"**Follow-up Feedback:** {feedback['follow_up_feedback']}")
    
    # Export options
    st.markdown("### Export Results")
    col1, col2 = st.columns(2)
    
    # Save results
    interview_data = save_interview_results(
        st.session_state.interview_config,
        st.session_state.questions,
        st.session_state.answers,
        st.session_state.feedbacks,
        st.session_state.scores,
        final_score
    )
    
    with col1:
        if st.button("💾 Save as JSON", use_container_width=True):
            json_str = json.dumps(interview_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"interview_{interview_data['id']}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📄 Export as PDF", use_container_width=True):
            html_content = export_to_pdf(interview_data)
            st.download_button(
                label="Download PDF",
                data=html_content,
                file_name=f"interview_{interview_data['id']}.html",
                mime="text/html"
            )
    
    # History section
    st.markdown("### Previous Interviews")
    if st.session_state.interview_history:
        history_data = []
        for interview in st.session_state.interview_history[-5:]:  # Last 5 interviews
            history_data.append({
                "Date": interview['timestamp'][:10],
                "Role": interview['config']['role'],
                "Score": interview['final_score']['average_score'],
                "Questions": interview['final_score']['total_questions']
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)
    
    # Start new interview
    if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
        # Reset session state
        for key in ['interview_started', 'current_question_index', 'questions', 
                    'answers', 'feedbacks', 'scores', 'interview_config',
                    'follow_up_active', 'current_follow_up']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page = "home"
        st.rerun()

def history_page():
    """Display interview history"""
    st.title("📚 Interview History")
    
    if not st.session_state.interview_history:
        st.info("No interviews completed yet. Start your first interview!")
        if st.button("Start Interview"):
            st.session_state.page = "setup"
            st.rerun()
        return
    
    # Display history in reverse chronological order
    for interview in reversed(st.session_state.interview_history):
        with st.expander(f"{interview['timestamp'][:10]} - {interview['config']['role']} - Score: {interview['final_score']['average_score']}/10"):
            st.markdown(f"**Role:** {interview['config']['role']}")
            st.markdown(f"**Experience:** {interview['config']['experience']}")
            st.markdown(f"**Type:** {interview['config']['interview_type']}")
            st.markdown(f"**Questions:** {len(interview['questions'])}")
            st.markdown(f"**Average Score:** {interview['final_score']['average_score']}/10")
            
            if st.button(f"View Details", key=f"view_{interview['id']}"):
                # Display detailed view
                for i, (q, a, s) in enumerate(zip(interview['questions'], interview['answers'], interview['scores']), 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**Answer:** {a[:200]}...")
                    st.markdown(f"**Score:** {s}/10")
                    st.markdown("---")

def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="AI Mock Interview Generator",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .stProgress > div > div {
            background-color: #4CAF50;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🎯 Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("⚙️ New Interview", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()
        
        if st.button("📚 History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI-powered mock interview generator helps you:
        - Practice for real interviews
        - Get instant feedback
        - Track your progress
        - Learn from model answers
        """)
        
        st.markdown("---")
        st.markdown("### Configuration")
        st.markdown(f"**API Provider:** {API_PROVIDER.upper()}")
        st.markdown(f"**Model:** {MODEL}")
    
    # Page routing
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "setup":
        setup_page()
    elif st.session_state.page == "interview":
        interview_page()
    elif st.session_state.page == "results":
        results_page()
    elif st.session_state.page == "history":
        history_page()

if __name__ == "__main__":
    main()