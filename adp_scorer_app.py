import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Punjab ADP Project Scorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e5c8a;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .score-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .highly-recommended {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .recommended {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    .conditional {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .needs-revision {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .not-recommended {
        background-color: #f8d7da;
        border-left: 5px solid #721c24;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'projects' not in st.session_state:
    st.session_state.projects = []
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

# Scoring criteria data
FACTORS = {
    "Strategic Factors": {
        "weight": 35,
        "factors": [
            {
                "id": 1,
                "name": "Alignment with Development Goals",
                "weight": 20,
                "min_score": 2,
                "criteria": {
                    4: "Fully aligned - Explicitly supports high-priority goals with clear evidence and measurable targets",
                    3: "Substantially aligned - Supports strategic goals but may not address top priority or lacks direct measurable outcomes",
                    2: "Moderately aligned - Some relevance to strategic goals but lacks strong connection or measurable impact",
                    1: "Marginally aligned - Weak or peripheral link to strategic goals with limited developmental impact",
                    0: "Not aligned - Does not contribute to any strategic goals"
                }
            },
            {
                "id": 2,
                "name": "Economic/Social Returns",
                "weight": 10,
                "min_score": None,
                "criteria": {
                    4: "Very high returns - Significant, measurable economic and/or social benefits",
                    3: "High returns - Notable benefits with some limitations in scale or scope",
                    2: "Moderate returns - Some benefits but impact is limited or not well-documented",
                    1: "Low returns - Minimal, localized, or difficult to quantify benefits",
                    0: "No returns - No discernible economic or social benefits"
                }
            },
            {
                "id": 3,
                "name": "Programmatic Alignment",
                "weight": 5,
                "min_score": None,
                "criteria": {
                    4: "Strongly programmatic - Core component of broader program with significant synergies",
                    3: "Moderately programmatic - Supports broader initiative with less direct integration",
                    2: "Mildly programmatic - Some relevance to other initiatives but operates independently",
                    1: "Weakly programmatic - Loosely related to other initiatives but lacks tangible connections",
                    0: "Stand-alone - Entirely independent with no connection to broader programs"
                }
            }
        ]
    },
    "Implementation Readiness Factors": {
        "weight": 40,
        "factors": [
            {
                "id": 4,
                "name": "Quality of Preparation and Risk Assessment",
                "weight": 20,
                "min_score": 2,
                "criteria": {
                    4: "Excellent - Thorough preparation with high-quality feasibility studies, technical validation, and environmental assessment",
                    3: "Above average - Strong preparatory work but may lack detail in technical validation or environmental aspects",
                    2: "Average - Adequate preparation but includes gaps in technical documentation or environmental considerations",
                    1: "Below average - Weak preparatory work with significant omissions",
                    0: "Poor - No meaningful preparatory work, feasibility studies, or environmental assessment"
                }
            },
            {
                "id": 5,
                "name": "Implementation Feasibility",
                "weight": 10,
                "min_score": 2,
                "criteria": {
                    4: "Highly feasible - All prerequisites secured and institutional capacity fully demonstrated",
                    3: "Feasible with minor gaps - Most requirements in place and capacity largely adequate",
                    2: "Moderately feasible - Several prerequisites incomplete and institutional capacity shows notable gaps",
                    1: "Feasible with major gaps - Key prerequisites missing and institutional capacity significantly limited",
                    0: "Not feasible - Lacks fundamental prerequisites and insufficient institutional capacity"
                }
            },
            {
                "id": 6,
                "name": "Affordability and Financing",
                "weight": 10,
                "min_score": 2,
                "criteria": {
                    4: "Fully affordable - Well within budget, fully financed, with sustainable long-term funding plan",
                    3: "Affordable with minor gaps - Largely within budget with minor financing uncertainties",
                    2: "Moderately affordable - Exceeds budget ceilings or has significant funding uncertainties",
                    1: "Barely affordable - Exceeds budget significantly or has major unresolved financing gaps",
                    0: "Not affordable - Fiscally unsustainable with no clear financing plan"
                }
            }
        ]
    },
    "Community and Political Factors": {
        "weight": 25,
        "factors": [
            {
                "id": 7,
                "name": "Community Needs",
                "weight": 10,
                "min_score": None,
                "criteria": {
                    4: "Fully demand-driven - Directly addresses well-documented critical community needs",
                    3: "Substantially demand-driven - Reflects significant community needs with limited stakeholder input",
                    2: "Moderately demand-driven - Some relevance to community needs but lacks evidence of criticality",
                    1: "Weakly demand-driven - Loosely linked to community needs without clear basis",
                    0: "Not demand-driven - Does not address any identifiable community needs"
                }
            },
            {
                "id": 8,
                "name": "Equity Aspects",
                "weight": 7,
                "min_score": None,
                "criteria": {
                    4: "Highly equitable - Directly targets underserved regions or marginalized communities",
                    3: "Moderately equitable - Benefits underserved groups but also includes better-served regions",
                    2: "Limited equity impact - Some relevance to equity but primarily benefits advantaged regions",
                    1: "Marginally equitable - Minimal impact on reducing disparities",
                    0: "Reinforces inequities - Exacerbates disparities, benefiting privileged groups disproportionately"
                }
            },
            {
                "id": 9,
                "name": "Political Economy",
                "weight": 8,
                "min_score": None,
                "criteria": {
                    4: "Excellent alignment - Politically viable, strengthens governance, promotes institutional reforms",
                    3: "Good alignment - Aligns with governance objectives with some potential to improve institutions",
                    2: "Moderate alignment - Some alignment with governance objectives but lacks significant systemic impact",
                    1: "Weak alignment - Weakly aligned with governance objectives, risks creating conflicts",
                    0: "No alignment or adverse impact - Creates governance challenges or institutional conflicts"
                }
            }
        ]
    }
}

def calculate_weighted_score(scores):
    """Calculate weighted total score"""
    total_score = 0
    total_weight = 0
    
    for category, data in FACTORS.items():
        for factor in data["factors"]:
            factor_id = factor["id"]
            if factor_id in scores and scores[factor_id] is not None:
                weight = factor["weight"] / 100
                total_score += scores[factor_id] * weight
                total_weight += weight
    
    return total_score if total_weight > 0 else 0

def get_classification(weighted_score):
    """Get project classification based on weighted score"""
    if weighted_score >= 3.4:
        return "Highly Recommended", "highly-recommended"
    elif weighted_score >= 2.8:
        return "Recommended", "recommended"
    elif weighted_score >= 2.2:
        return "Conditionally Recommended", "conditional"
    elif weighted_score >= 1.6:
        return "Needs Substantial Revision", "needs-revision"
    else:
        return "Not Recommended", "not-recommended"

def check_minimum_scores(scores):
    """Check if minimum score requirements are met"""
    violations = []
    
    for category, data in FACTORS.items():
        for factor in data["factors"]:
            factor_id = factor["id"]
            min_score = factor["min_score"]
            
            if min_score is not None:
                if factor_id not in scores or scores[factor_id] is None:
                    violations.append(f"{factor['name']}: Score not provided (minimum required: {min_score})")
                elif scores[factor_id] < min_score:
                    violations.append(f"{factor['name']}: Score {scores[factor_id]} is below minimum required ({min_score})")
    
    return violations

def login_page():
    """Login page"""
    st.markdown('<p class="main-header">Punjab ADP Project Scorer</p>', unsafe_allow_html=True)
    st.markdown("### Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        username = st.text_input("Username")
        role = st.selectbox("Role", ["Project Sponsor", "PDB Reviewer", "Administrator"])
        
        if st.button("Login", type="primary"):
            if username:
                st.session_state.current_user = username
                st.session_state.user_role = role
                st.rerun()
            else:
                st.error("Please enter a username")

def dashboard_page():
    """Dashboard page"""
    st.markdown('<p class="main-header">Dashboard</p>', unsafe_allow_html=True)
    
    # User info
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.write(f"**User:** {st.session_state.current_user}")
    with col2:
        st.write(f"**Role:** {st.session_state.user_role}")
    with col3:
        if st.button("Logout"):
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.rerun()
    
    st.markdown("---")
    
    # Metrics
    total_projects = len(st.session_state.projects)
    submitted = len([p for p in st.session_state.projects if p.get('status') == 'Submitted'])
    under_review = len([p for p in st.session_state.projects if p.get('status') == 'Under Review'])
    approved = len([p for p in st.session_state.projects if p.get('status') == 'Approved'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", total_projects)
    with col2:
        st.metric("Submitted", submitted)
    with col3:
        st.metric("Under Review", under_review)
    with col4:
        st.metric("Approved", approved)
    
    st.markdown("---")
    
    # Projects list
    st.markdown("### My Projects")
    
    if st.session_state.projects:
        df = pd.DataFrame([
            {
                "GS Code": p.get('gs_code', 'N/A'),
                "Title": p.get('title', 'N/A'),
                "Status": p.get('status', 'Draft'),
                "Weighted Score": f"{p.get('weighted_score', 0):.2f}",
                "Classification": p.get('classification', 'N/A'),
                "Submitted Date": p.get('submission_date', 'N/A')
            }
            for p in st.session_state.projects
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No projects found. Create your first project using the 'New Project' page.")

def new_project_page():
    """New project submission page"""
    st.markdown('<p class="main-header">New Project Submission</p>', unsafe_allow_html=True)
    
    # Project Information
    st.markdown('<p class="section-header">Project Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        gs_code = st.text_input("Project GS Code*", help="Enter the unique GS code for this project")
        title = st.text_input("Project Title*", help="Enter the full project title")
    with col2:
        department = st.text_input("Department/Organization*", value=st.session_state.current_user)
        submission_date = st.date_input("Submission Date", value=datetime.now())
    
    description = st.text_area("Brief Description*", help="Provide a concise description of the project objectives and outcomes", height=100)
    
    st.markdown("---")
    
    # Scoring Interface
    st.markdown('<p class="section-header">Project Scoring</p>', unsafe_allow_html=True)
    st.info("📋 **Instructions:** Score each factor from 0-4 based on the criteria provided. Factors marked with ⚠️ have minimum score requirements.")
    
    scores = {}
    justifications = {}
    attachments = {}
    
    for category, data in FACTORS.items():
        st.markdown(f"### {category} ({data['weight']}%)")
        
        for factor in data["factors"]:
            factor_id = factor["id"]
            factor_name = factor["name"]
            weight = factor["weight"]
            min_score = factor["min_score"]
            
            # Factor header
            min_indicator = " ⚠️ **Min: 2**" if min_score else ""
            st.markdown(f"**Factor {factor_id}: {factor_name} ({weight}%)**{min_indicator}")
            
            # Scoring criteria in expander
            with st.expander("View Scoring Criteria"):
                for score_val, criteria_text in factor["criteria"].items():
                    st.write(f"**{score_val}:** {criteria_text}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                score = st.select_slider(
                    f"Score for Factor {factor_id}",
                    options=[0, 1, 2, 3, 4],
                    key=f"score_{factor_id}",
                    label_visibility="collapsed"
                )
                scores[factor_id] = score
                
                # Show selected criteria
                st.caption(f"**Selected:** {factor['criteria'][score]}")
            
            with col2:
                justification = st.text_area(
                    f"Justification for Factor {factor_id}*",
                    key=f"just_{factor_id}",
                    help="Provide detailed justification with specific references and evidence",
                    height=100,
                    label_visibility="collapsed",
                    placeholder="Enter your justification here..."
                )
                justifications[factor_id] = justification
            
            # File upload
            uploaded_files = st.file_uploader(
                f"Upload supporting documents for Factor {factor_id}",
                accept_multiple_files=True,
                key=f"files_{factor_id}",
                help="Attach relevant documents (PDF, DOC, DOCX, XLS, XLSX, images)"
            )
            attachments[factor_id] = uploaded_files if uploaded_files else []
            
            st.markdown("---")
    
    # Score Calculation and Summary
    st.markdown('<p class="section-header">Score Summary</p>', unsafe_allow_html=True)
    
    weighted_score = calculate_weighted_score(scores)
    classification, css_class = get_classification(weighted_score)
    violations = check_minimum_scores(scores)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weighted Total Score", f"{weighted_score:.2f}/4.00")
    with col2:
        st.metric("Percentage", f"{(weighted_score/4)*100:.1f}%")
    with col3:
        st.metric("Classification", classification)
    
    # Show score by category
    st.markdown("#### Score Breakdown by Category")
    category_scores = []
    for category, data in FACTORS.items():
        cat_score = 0
        cat_weight = 0
        for factor in data["factors"]:
            if factor["id"] in scores and scores[factor["id"]] is not None:
                cat_score += scores[factor["id"]] * (factor["weight"]/100)
                cat_weight += factor["weight"]/100
        
        category_scores.append({
            "Category": category,
            "Weight": f"{data['weight']}%",
            "Score": f"{cat_score:.2f}",
            "Contribution": f"{cat_score:.2f}"
        })
    
    st.dataframe(pd.DataFrame(category_scores), use_container_width=True)
    
    # Validation warnings
    if violations or weighted_score < 2.2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ **Submission Requirements Not Met**")
        
        if violations:
            st.markdown("**Minimum Score Violations:**")
            for violation in violations:
                st.write(f"- {violation}")
        
        if weighted_score < 2.2:
            st.write(f"- Overall weighted score ({weighted_score:.2f}) is below minimum required (2.2)")
        
        st.markdown("Please address these issues before submission.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Classification description
    st.markdown(f'<div class="score-box {css_class}">', unsafe_allow_html=True)
    st.markdown(f"### Classification: {classification}")
    
    if classification == "Highly Recommended":
        st.write("✅ This project demonstrates exceptional strategic alignment, strong implementation readiness, and clear community benefits. It should be prioritized for ADP inclusion.")
    elif classification == "Recommended":
        st.write("✅ This project shows strong potential. Address any identified gaps before final approval.")
    elif classification == "Conditionally Recommended":
        st.write("⚠️ While meeting minimum thresholds, this project needs significant enhancement in multiple areas.")
    elif classification == "Needs Substantial Revision":
        st.write("❌ This project exhibits fundamental weaknesses requiring major revision.")
    else:
        st.write("❌ This project demonstrates critical deficiencies and requires fundamental reconceptualization.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save as Draft", use_container_width=True):
            if gs_code and title and description:
                project = {
                    'gs_code': gs_code,
                    'title': title,
                    'department': department,
                    'description': description,
                    'submission_date': submission_date.strftime("%Y-%m-%d"),
                    'status': 'Draft',
                    'scores': scores,
                    'justifications': justifications,
                    'weighted_score': weighted_score,
                    'classification': classification,
                    'created_by': st.session_state.current_user,
                    'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.projects.append(project)
                st.success("✅ Project saved as draft!")
            else:
                st.error("Please fill in all required fields marked with *")
    
    with col2:
        submit_disabled = bool(violations) or weighted_score < 2.2
        if st.button("📤 Submit for Review", use_container_width=True, disabled=submit_disabled, type="primary"):
            if gs_code and title and description:
                project = {
                    'gs_code': gs_code,
                    'title': title,
                    'department': department,
                    'description': description,
                    'submission_date': submission_date.strftime("%Y-%m-%d"),
                    'status': 'Submitted',
                    'scores': scores,
                    'justifications': justifications,
                    'weighted_score': weighted_score,
                    'classification': classification,
                    'created_by': st.session_state.current_user,
                    'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.projects.append(project)
                st.success("✅ Project submitted for review!")
                st.balloons()
            else:
                st.error("Please fill in all required fields marked with *")
    
    with col3:
        if st.button("🔄 Reset Form", use_container_width=True):
            st.rerun()

def main():
    """Main application"""
    
    # Check if user is logged in
    if st.session_state.current_user is None:
        login_page()
        return
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "New Project", "View Projects", "Reports"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "Punjab ADP Project Scorer helps evaluate and select projects "
        "for inclusion in the Annual Development Plan using a systematic "
        "9-factor scoring framework."
    )
    
    # Route to pages
    if page == "Dashboard":
        dashboard_page()
    elif page == "New Project":
        new_project_page()
    elif page == "View Projects":
        st.markdown('<p class="main-header">View Projects</p>', unsafe_allow_html=True)
        st.info("🚧 Project viewing and editing functionality coming soon!")
    elif page == "Reports":
        st.markdown('<p class="main-header">Reports</p>', unsafe_allow_html=True)
        st.info("🚧 Reporting and analytics functionality coming soon!")

if __name__ == "__main__":
    main()
