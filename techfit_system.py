print("Initializing imports...", flush=True)
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
print("Importing sklearn...", flush=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import os
print("Imports complete.", flush=True)

# --- Configuration ---
DATASET_PATH = "Techfit_dataset.xlsx"
MODEL_FILES = ['all_models.pkl', 'all_scalers.pkl', 'all_selectors.pkl', 'all_selected_features.pkl', 'all_label_encoders.pkl', 'model_accuracies.pkl']

# --- Helper Functions ---


# --- Enhanced Scoring Engine ---

CAREER_PROFILES = {
    "AI ML Specialist": {
        "academics": ["Mathematics", "Algorithms", "Programming"],
        "skills": ["Logical quotient rating", "coding skills rating"],
        "interests": ["Artificial Intelligence", "Analysis", "Technical"],
        "companies": ["OpenAI", "Google DeepMind", "Anthropic", "Tesla", "NVIDIA"],
        "tools": ["TensorFlow", "PyTorch", "Scikit-learn", "Python", "Jupyter"]
    },
    "API Integration Specialist": {
        "academics": ["Software Engineering", "Programming Concepts", "Computer Networks"],
        "skills": ["coding skills rating"],
        "interests": ["API", "Backend", "Technical"],
        "companies": ["Stripe", "Twilio", "Postman", "MuleSoft"],
        "tools": ["Postman", "REST", "GraphQL", "Python", "Node.js"]
    },
    "Application Developer": {
        "academics": ["Programming Concepts", "Software Engineering", "Algorithms"],
        "skills": ["coding skills rating", "hackathons"],
        "interests": ["Application", "Development", "Technical"],
        "companies": ["Microsoft", "Oracle", "SAP", "ServiceNow"],
        "tools": ["Java", "C#", ".NET", "SQL", "Git"]
    },
    "Big Data Engineer": {
        "academics": ["Mathematics", "Algorithms", "Operating Systems"],
        "skills": ["Logical quotient rating", "coding skills rating"],
        "interests": ["Data", "Analysis", "Technical"],
        "companies": ["Databricks", "Snowflake", "Palantir", "Cloudera"],
        "tools": ["Hadoop", "Spark", "Kafka", "Scala", "AWS"]
    },
    "Business Intelligence Analyst": {
        "academics": ["Mathematics", "Communication skills"],
        "skills": ["Logical quotient rating"],
        "interests": ["Analysis", "Management", "Business"],
        "companies": ["Tableau", "Salesforce", "Microsoft", "Qlik"],
        "tools": ["Power BI", "SQL", "Excel", "Tableau"]
    },
    "Cloud Solutions Architect": {
        "academics": ["Computer Networks", "Operating Systems", "Software Engineering"],
        "skills": ["coding skills rating", "Logical quotient rating"],
        "interests": ["Cloud", "System", "Technical"],
        "companies": ["AWS", "Azure", "Google Cloud", "DigitalOcean"],
        "tools": ["AWS CloudFormation", "Terraform", "Docker", "Kubernetes"]
    },
    "Cyber Security Specialist": {
        "academics": ["Computer Networks", "Operating Systems", "Programming Concepts"],
        "skills": ["Logical quotient rating", "hackathons"],
        "interests": ["Security", "Networks", "Technical"],
        "companies": ["CrowdStrike", "Palo Alto Networks", "Fortinet", "Cisco"],
        "tools": ["Wireshark", "Metasploit", "Nmap", "Linux", "SIEM"]
    },
    "Data Scientist": {
        "academics": ["Mathematics", "Algorithms", "Programming Concepts"],
        "skills": ["Logical quotient rating", "coding skills rating"],
        "interests": ["Data", "Analysis", "Technical"],
        "companies": ["Google", "Meta", "Amazon", "IBM"],
        "tools": ["Python", "R", "Pandas", "SQL", "Tableau"]
    },
    "Database Administrator": {
        "academics": ["Software Engineering", "Operating Systems"],
        "skills": ["Logical quotient rating"],
        "interests": ["Database", "Management", "Technical"],
        "companies": ["Oracle", "MongoDB", "Redis", "Snowflake"],
        "tools": ["Oracle DB", "MySQL", "PostgreSQL", "MongoDB", "Linux"]
    },
    "DevOps Engineer": {
        "academics": ["Operating Systems", "Computer Networks", "Software Engineering"],
        "skills": ["coding skills rating", "hackathons"],
        "interests": ["Cloud", "Operations", "Technical"],
        "companies": ["GitLab", "HashiCorp", "Docker", "Atlassian"],
        "tools": ["Docker", "Kubernetes", "Jenkins", "Git", "Ansible"]
    },
    "Full Stack Developer": {
        "academics": ["Programming Concepts", "Software Engineering", "Algorithms"],
        "skills": ["coding skills rating", "hackathons", "Hours working per day"],
        "interests": ["Web", "Development", "Technical"],
        "companies": ["Airbnb", "Uber", "Netflix", "Spotify"],
        "tools": ["React", "Node.js", "MongoDB", "Express", "TypeScript"]
    },
    "Game Developer": {
        "academics": ["Programming Concepts", "Algorithms", "Mathematics"],
        "skills": ["coding skills rating", "hackathons"],
        "interests": ["Game", "Creative", "Technical"],
        "companies": ["Unity", "Epic Games", "Electronic Arts", "Ubisoft"],
        "tools": ["Unity", "Unreal Engine", "C++", "C#", "Blender"]
    },
    "Information Security Analyst": {
        "academics": ["Computer Networks", "Communication skills"],
        "skills": ["Logical quotient rating"],
        "interests": ["Security", "Analysis", "Technical"],
        "companies": ["FireEye", "Symantec", "Check Point"],
        "tools": ["Splunk", "Wireshark", "Nessus", "Linux"]
    },
    "Network Security Engineer": {
        "academics": ["Computer Networks", "Operating Systems"],
        "skills": ["Logical quotient rating", "hackathons"],
        "interests": ["Networks", "Security", "Technical"],
        "companies": ["Juniper Networks", "Arista", "F5"],
        "tools": ["Firewalls", "VPN", "Cisco IOS", "Wireshark"]
    },
    "Penetration Tester": {
        "academics": ["Computer Networks", "Operating Systems", "Programming Concepts"],
        "skills": ["coding skills rating", "hackathons", "Logical quotient rating"],
        "interests": ["Security", "Hacking", "Technical"],
        "companies": ["HackerOne", "NCC Group", "Rapid7"],
        "tools": ["Burp Suite", "Metasploit", "Nmap", "Kali Linux"]
    },
    "Project Manager": {
        "academics": ["Communication skills", "Software Engineering"],
        "skills": ["Logical quotient rating"],
        "interests": ["Management", "Leadership", "Business"],
        "companies": ["Jira", "Asana", "Trello", "Basecamp"],
        "tools": ["Jira", "Trello", "Microsoft Project", "Excel", "Slack"]
    },
    "Software Developer": {
        "academics": ["Programming Concepts", "Algorithms", "Software Engineering"],
        "skills": ["coding skills rating", "hackathons"],
        "interests": ["Software", "Development", "Technical"],
        "companies": ["Google", "Microsoft", "Amazon", "Apple"],
        "tools": ["Java", "Python", "C++", "Git", "VS Code"]
    },
    "Software Quality Assurance (QA) / Tester": {
        "academics": ["Software Engineering", "Programming Concepts"],
        "skills": ["Logical quotient rating"],
        "interests": ["Testing", "Quality", "Technical"],
        "companies": ["Selenium", "SmartBear", "Katalon"],
        "tools": ["Selenium", "Jira", "TestRail", "Postman"]
    },
    "System Administrator": {
        "academics": ["Operating Systems", "Computer Networks"],
        "skills": ["Logical quotient rating"],
        "interests": ["System", "Administration", "Technical"],
        "companies": ["Red Hat", "Canonical", "Microsoft"],
        "tools": ["Linux", "Windows Server", "Active Directory", "Bash"]
    },
    "Technical Writer": {
        "academics": ["Communication skills", "Software Engineering"],
        "skills": ["Logical quotient rating"],
        "interests": ["Writing", "Documentation", "Technical"],
        "companies": ["Google", "Stripe", "Twilio", "GitLab"],
        "tools": ["Markdown", "Git", "Jira", "Confluence"]
    },
    "Web Developer": {
        "academics": ["Programming Concepts", "Software Engineering"],
        "skills": ["coding skills rating", "hackathons"],
        "interests": ["Web", "Development", "Technical"],
        "companies": ["Automattic", "Vercel", "Netlify"],
        "tools": ["HTML/CSS", "JavaScript", "React", "Git"]
    }
}

def calculate_weighted_score(user_inputs, profile_data):
    """
    Calculates weighted score (0-100) based on:
    - 40% Academic (Matches fields)
    - 30% Skills (Matches skills)
    - 30% Interests (Keywords in flexible fields)
    """
    score = 0.0
    
    # 1. Academic (40%)
    acad_score = 0.0
    relevant_acad = profile_data.get('academics', [])
    count = 0
    
    if relevant_acad:
        for k, v in user_inputs.items():
            if "percentage" in k.lower():
                # Check if k contains any subject kw
                for sub in relevant_acad:
                    if sub.lower() in k.lower():
                        try:
                            val = float(v)
                            acad_score += val
                            count += 1
                        except: pass
        if count > 0:
            acad_score /= count
    else:
        acad_score = 50.0

    score += (acad_score * 0.4)
    
    # 2. Skills (30%)
    skill_score = 0.0
    relevant_skills = profile_data.get('skills', [])
    count = 0
    
    if relevant_skills:
        for sk in relevant_skills:
            if sk in user_inputs:
                try:
                    val = float(user_inputs[sk])
                    # Normalize
                    norm = 0.0
                    if "working per day" in sk.lower():
                        norm = min(val / 12.0, 1.0) * 100
                    elif "hackathons" in sk.lower():
                        norm = min(val / 5.0, 1.0) * 100
                    else: # 1-10
                        norm = (val / 9.0) * 100 # assuming max 9 but let's say 10
                        if norm > 100: norm = 100
                    
                    skill_score += norm
                    count += 1
                except: pass
        if count > 0:
            skill_score /= count
    else:
        skill_score = 50.0
        
    score += (skill_score * 0.3)
    
    # 3. Interests (30%)
    interest_score = 0.0
    hits = 0
    relevant_interests = profile_data.get('interests', [])
    
    # Gather interest text
    interest_text = ""
    for k, v in user_inputs.items():
        if "interest" in k.lower() or "management" in k.lower() or "company" in k.lower():
            interest_text += str(v).lower() + " "
            
    for kw in relevant_interests:
        if kw.lower() in interest_text:
            hits += 1
            
    if hits >= 2: interest_score = 100.0
    elif hits == 1: interest_score = 60.0
    else: interest_score = 20.0
    
    score += (interest_score * 0.3)
    
    return score

def run_score_engine(user_inputs):
    results = []
    for role, profile in CAREER_PROFILES.items():
        s = calculate_weighted_score(user_inputs, profile)
        results.append((role, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def generate_explanation(role, user_inputs):
    profile = CAREER_PROFILES.get(role, {})
    reasons = []
    
    # High Academics
    for sub in profile.get('academics', []):
        for k, v in user_inputs.items():
            if sub.lower() in k.lower() and "percentage" in k.lower():
                try:
                    target = 70.0
                    if float(v) >= target:
                        reasons.append(f"Strong {sub}")
                except: pass
                
    # Interests
    interest_text = ""
    for k, v in user_inputs.items():
        if "interest" in k.lower() or "management" in k.lower():
            interest_text += str(v).lower() + " "
    
    for kw in profile.get('interests', []):
        if kw.lower() in interest_text:
            reasons.append(f"Interest in {kw}")
            
    if not reasons:
        return "Balanced profile match based on skill alignment."
        
    return "Matched due to: " + ", ".join(reasons[:3])

def generate_roadmap(role, user_inputs):
    months = ["Month 1-2", "Month 3-4", "Month 5-6"]
    steps = [
        f"Master core fundamentals for {role}",
        f"Build 2-3 projects related to {role}",
        f"Apply for internships and prepare for interviews"
    ]
    
    profile = CAREER_PROFILES.get(role, {})
    # Check weakest skill
    weakest = None
    min_v = 100
    for sk in profile.get('skills', []):
        if sk in user_inputs:
            try:
                v = float(user_inputs[sk])
                if v < 6:
                    if v < min_v:
                        min_v = v
                        weakest = sk
            except: pass
            
    if weakest:
        clean = weakest.replace("rating", "").replace("skills", "").strip().title()
        steps[0] = f"Focus on improving {clean} skills"
        
    return list(zip(months, steps))

def generate_companies(role):
    return CAREER_PROFILES.get(role, {}).get('companies', ["Google", "Microsoft", "Amazon"])

def generate_tools(role):
    return CAREER_PROFILES.get(role, {}).get('tools', ["Excel", "Git", "Python"])

def identify_skill_gaps(input_df):
    """
    Identifies skill gaps based on thresholds:
    - Academic % < 60
    - Skill Rating (1-10) < 6
    - Hackathons < 1
    """
    gaps = []
    
    # Define thresholds
    # Format: (Col Name, Threshold, Type 'percent'|'rating'|'count')
    # Use exact column names from input_df (which uses 'academic_fields' and 'skill_fields' keys)
    
    # ACADEMIC (Threshold: 60%)
    academic_map = {
        'Acedamic percentage in Operating Systems': 'OS',
        'percentage in Algorithms': 'Algorithms',
        'Percentage in Programming Concepts': 'Programming',
        'Percentage in Software Engineering': 'Soft. Eng',
        'Percentage in Computer Networks': 'Networks',
        'Percentage in Mathematics': 'Maths',
        'Percentage in Communication skills': 'Comm. Skills'
    }
    
    for col, name in academic_map.items():
        if col in input_df.columns and float(input_df[col].iloc[0]) < 60:
            gaps.append(name)
            
    # SKILLS (Threshold: 6/10)
    skill_map = {
        'Logical quotient rating': 'Logical Thinking',
        'coding skills rating': 'Coding',
    }
    
    for col, name in skill_map.items():
        if col in input_df.columns and float(input_df[col].iloc[0]) < 6:
            gaps.append(name)
            
    # EXTRAS
    if 'hackathons' in input_df.columns and float(input_df['hackathons'].iloc[0]) < 1:
        gaps.append("Hackathons")
        
    if not gaps:
        return "None! You are well prepared."
        
    return ", ".join(gaps[:3]) + ("..." if len(gaps) > 3 else "")

def train_and_save_models():
    """Trains models for Job Role, Career Growth, and Salary Growth."""
    print("Loading dataset...")
    try:
        df = pd.read_excel(DATASET_PATH)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dataset not found at {DATASET_PATH}.\nPlease ensure the file is in the same directory.")
        return None

    # Make a copy for preprocessing
    data = df.copy()
    
    # Preprocess the data
    # Handle missing values if any
    data = data.fillna(data.mean(numeric_only=True))
    
    # Encode categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Define targets and features
    targets = {
        'job_role': 'Suggested Job Role',
        #'skill_gaps': 'Skill Gaps', # REMOVED: Using dynamic logic instead
        'career_growth': 'Career_Growth_Potential',
        'salary_growth': 'Expected_Salary_Growth_%'
    }
    
    # Dictionary to store models and preprocessing objects
    models = {}
    scalers = {}
    selectors = {}
    selected_features = {}
    model_accuracies = {}
    
    # Process each target
    for target_name, target_col in targets.items():
        print(f"\\nProcessing {target_name}...")
        
        # Define features (exclude all target columns AND Skill Gaps which is now handled dynamically)
        exclude_cols = list(targets.values()) + ['Skill Gaps']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols]
        y = data[target_col]
        
        # Select features based on the target type
        if target_name == 'salary_growth':
            # For regression, use mutual_info_regression
            selector = SelectKBest(mutual_info_regression, k=15)
        else:
            # For classification, use mutual_info_classif
            selector = SelectKBest(mutual_info_classif, k=15)
        
        X_selected = selector.fit_transform(X, y)
        selected_feature_indices = selector.get_support(indices=True)
        selected_feature_names = X.columns[selected_feature_indices].tolist()
        selected_features[target_name] = selected_feature_names
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define models based on target type
        if target_name == 'salary_growth':
            # Regression models
            models_to_test = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(kernel='rbf'),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
        else:
            # Classification models
            # Classification models with class_weight='balanced' to handle imbalance
            models_to_test = {
                'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
                'Logistic Regression': LogisticRegression(multi_class='ovr', class_weight='balanced', max_iter=1000),
                'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Naive Bayes': GaussianNB()
            }
        
        # Train and evaluate models
        best_model = None
        best_score = -np.inf if target_name == 'salary_growth' else 0
        best_model_name = ""
        
        for name, model in models_to_test.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if target_name == 'salary_growth':
                # For regression, use R2 score
                score = r2_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                print(f"{name} - R2: {score:.4f}")
            else:
                # For classification, use accuracy
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                print(f"{name} - Accuracy: {score:.4f}")
        
        print(f"Best model for {target_name}: {best_model_name} with score {best_score:.4f}")
        
        # Store the best model and preprocessing objects
        models[target_name] = best_model
        scalers[target_name] = scaler
        selectors[target_name] = selector
        model_accuracies[target_name] = best_score
    
    # Save all models and preprocessing objects
    joblib.dump(models, 'all_models.pkl')
    joblib.dump(scalers, 'all_scalers.pkl')
    joblib.dump(selectors, 'all_selectors.pkl')
    joblib.dump(selected_features, 'all_selected_features.pkl')
    joblib.dump(label_encoders, 'all_label_encoders.pkl')
    joblib.dump(model_accuracies, 'model_accuracies.pkl')
    
    return data, models, scalers, selectors, selected_features, label_encoders, model_accuracies

# --- Main Application ---

def main():
    print("Starting Techfit System v2.0...", flush=True)
    # check for model files
    # CHECK FOR UPDATES: If dataset is newer than models, force retrain
    dataset_mtime = 0
    model_mtime = 0
    
    print("Checking file timestamps...", flush=True)
    if os.path.exists(DATASET_PATH):
        dataset_mtime = os.path.getmtime(DATASET_PATH)
        
    if os.path.exists(MODEL_FILES[0]):
        model_mtime = os.path.getmtime(MODEL_FILES[0])
        
    print(f"Dataset mtime: {dataset_mtime}, Model mtime: {model_mtime}", flush=True)

    if dataset_mtime > model_mtime or not all(os.path.exists(f) for f in MODEL_FILES):
        print("Dataset has changed or models missing. Retraining models...", flush=True)
        _encoded_data, models, scalers, selectors, selected_features, label_encoders, model_accuracies = train_and_save_models()
        if _encoded_data is None: return # Error loading data
        # Reload raw data for GUI population (so dropdowns have text, not numbers)
        try:
            data = pd.read_excel(DATASET_PATH)
        except:
            data = _encoded_data # Fallback
            
        print("Model training completed and saved!", flush=True)
    else:
        print("Loading existing models...", flush=True)
        models = joblib.load('all_models.pkl')
        scalers = joblib.load('all_scalers.pkl')
        selectors = joblib.load('all_selectors.pkl')
        selected_features = joblib.load('all_selected_features.pkl')
        label_encoders = joblib.load('all_label_encoders.pkl')
        model_accuracies = joblib.load('model_accuracies.pkl')
        
        try:
            data = pd.read_excel(DATASET_PATH)
        except FileNotFoundError:
            messagebox.showerror("Error", f"Dataset not found at {DATASET_PATH}")
            return

    # Create the main application window
    root = tk.Tk()
    root.title("Tech Career Prediction System")
    root.geometry("1400x900")
    root.state('zoomed') # Maximize window

    # Set a modern color scheme
    bg_color = "#1e3a5f"  # Dark blue
    primary_color = "#00a8ff"  # Light Blue
    secondary_color = "#4cd137"  # Green
    accent_color = "#e84118"  # Red
    text_color = "#f5f6fa"  # White
    panel_bg_solid = "#2c3e50" 

    # --- Scrollable Wrapper ---
    # Container for everything
    main_frame = tk.Frame(root, bg=bg_color)
    main_frame.pack(fill="both", expand=True)

    # Canvas for scrolling
    canvas = tk.Canvas(main_frame, bg=bg_color, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    # Scrollbar
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)

    # Content Frame (scrollable)
    content_frame_container = tk.Frame(canvas, bg=bg_color)
    
    # Window for content
    canvas_window = canvas.create_window((0,0), window=content_frame_container, anchor="nw")
    
    def configure_scroll(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        # Center horizontally
        width = event.width
        canvas.itemconfig(canvas_window, width=width)

    canvas.bind("<Configure>", configure_scroll)
    
    # Mousewheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Header
    header_frame = tk.Frame(content_frame_container, bg=bg_color) # Transparent effect tricky in pure Tkinter, using solid
    header_frame.pack(pady=20, fill="x")
    
    title_label = tk.Label(header_frame, text="TECHFIT CAREER PREDICTION", 
                          font=('Helvetica', 32, 'bold'), bg=bg_color, fg=text_color)
    title_label.pack()
    
    subtitle_label = tk.Label(header_frame, text="Discover your perfect tech career path", 
                             font=('Helvetica', 14, 'italic'), bg=bg_color, fg="#bdc3c7")
    subtitle_label.pack(pady=5)

    # --- Main Input Dashboard (3 Columns) ---
    dashboard_frame = tk.Frame(content_frame_container, bg=bg_color)
    dashboard_frame.pack(padx=20, pady=10, fill="both", expand=True)
    
    # For grid weights to make columns even
    dashboard_frame.grid_columnconfigure(0, weight=1)
    dashboard_frame.grid_columnconfigure(1, weight=1)
    dashboard_frame.grid_columnconfigure(2, weight=1)

    # Dictionary to store input widgets
    input_widgets = {}

    # Helper to create styled sections
    def create_section(parent, title, col):
        frame = tk.LabelFrame(parent, text=title, font=('Helvetica', 14, 'bold'), 
                             bg=panel_bg_solid, fg=primary_color, bd=2, relief="groove")
        frame.grid(row=0, column=col, padx=15, pady=10, sticky="nsew")
        return frame

    # Column 1: Academic Performance
    col1 = create_section(dashboard_frame, "  Academic Performance  ", 0)
    
    academic_fields = [
        'Acedamic percentage in Operating Systems',
        'percentage in Algorithms',
        'Percentage in Programming Concepts',
        'Percentage in Software Engineering',
        'Percentage in Computer Networks',
        'Percentage in Mathematics',
        'Percentage in Communication skills'
    ]

    for i, field in enumerate(academic_fields):
        # Shorten label for cleaner UI
        display_text = field.replace("Percentage in ", "").replace("Acedamic percentage in ", "").title()
        
        lbl = tk.Label(col1, text=display_text, bg=panel_bg_solid, fg=text_color, font=('Helvetica', 10))
        lbl.pack(anchor="w", padx=10, pady=(10, 0))
        
        # Slider container
        s_frame = tk.Frame(col1, bg=panel_bg_solid)
        s_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        val_lbl = tk.Label(s_frame, text="70%", bg=panel_bg_solid, fg=secondary_color, width=4, font=('Helvetica', 10, 'bold'))
        val_lbl.pack(side="right")
        
        def make_update_cmd(lbl):
            return lambda val: lbl.config(text=f"{int(float(val))}%")

        slider = ttk.Scale(s_frame, from_=0, to=100, value=70, command=make_update_cmd(val_lbl))
        slider.pack(side="left", fill="x", expand=True)
        
        input_widgets[field] = slider

    # Column 2: Skills & Capability
    col2 = create_section(dashboard_frame, "  Skills & Capability  ", 1)
    
    # Sliders
    skill_fields = [
        ('Hours working per day', 1, 16, 8),
        ('Logical quotient rating', 1, 10, 5),
        ('hackathons', 0, 10, 2),
        ('coding skills rating', 1, 10, 5)
    ]

    for field, min_v, max_v, default in skill_fields:
        lbl = tk.Label(col2, text=field.title(), bg=panel_bg_solid, fg=text_color, font=('Helvetica', 10))
        lbl.pack(anchor="w", padx=10, pady=(10, 0))
        
        s_frame = tk.Frame(col2, bg=panel_bg_solid)
        s_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        val_lbl = tk.Label(s_frame, text=str(default), bg=panel_bg_solid, fg=secondary_color, width=4, font=('Helvetica', 10, 'bold'))
        val_lbl.pack(side="right")
        
        def make_update_cmd(lbl):
            return lambda val: lbl.config(text=f"{int(float(val))}")

        slider = ttk.Scale(s_frame, from_=min_v, to=max_v, value=default, command=make_update_cmd(val_lbl))
        slider.pack(side="left", fill="x", expand=True)
        
        input_widgets[field] = slider

    # Checkboxes
    tk.Label(col2, text="Personal Traits", bg=panel_bg_solid, fg=primary_color, font=('Helvetica', 11, 'bold')).pack(anchor="w", padx=10, pady=(15, 5))
    
    checkbox_fields = [
        ('Can work long hours?', 'can work long time before system?'),
        ('Self-learning capability?', 'self-learning capability?'),
        ('Team player?', 'worked in teams ever?')
    ]

    for display, field in checkbox_fields:
        var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(col2, text=display, variable=var, bg=panel_bg_solid, fg=text_color, 
                           selectcolor="#2c3e50", activebackground=panel_bg_solid, activeforeground=text_color, font=('Helvetica', 10))
        cb.pack(anchor="w", padx=20, pady=2)
        input_widgets[field] = var

    # Column 3: Interest & Preferences
    col3 = create_section(dashboard_frame, "  Future Goals & Interests  ", 2)
    
    categorical_fields = [
        'Interested subjects',
        'interested career area ',
        'Type of company want to settle in?',
        'Management or Technical',
        'Salary/work',
        'certifications',
        'workshops'
    ]

    for field in categorical_fields:
        matched_col = next((col for col in data.columns if col.strip() == field.strip()), field)
        
        # Clean label
        clean_label = field.strip().replace("interested", "").replace("Interested", "").replace("Type of", "").strip().title()
        if clean_label == "Subjects": clean_label = "Fav Subject"
        if clean_label == "Career Area": clean_label = "Target Area"
        
        lbl = tk.Label(col3, text=clean_label, bg=panel_bg_solid, fg=text_color, font=('Helvetica', 10))
        lbl.pack(anchor="w", padx=10, pady=(10, 0))
        
        if matched_col in data.columns:
            # Use raw values from data since it's loaded fresh and not encoded
            values = sorted(data[matched_col].astype(str).unique().tolist())
            
            combo = ttk.Combobox(col3, values=values, state="readonly")
            if values: combo.current(0)
            combo.pack(fill="x", padx=10, pady=(0, 5))
            input_widgets[field] = combo
            
        else:
            # Fallback for missing columns
            print(f"Warning: Column {field} not found")

    # --- Action Buttons ---
    btn_frame = tk.Frame(content_frame_container, bg=bg_color)
    btn_frame.pack(pady=30)
    
    def show_report_window(results):
        """Creates the Enhanced Report Window (Full Screen Redesign)"""
        report_win = tk.Toplevel(root)
        report_win.title("Techfit Prediction Report")
        report_win.state('zoomed') # Full screen
        report_win.configure(bg="white")
        
        # Main container with margins
        main_container = tk.Frame(report_win, bg="white")
        main_container.pack(fill="both", expand=True, padx=50, pady=20)

        # 1. Header Section
        header_frame = tk.Frame(main_container, bg="white")
        header_frame.pack(fill="x", pady=(10, 30))
        
        tk.Label(header_frame, text="TECHFIT CAREER REPORT", font=('Helvetica', 32, 'bold'), bg="white", fg="#2c3e50").pack()
        tk.Label(header_frame, text=f"Top Match: {results['top_role']}", font=('Helvetica', 16, 'italic'), bg="white", fg=primary_color).pack(pady=(5, 0))

        # Scrollable Content Area (to ensure it fits on all screens)
        canvas = tk.Canvas(main_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Use a window in the canvas that spans the full width of the canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        content = tk.Frame(scrollable_frame, bg="white", padx=20)
        content.pack(fill="both", expand=True)

        # 2. Top Career Matches (Horizontal Row)
        tk.Label(content, text="TOP CAREER MATCHES", font=('Helvetica', 14, 'bold'), bg="white", fg="#7f8c8d").pack(anchor="w", pady=(0, 15))
        
        matches_frame = tk.Frame(content, bg="white")
        matches_frame.pack(fill="x", pady=5)
        
        # Configure grid for equal spacing
        matches_frame.grid_columnconfigure(0, weight=1)
        matches_frame.grid_columnconfigure(1, weight=1)
        matches_frame.grid_columnconfigure(2, weight=1)
        
        careers = results['careers']
        probs = results['probs']
        
        for i in range(len(careers)):
            # #1 gets accent border & highlight
            is_top = (i == 0)
            bg_col = "#e3f2fd" if is_top else "#f7f9fa"
            border_col = primary_color if is_top else "#dcdde1"
            thickness = 2 if is_top else 1
            
            card = tk.Frame(matches_frame, bg=bg_col, bd=0, highlightthickness=thickness, highlightbackground=border_col)
            card.grid(row=0, column=i, sticky="nsew", padx=10, ipady=15)
            
            # Rank
            tk.Label(card, text=f"#{i+1}", font=('Helvetica', 12, 'bold'), bg=primary_color if is_top else "#95a5a6", fg="white", width=30).pack(pady=(0, 10))
            
            # Role Name
            tk.Label(card, text=careers[i], font=('Helvetica', 18, 'bold'), bg=bg_col, fg="#2c3e50", wraplength=300).pack(pady=5)
            
            # Probability
            tk.Label(card, text=f"{probs[i]:.0f}% Match", font=('Helvetica', 16, 'bold'), bg=bg_col, fg=secondary_color).pack(pady=5)
            
            if is_top:
                tk.Label(card, text="★ Best Fit Based on Your Inputs", font=('Helvetica', 10, 'bold'), bg=bg_col, fg=accent_color).pack(pady=(5, 0))

        tk.Frame(content, height=2, bg="#ecf0f1").pack(fill="x", pady=40)

        # 3. Split Section: Why (Left) vs Roadmap (Right)
        split_frame = tk.Frame(content, bg="white")
        split_frame.pack(fill="x")
        split_frame.grid_columnconfigure(0, weight=4) # Left slightly wider
        split_frame.grid_columnconfigure(1, weight=5) # Right
        
        # --- LEFT: Why This Suits You ---
        left_col = tk.Frame(split_frame, bg="white")
        left_col.grid(row=0, column=0, sticky="nsw", padx=(0, 30))
        
        tk.Label(left_col, text="WHY THIS SUITS YOU", font=('Helvetica', 14, 'bold'), bg="white", fg=primary_color).pack(anchor="w", pady=(0, 10))
        
        # Format text as bullets
        why_text = results['why_match'].replace("Matched due to: ", "")
        reasons = why_text.split(", ")
        
        why_box = tk.Frame(left_col, bg="#fdfefe", bd=1, relief="solid", highlightthickness=0)
        why_box.pack(fill="x", ipady=10)
        
        for reason in reasons:
            r_frame = tk.Frame(why_box, bg="#fdfefe")
            r_frame.pack(fill="x", padx=15, pady=5)
            tk.Label(r_frame, text="•", font=('Helvetica', 16, 'bold'), bg="#fdfefe", fg=secondary_color).pack(side="left", anchor="n")
            tk.Label(r_frame, text=reason, font=('Helvetica', 12), bg="#fdfefe", fg="#34495e", wraplength=400, justify="left").pack(side="left", padx=10)

        # Target Companies (Under Why)
        tk.Label(left_col, text="TARGET COMPANIES", font=('Helvetica', 14, 'bold'), bg="white", fg="#8e44ad").pack(anchor="w", pady=(30, 10))
        
        comp_grid = tk.Frame(left_col, bg="white")
        comp_grid.pack(fill="x")
        
        for idx, comp in enumerate(results['companies']):
            c_btn = tk.Label(comp_grid, text=comp, bg="#f0f3f6", fg="#2c3e50", font=('Helvetica', 11, 'bold'), padx=15, pady=10, bd=0)
            c_btn.pack(side="left", padx=(0, 10))

        # --- RIGHT: Skills Roadmap ---
        right_col = tk.Frame(split_frame, bg="white")
        right_col.grid(row=0, column=1, sticky="nsew")
        
        tk.Label(right_col, text="SKILLS ROADMAP (6 MONTHS)", font=('Helvetica', 14, 'bold'), bg="white", fg="#27ae60").pack(anchor="w", pady=(0, 10))
        
        roadmap_container = tk.Frame(right_col, bg="white", bd=0)
        roadmap_container.pack(fill="both", expand=True)
        
        for month, step in results['roadmap']:
            step_frame = tk.Frame(roadmap_container, bg="white", pady=10)
            step_frame.pack(fill="x")
            
            # Month Block
            m_lbl = tk.Label(step_frame, text=month, font=('Helvetica', 11, 'bold'), bg="#ecf0f1", fg="#2c3e50", width=12, pady=5)
            m_lbl.pack(side="left", anchor="n")
            
            # Connector Line (Visual Hack)
            tk.Label(step_frame, text="➜", font=('Helvetica', 14), bg="white", fg=secondary_color).pack(side="left", padx=10)
            
            # Step Text
            tk.Label(step_frame, text=step, font=('Helvetica', 12), bg="white", fg="#34495e", wraplength=450, justify="left").pack(side="left")

        # Recommended Tools
        tools_str = ", ".join(results.get('tools', ["Excel", "Python"])) # Fallback if key missing
        tk.Label(right_col, text=f"Recommended Tools: {tools_str}", font=('Helvetica', 11, 'italic'), bg="white", fg="#7f8c8d").pack(anchor="w", pady=(15, 0))

        tk.Frame(content, height=2, bg="#ecf0f1").pack(fill="x", pady=40)

        # 4. Additional Analytics (3 Cards Row)
        tk.Label(content, text="ADDITIONAL ANALYTICS", font=('Helvetica', 14, 'bold'), bg="white", fg="#7f8c8d").pack(anchor="w", pady=(0, 15))
        
        analytics_frame = tk.Frame(content, bg="white")
        analytics_frame.pack(fill="x")
        analytics_frame.grid_columnconfigure(0, weight=1)
        analytics_frame.grid_columnconfigure(1, weight=1)
        analytics_frame.grid_columnconfigure(2, weight=1)
        
        def create_stat_card(col_idx, icon, title, value, color, detail=None):
            f = tk.Frame(analytics_frame, bg="#fbfcfc", bd=0, highlightthickness=1, highlightbackground="#ecf0f1")
            f.grid(row=0, column=col_idx, sticky="nsew", padx=10, ipady=15)
            
            tk.Label(f, text=f"{icon}  {title}", font=('Helvetica', 10, 'bold'), bg="#fbfcfc", fg="#95a5a6").pack()
            tk.Label(f, text=value, font=('Helvetica', 18, 'bold'), bg="#fbfcfc", fg=color, wraplength=250).pack(pady=10)
            
            if detail:
               detail_w = detail(f)
               detail_w.pack(pady=5)
               
        # Skill Gaps
        create_stat_card(0, "⚠️", "SKILL GAPS", results['skill_gaps'], accent_color)
        
        # Growth
        create_stat_card(1, "📈", "GROWTH POTENTIAL", results['career_growth'], "#27ae60")
        
        # Salary with Progress Bar
        def salary_progress(parent):
            # 59.8% -> ~60%
            val = results['salary_growth']
            p_frame = tk.Frame(parent, bg="#e0e0e0", height=8, width=150)
            p_fill = tk.Frame(p_frame, bg="#8e44ad", height=8, width=int(1.5 * val)) # scaling factor
            p_fill.place(x=0, y=0)
            return p_frame

        create_stat_card(2, "💰", "SALARY OUTLOOK", f"{results['salary_growth']:.1f}% Boost", "#8e44ad", detail=salary_progress)

        # Footer Actions
        action_frame = tk.Frame(content, bg="white")
        action_frame.pack(pady=50)
        
        tk.Button(action_frame, text="MODIFY INPUTS & RECALCULATE", command=report_win.destroy,
                 font=('Helvetica', 14, 'bold'), bg=primary_color, fg="white", padx=40, pady=15, bd=0, cursor="hand2").pack()
        
        tk.Label(content, text="CONFIDENTIAL REPORT | GENERATED BY TECHFIT AI", font=('Arial', 9), bg="white", fg="#bdc3c7").pack(side="bottom", pady=20)


    def on_predict():
        try:
            # Collect input values
            input_data = {}
            for field in academic_fields + [f[0] for f in skill_fields]:
                input_data[field] = input_widgets[field].get()
            
            for display, field in checkbox_fields:
                val = input_widgets[field].get()
                
                # Check if this field appears in label_encoders
                # If so, it expects a string. Else it likely expects a number.
                if field in label_encoders:
                    # It expects a string label.
                    # We default to 'yes'/'no' which is common, but let's check classes if possible or handle error
                    input_data[field] = 'yes' if val else 'no' 
                else:
                    # It expects a number (0/1)
                    input_data[field] = 1 if val else 0
            
            for field in categorical_fields:
                if field in input_widgets:
                    input_data[field] = input_widgets[field].get()
                else: 
                    # If fallback happened and widget missing
                    input_data[field] = "Unknown" 

            # Create input df
            input_df = pd.DataFrame([input_data])
            
            # Encode
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        val = input_df[col].iloc[0]
                        # Handling generic mismatches by checking LE classes
                        if str(val) not in le.classes_:
                            # Try mapped bool
                            if str(val).lower() == 'true': val = 'yes'
                            elif str(val).lower() == 'false': val = 'no'
                        
                        input_df[col] = le.transform([str(val)])
                    except:
                        input_df[col] = 0 # Fallback
            
            # Align cols (must match training: exclude targets AND 'Skill Gaps')
            # Use same exclusion logic as in train_and_save_models() line 117
            targets_list = ['Suggested Job Role', 'Expected_Salary_Growth_%', 'Career_Growth_Potential']
            exclude_cols = targets_list + ['Skill Gaps']  # Match training exclusion
            all_feature_cols = [col for col in data.columns if col not in exclude_cols]
            for col in all_feature_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[all_feature_cols]

            # Model Prediction Logic
            results = {}
            
            # 1. Job Role (Enhanced Weighted Scoring)
            scored_roles = run_score_engine(input_data)
            top_3 = scored_roles[:3]
            
            results['careers'] = [r[0] for r in top_3]
            results['probs'] = [r[1] for r in top_3]
            
            # Generate detailed metadata for the Top 1 match
            top_role = top_3[0][0]
            results['why_match'] = generate_explanation(top_role, input_data)
            results['roadmap'] = generate_roadmap(top_role, input_data)
            results['companies'] = generate_companies(top_role)
            results['tools'] = generate_tools(top_role)
            results['top_role'] = top_role

            # 2. Skill Gaps (DYNAMIC)
            results['skill_gaps'] = identify_skill_gaps(input_df)

            # 3. Career Growth (Keep ML or heuristic? Keeping ML for now if available, else static)
            try:
                mdl = models['career_growth']
                scl = scalers['career_growth']
                feats = selected_features['career_growth']
                X_in = scl.transform(input_df[feats])
                pred = mdl.predict(X_in)[0]
                results['career_growth'] = label_encoders['Career_Growth_Potential'].inverse_transform([pred])[0]
            except:
                results['career_growth'] = "High" # Fallback
            
            # 4. Salary (Keep ML)
            try:
                mdl = models['salary_growth']
                scl = scalers['salary_growth']
                feats = selected_features['salary_growth']
                X_in = scl.transform(input_df[feats])
                pred = mdl.predict(X_in)[0]
                results['salary_growth'] = float(pred)
            except:
                results['salary_growth'] = 15.5 # Fallback

            # Show Report
            show_report_window(results)

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            import traceback
            traceback.print_exc()

    btn = tk.Button(btn_frame, text="GENERATE PREDICTION REPORT", command=on_predict, 
                   font=('Helvetica', 16, 'bold'), bg=secondary_color, fg="white", 
                   padx=40, pady=15, bd=0, activebackground="#44bd32", cursor="hand2")
    btn.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
