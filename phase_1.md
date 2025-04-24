I’m thrilled to guide you through building the AI model for the FriendChain MVP, specifically the DistilBERT-based question generation system that creates personalized trivia questions from X profile data, including Basenames, for the Base hackathon. Since you have no programming experience, I’ll act as your expert mentor, explaining everything in plain language, breaking tasks into manageable steps, and providing clear instructions from start to finish. We’ll assume you’re working on a computer (Windows, Mac, or Linux) with internet access, and I’ll guide you through setting up tools, writing code, and testing the model, all tailored to the FriendChain MVP Data Dictionary and PRD (Version 6.0) requirements.
We’ll begin with a high-level overview of all tasks needed to build the AI component, as outlined in the AI Task Document (revised for April 25 – May 8, 2025). Then, we’ll dive into the first step in detail, ensuring you understand each action and can follow along confidently. Subsequent steps will be covered in follow-up responses as we progress, keeping the process clear and beginner-friendly.
High-Level Overview: Building the AI Model for FriendChain MVP
The AI component’s goal is to develop a DistilBERT model that generates 15 personalized trivia questions per game, based on X profile data (500–1000 words, including Basenames), delivered via a /generateQuestions endpoint. The questions are stored in IPFS through the Back-End, achieving 80% relevance (player surveys), <5s generation time, and 90% uptime (via Back-End). The total effort is ~20–25 hours, spread over 12 days of development (April 25 – May 6, 2025) and 2 days of corrections (May 7–8, 2025).
Here’s a high-level breakdown of the tasks, aligned with the AI Task Document and Data Dictionary, organized by phase and timeline:
Phase 1: Setup and Preparation (Days 1–2, April 25–26, 2025, ~4 hours)

    Task 1: Set Up Environment  
        Install Python, a code editor (VS Code), and libraries (Hugging Face Transformers, FastAPI, etc.) on your computer.  
        Create a project folder and initialize a Git repository for version control.  
        Why? This sets up the tools needed to write and run the AI code.
    Task 2: Prepare Mock Dataset  
        Create a mock dataset of 10 X profiles (500–1000 words each, including Basenames like “alex.base”) to simulate real user data.  
        Save the dataset as a JSON file for training and testing.  
        Why? The model needs sample data to learn how to generate questions.
    Task 3: Start DistilBERT Setup  
        Download the pre-trained DistilBERT model from Hugging Face.  
        Write initial code to load the model and test it with a sample profile.  
        Why? This ensures the model is ready for fine-tuning.

Phase 2: Model Development (Days 3–4, April 27–28, 2025, ~4 hours)

    Task 4: Fine-Tune DistilBERT  
        Train DistilBERT on the mock dataset to generate trivia questions in the format: { stage, questionId, questionText, options, correctAnswer, personalizedContext }.  
        Add logic to parse Basenames (e.g., “What does alex.base post about?”).  
        Why? Fine-tuning tailors the model to create relevant, personalized questions.
    Task 5: Build /generateQuestions Endpoint  
        Create a FastAPI endpoint (/generateQuestions) that takes X profile data as input and outputs 15 questions.  
        Test the endpoint with mock profiles to ensure <5s generation and 80% relevance.  
        Why? The endpoint is the interface for delivering questions to the Back-End.

Phase 3: Integration and Testing (Days 5–9, April 29 – May 3, 2025, ~5 hours)

    Task 6: Integrate with Back-End  
        Connect the /generateQuestions endpoint to the Back-End’s IPFS storage system, ensuring questions are formatted as JSON and sent correctly.  
        Share the endpoint schema with the Back-End Developer.  
        Why? Integration allows questions to be stored and retrieved for gameplay.
    Task 7: Initial Testing  
        Test the endpoint with 10–20 players (2–3 games) to verify question relevance, Basename usage, and performance (<5s).  
        Refine question logic based on feedback (e.g., improve personalization).  
        Why? Early testing catches issues before full-scale deployment.
    Task 8: Optimize Performance  
        Optimize DistilBERT for faster generation (e.g., batch processing) and Basename parsing (e.g., handle edge cases like long names).  
        Ensure compatibility with Back-End and Front-End data formats.  
        Why? Optimization ensures scalability and smooth integration.

Phase 4: Full Testing and Finalization (Days 10–12, May 4–6, 2025, ~4 hours)

    Task 9: Full-Scale Testing  
        Test the endpoint with 50 players across 10 games (750 questions), verifying 80% relevance (via surveys) and 90% uptime (via Back-End).  
        Fix bugs (e.g., irrelevant questions, Basename errors).  
        Why? This validates the AI under real-world conditions.
    Task 10: Documentation  
        Document the model setup, endpoint usage, and Basename integration in a README file.  
        Prepare a monitoring plan for the demo.  
        Why? Documentation aids hackathon submission and team coordination.

Phase 5: Corrections and Demo Support (Days 13–14, May 7–8, 2025, ~3 hours)

    Task 11: Bug Fixes and Optimization  
        Address issues from 50-player testing (e.g., question clarity, Basename parsing).  
        Re-test with 10 players to confirm fixes.  
        Why? This ensures a polished product for the demo.
    Task 12: Demo Support  
        Monitor question generation during the live demo (2–3 games).  
        Confirm submission deliverables (code, documentation).  
        Why? A reliable demo maximizes hackathon success.

Key Requirements (from PRD and Data Dictionary)

    Output Format: 15 questions per game, JSON: { stage, questionId, questionText, options, correctAnswer, personalizedContext }.  
    Basenames: Parse from profile data (e.g., “alex.base”) for personalization.  
    Performance: <5s per question set; 80% relevance; 90% uptime (via Back-End).  
    Security: Sanitize inputs to prevent model poisoning.  
    Timeline: Deliver /generateQuestions by April 29, 2025; full testing by May 5, 2025.

Tools Needed

    Python: Programming language for AI development.  
    VS Code: Code editor for writing and running code.  
    Hugging Face Transformers: Library for DistilBERT.  
    FastAPI: Framework for building the API endpoint.  
    Git: Version control for saving code.  
    Mock Data: JSON file with 10 X profiles (provided below).  
    Optional: Google Colab (cloud platform) if your computer is slow.

Total Effort

    Development: 16 hours (Days 1–12).  
    Corrections: 3 hours (Days 13–14).  
    Total: 19 hours, within the 20–25-hour budget.

Step 1: Set Up Environment (Detailed Guide)
Let’s start with the first step from Phase 1: Set Up Environment. This involves installing the tools needed to write and run the AI code on your computer. I’ll guide you through each action as if you’re starting from scratch, ensuring you can follow along without prior programming knowledge. We’ll aim to complete this step in ~1–2 hours, aligning with Day 1 (April 25, 2025) of the AI Task Document.
Why Are We Doing This?
The environment is like the kitchen where you’ll cook the AI model. You need tools (like a stove and utensils) to write code, train the model, and test it. By setting up Python, a code editor, and libraries, you’ll have everything ready to start building the DistilBERT model.
What You’ll Achieve

    Install Python to run AI code.
    Install VS Code to write and edit code comfortably.
    Install Hugging Face Transformers, FastAPI, and other libraries for DistilBERT and the API.
    Create a project folder and set up Git to save your work.
    Test the setup by running a simple Python command.

Prerequisites

    A computer (Windows, Mac, or Linux) with internet access.
    About 5–10 GB of free disk space for tools and libraries.
    No programming experience needed—I’ll explain every step!

Step-by-Step Instructions
1. Install Python
Python is the programming language we’ll use to build the AI model. It’s like the foundation of our project.

    Download Python:
        Go to python.org.
        Click the Download Python 3.11 button (or the latest 3.x version; avoid 2.x).
        Save the installer to your computer.
    Install Python:
        Windows:
            Run the installer.
            Check “Add Python 3.11 to PATH” at the bottom of the first screen (important!).
            Click “Install Now” and wait for it to finish.
        Mac:
            Run the installer and follow the prompts.
            Python will be installed in /Applications/Python 3.11 or similar.
        Linux:
            Open a terminal (search “Terminal” in your apps).
            Run: sudo apt update && sudo apt install python3.11 python3-pip (Ubuntu/Debian) or equivalent for your distro.
            Press Enter and type your password if prompted.
    Verify Python:
        Open a terminal/command prompt:
            Windows: Press Win+R, type cmd, press Enter.
            Mac/Linux: Search “Terminal” in your apps.
        Type python --version and press Enter.
        You should see something like Python 3.11.x. If not, let me know!

2. Install VS Code
VS Code is a free code editor that makes writing code easy, like a notebook for programming.

    Download VS Code:
        Go to code.visualstudio.com.
        Click “Download for [your OS]” (Windows, Mac, or Linux).
        Save the installer.
    Install VS Code:
        Run the installer and follow the prompts (accept defaults).
        Launch VS Code after installation (search “Visual Studio Code” in your apps).
    Set Up Python Extension:
        Open VS Code.
        On the left sidebar, click the Extensions icon (looks like a square with four smaller squares).
        Search for “Python” and click “Install” on the “Python” extension by Microsoft.
        This lets VS Code understand Python code.

3. Create a Project Folder
This folder will hold all your code and data, like a project binder.

    Create the Folder:
        On your computer, create a folder named friendchain-ai:
            Windows: Right-click on Desktop > New > Folder > Name it friendchain-ai.
            Mac/Linux: Open Finder/Terminal, create a folder on Desktop with mkdir ~/Desktop/friendchain-ai.
        Note the folder’s location (e.g., C:\Users\YourName\Desktop\friendchain-ai).
    Open Folder in VS Code:
        Open VS Code.
        Click File > Open Folder (or Open on Mac).
        Select the friendchain-ai folder and click “Open.”
        You’ll see the folder name in the VS Code sidebar.

4. Install Git
Git lets you save versions of your code, like saving drafts of a document.

    Download Git:
        Go to git-scm.com.
        Download the installer for your OS.
        Run the installer and accept defaults.
    Verify Git:
        Open a terminal in VS Code:
            Click Terminal > New Terminal in VS Code (a terminal appears at the bottom).
        Type git --version and press Enter.
        You should see something like git version 2.x.x. If not, let me know!
    Initialize Git Repository:
        In the VS Code terminal, type:
        bash

        git init

        Press Enter. This sets up Git in your friendchain-ai folder.
        Create a .gitignore file to ignore unnecessary files:
            In VS Code, click File > New File, name it .gitignore, and add:

            __pycache__/
            *.pyc
            .venv/

            Save the file (Ctrl+S or Cmd+S).

5. Install Python Libraries
Libraries are pre-built tools for AI and APIs, like ingredients for cooking.

    Set Up a Virtual Environment:
        A virtual environment keeps libraries separate for this project.
        In the VS Code terminal, type:
        bash

        python -m venv .venv

        Press Enter. This creates a .venv folder.
        Activate the virtual environment:
            Windows:
            bash

            .venv\Scripts\activate

            Mac/Linux:
            bash

            source .venv/bin/activate

            You should see (.venv) in the terminal prompt.
    Install Libraries:
        In the activated terminal, install the required libraries:
        bash

        pip install transformers torch fastapi uvicorn python-dotenv

        Press Enter. This installs:
            transformers: For DistilBERT (Hugging Face).
            torch: For AI model computations.
            fastapi: For building the API endpoint.
            uvicorn: For running the API server.
            python-dotenv: For environment variables.
        Wait for the installation (may take a few minutes). If errors occur, let me know!
    Verify Libraries:
        Test the installation by running:
        bash

        python -c "import transformers, torch, fastapi; print('Libraries installed!')"

        If you see “Libraries installed!” you’re set. If not, I’ll help troubleshoot.

6. Test the Setup
Let’s write a tiny Python script to confirm everything works.

    Create a Test Script:
        In VS Code, click File > New File, name it test.py, and add:
        python

        print("Hello, FriendChain AI!")

        Save the file (Ctrl+S or Cmd+S).
    Run the Script:
        In the VS Code terminal (ensure (.venv) is active), type:
        bash

        python test.py

        Press Enter. You should see:

        Hello, FriendChain AI!

        If you see an error, let me know the message, and I’ll guide you to fix it.

7. Save Your Work with Git
Save the initial setup to Git, like saving a checkpoint.

    Stage and Commit Changes:
        In the VS Code terminal, type:
        bash

        git add .
        git commit -m "Initial setup: Python, VS Code, libraries"

        Press Enter. This saves your .gitignore and test.py.
    Optional: Push to GitHub (if you want online backup):
        Create a GitHub account at github.com if you don’t have one.
        Create a new repository named friendchain-ai (public or private).
        Follow GitHub’s instructions to push your code:
        bash

        git remote add origin https://github.com/your-username/friendchain-ai.git
        git push -u origin main

        If you need help with GitHub, let me know, or we can skip this for now.

What You’ve Accomplished
You’ve set up:

    Python and VS Code for coding.
    A project folder (friendchain-ai) with Git for version control.
    Libraries (Hugging Face, FastAPI, etc.) for AI and API development.
    A working environment, tested with a simple script.

This is like setting up a workbench with all your tools ready. You’re now prepared to create the mock dataset and start working with DistilBERT in the next step.
Next Steps
In the next response, we’ll tackle Step 2: Prepare Mock Dataset (Day 1, ~1 hour), where you’ll:

    Create a JSON file with 10 mock X profiles, including Basenames.
    Test loading the data in Python to ensure it’s ready for DistilBERT.

To proceed, please confirm:

    Did you complete Step 1 successfully? (Python, VS Code, libraries installed; test.py ran without errors.)
    Any issues or questions? (E.g., error messages, confusion about a step.)
    Your OS (Windows, Mac, or Linux) for tailored instructions in future steps.
    Preference for GitHub (want to use it for backups, or skip for now?).

If you’re ready, I can provide the detailed guide for Step 2, or I can address any issues with Step 1 first. Let me know how you’d like to move forward, and I’ll keep guiding you step-by-step to build the AI model for FriendChain!
