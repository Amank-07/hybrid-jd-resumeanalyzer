# AI JD → Resume Modifier

A simple Streamlit application that uses AI to analyze your resume against a job description and provides actionable insights to improve your resume's match with the job requirements.

## 🎯 Features

- **PDF Upload Support**: Upload PDF files for both job descriptions and resumes - text is automatically extracted
- **Text Input Support**: Alternatively, paste text directly into text areas
- **ATS Score Analysis**: 
  - **Current ATS Score**: Shows your resume's current ATS compatibility score (0-100%)
  - **Expected ATS Score**: Predicts the ATS score after implementing suggested changes
  - **Improvement Potential**: Displays the potential score increase
- **Missing Skills Detection**: Identifies skills mentioned in the job description that are not clearly evident in your resume
- **Bullet Points Suggestions**: Provides specific, actionable bullet points you should add to your resume
- **ATS Keywords**: Highlights important keywords that ATS (Applicant Tracking System) systems typically look for
- **Fit Percentage**: Calculates a match score (0-100%) with a brief explanation
- **Summary Feedback**: Provides overall assessment and recommendations
- **NLP-Based Metrics (Classical NLP)**:
  - **TF-IDF Keyword Extraction**: Identifies top keywords in JD and resume using TF-IDF
  - **Cosine Similarity**: Computes a numeric similarity score between JD and resume texts
  - **Keyword Overlap & Gaps**: Shows overlapping keywords and important JD keywords missing from the resume

## 📋 Requirements

- Python 3.8 or higher
- OpenAI API key

## 🚀 Setup

1. **Clone or download this project**

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key** (choose one method):
   
   **Option A: Use the helper script** (easiest):
   - **PowerShell**: Double-click `run_app.ps1` or run `.\run_app.ps1`
   - **Command Prompt**: Double-click `run_app.bat` or run `run_app.bat`
   
   **Option B: Set manually**:
   
   **Windows (PowerShell)**:
   ```powershell
   $env:OPENAI_API_KEY="your_api_key_here"
   ```
   
   **Windows (Command Prompt)**:
   ```cmd
   set OPENAI_API_KEY=your_api_key_here
   ```
   
   **Linux/Mac**:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   
   **Note**: If using Option B, you'll need to set this each time you open a new terminal.

5. **Run the application**:
   
   If you used Option A (helper script), the app will start automatically.
   
   If you used Option B, run:
   ```bash
   streamlit run app.py
   ```
   
   The app will open in your default web browser at `http://localhost:8501`

## 📖 Usage

1. **Input Job Description**: 
   - **Option A**: Upload a PDF file containing the job description
   - **Option B**: Copy and paste the complete job description into the text area
2. **Input Resume**: 
   - **Option A**: Upload a PDF file containing your resume
   - **Option B**: Copy and paste your resume text into the text area
3. **Click "Analyze Resume"**: The AI will analyze your resume against the job description
4. **Review Results**: 
   - **Check ATS Scores**: See your current ATS score and expected score after improvements
   - **Check Fit Percentage**: See overall match score
   - **Review Missing Skills**: Identify gaps in your resume
   - **Add Suggested Bullet Points**: Incorporate actionable improvements
   - **Incorporate ATS Keywords**: Add important keywords to boost ATS compatibility

**Note**: PDF uploads are automatically processed to extract text. You can preview the extracted text before analysis.

## 🔧 Configuration

The app uses `gpt-4o-mini` by default for cost efficiency. You can change the model in `app.py`:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Change to "gpt-4o" for better results
    ...
)
```

## 📝 Notes

- The app makes a single LLM API call to generate all analysis results
- Results are returned in JSON format for structured parsing
- The API key is read from environment variables - never hardcoded
- All processing happens locally in your browser (Streamlit) and via API calls

## 🛠️ Troubleshooting

- **"Please set OPENAI_API_KEY environment variable"**: Make sure you've set the environment variable in your terminal before running the app
- **API errors**: Check that your OpenAI API key is valid and you have sufficient credits
- **JSON parsing errors**: If this occurs, the raw response will be shown - this is rare but can happen with certain API responses

## 📄 License

This project is provided as-is for educational and personal use.

