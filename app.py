import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
SKILLS = [
    "python", "java", "c", "c++", "r", "sql", "html", "css", "javascript",
    "react", "node.js", "flask", "django", "git", "github", "pandas",
    "numpy", "scikit-learn", "machine learning", "deep learning", "nlp",
    "data analysis", "data structures", "algorithms", "excel", "power bi",
    "tableau", "mysql", "mongodb", "docker", "linux", "tensorflow",
]
def find_skills(text):
    """Return the set of known skills that appear in the text."""
    text = text.lower()
    found = set()
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, text):
            found.add(skill)
    return found
def text_similarity(resume, job):
    """Cosine similarity between two texts, as a percentage (0 to 100)."""
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([resume, job])
        return cosine_similarity(matrix[0], matrix[1])[0][0] * 100
    except ValueError:
        return 0.0
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        resume = request.form["resume"].strip()
        job = request.form["job"].strip()

        if not resume or not job:
            error = "Please fill in both boxes."
        else:
            resume_skills = find_skills(resume)
            job_skills = find_skills(job)
            matched = sorted(resume_skills & job_skills)
            missing = sorted(job_skills - resume_skills)

            similarity = text_similarity(resume, job)

            if job_skills:
                skill_score = len(matched) / len(job_skills) * 100
                final = 0.6 * skill_score + 0.4 * similarity
            else:
                final = similarity

            result = {
                "score": round(final),
                "similarity": round(similarity),
                "matched": matched,
                "missing": missing,
            }

    return render_template("index.html", result=result, error=error)
if __name__ == "__main__":
    app.run(debug=True)