from flask import Flask, request, render_template, send_file
import os
import report_generator

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Update paths in report_generator
        report_generator.EXCEL_FILE = filepath
        report_generator.PDF_FILE = os.path.join(UPLOAD_FOLDER, "analysis_report.pdf")

        # Run report generation
        report_generator.generate_report()

        return send_file(report_generator.PDF_FILE, as_attachment=True)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)