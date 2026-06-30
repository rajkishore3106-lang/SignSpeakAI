from flask import Flask, render_template, request, jsonify, redirect, session
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
import base64

from predict import predict_image
from database import db, User

app = Flask(__name__)

# -----------------------
# Flask Configuration
# -----------------------
app.config["SECRET_KEY"] = "signspeak123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()

# -----------------------
# Home
# -----------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------
# Detection
# -----------------------
@app.route("/detect")
def detect():
    return render_template("detect.html")

# -----------------------
# About
# -----------------------
@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------
# Contact
# -----------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# -----------------------
# Register
# -----------------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            return "Email already registered!"

        hashed_password = generate_password_hash(password)

        user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(user)
        db.session.commit()

        return redirect("/login")

    return render_template("register.html")

# -----------------------
# Login
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user"] = user.name

            return redirect("/")

        return "Invalid Email or Password"

    return render_template("login.html")

# -----------------------
# Logout
# -----------------------
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/login")

# -----------------------
# AI Prediction
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    image_data = base64.b64decode(data.split(",")[1])

    np_arr = np.frombuffer(image_data, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    prediction = predict_image(frame)

    return jsonify({
        "prediction": prediction
    })

# -----------------------
# Run Flask
# -----------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)