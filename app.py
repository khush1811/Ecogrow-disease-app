import os
import io
import pickle
import string

import bcrypt
import gdown
import numpy as np
import pandas as pd
import requests
import torch
from datetime import datetime
from flask import Flask, redirect, render_template, url_for, request
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from markupsafe import Markup
from PIL import Image
from torchvision import transforms
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError

import config
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.pth")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
FERTILIZER_CSV = os.path.join(BASE_DIR, "Data", "fertilizer.csv")

# -----------------------------------------------------------------------
# DOWNLOAD DISEASE MODEL IF MISSING
# -----------------------------------------------------------------------

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
    print("Downloading disease model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    url = "https://drive.google.com/uc?id=1P7o34U5S0aptxyA50odgqqpjKvtx7Vc7"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Download complete!")

# -----------------------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------------------

# Crop recommendation model
crop_recommendation_model = pickle.load(open(CROP_MODEL_PATH, "rb"))

# Disease classification model
disease_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=torch.device("cpu"))
)
disease_model.eval()

# -----------------------------------------------------------------------
# FLASK APP SETUP
# -----------------------------------------------------------------------

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = config.SECRET_KEY

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -----------------------------------------------------------------------
# DATABASE MODELS
# -----------------------------------------------------------------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class UserAdmin(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"{self.sno} - {self.name}"


with app.app_context():
    db.create_all()

# -----------------------------------------------------------------------
# FORMS
# -----------------------------------------------------------------------

class RegisterForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), Length(min=5, max=20)],
        render_kw={"placeholder": "username"},
    )
    password = PasswordField(
        validators=[InputRequired(), Length(min=5, max=20)],
        render_kw={"placeholder": "password"},
    )
    submit = SubmitField("Register")

    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first():
            raise ValidationError("That username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
    username = StringField(
        validators=[InputRequired(), Length(min=5, max=20)],
        render_kw={"placeholder": "username"},
    )
    password = PasswordField(
        validators=[InputRequired(), Length(min=5, max=20)],
        render_kw={"placeholder": "password"},
    )
    submit = SubmitField("Login")

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def weather_fetch(city_name):
    """Return (temperature_celsius, humidity) for a city, or None on failure."""
    api_key = config.weather_api_key
    if not api_key:
        print("weather_api_key not set in config.py")
        return None

    url = f"http://api.openweathermap.org/data/2.5/weather?appid={api_key}&q={city_name}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get("cod") != "404":
            temp = round(data["main"]["temp"] - 273.15, 2)
            humidity = data["main"]["humidity"]
            return temp, humidity
    except Exception as e:
        print(f"Weather fetch error: {e}")
    return None


def predict_image(img_bytes, model=disease_model):
    """Predict plant disease from raw image bytes."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img_bytes))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# -----------------------------------------------------------------------
# ROUTES — PUBLIC
# -----------------------------------------------------------------------

@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        entry = ContactUs(
            name=request.form["name"],
            email=request.form["email"],
            text=request.form["text"],
        )
        db.session.add(entry)
        db.session.commit()
    return render_template("contact.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html", form=form)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data)
        db.session.add(User(username=form.username.data, password=hashed_pw))
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html", form=form)


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("hello_world"))

# -----------------------------------------------------------------------
# ROUTES — USER
# -----------------------------------------------------------------------

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    return render_template("dashboard.html", title="Dashboard")


@app.route("/crop-recommend")
@login_required
def crop_recommend():
    return render_template("crop.html", title="Crop Recommendation")


@app.route("/crop-predict", methods=["POST"])
@login_required
def crop_prediction():
    title = "Crop Recommendation"
    N = int(request.form["nitrogen"])
    P = int(request.form["phosphorous"])
    K = int(request.form["pottasium"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])
    city = request.form.get("city")

    weather = weather_fetch(city)
    if weather is None:
        return render_template("try_again.html", title=title)

    temperature, humidity = weather
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_recommendation_model.predict(data)[0]
    return render_template("crop-result.html", prediction=prediction, title=title)


@app.route("/fertilizer")
@login_required
def fertilizer_recommendation():
    return render_template("fertilizer.html", title="Fertilizer Suggestion")


@app.route("/fertilizer-predict", methods=["POST"])
@login_required
def fert_recommend():
    title = "Fertilizer Suggestion"
    crop_name = request.form["cropname"]
    N = int(request.form["nitrogen"])
    P = int(request.form["phosphorous"])
    K = int(request.form["pottasium"])

    df = pd.read_csv(FERTILIZER_CSV)
    row = df[df["Crop"] == crop_name].iloc[0]
    n_diff = row["N"] - N
    p_diff = row["P"] - P
    k_diff = row["K"] - K

    diffs = {abs(n_diff): ("N", n_diff), abs(p_diff): ("P", p_diff), abs(k_diff): ("K", k_diff)}
    nutrient, diff = diffs[max(diffs.keys())]

    key_map = {"N": ("NHigh", "Nlow"), "P": ("PHigh", "Plow"), "K": ("KHigh", "Klow")}
    key = key_map[nutrient][0] if diff < 0 else key_map[nutrient][1]

    response = Markup(str(fertilizer_dic[key]))
    return render_template("fertilizer-result.html", recommendation=response, title=title)


@app.route("/disease-predict", methods=["GET", "POST"])
@login_required
def disease_prediction():
    title = "Disease Detection"
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("disease.html", title=title)
        try:
            prediction = predict_image(file.read())
            result = Markup(str(disease_dic[prediction]))
            return render_template("disease-result.html", prediction=result, title=title)
        except Exception as e:
            print(f"Disease prediction error: {e}")
            return render_template("disease.html", title=title, error=str(e))
    return render_template("disease.html", title=title)

# -----------------------------------------------------------------------
# ROUTES — ADMIN
# -----------------------------------------------------------------------

@app.route("/AdminLogin", methods=["GET", "POST"])
def AdminLogin():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for("admindashboard"))
    if form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for("admindashboard"))
    return render_template("adminlogin.html", form=form)


@app.route("/admindashboard")
@login_required
def admindashboard():
    return render_template(
        "admindashboard.html",
        alltodo=ContactUs.query.all(),
        alluser=User.query.all(),
    )


@app.route("/display")
def querydisplay():
    return render_template("display.html", alltodo=ContactUs.query.all())


@app.route("/reg", methods=["GET", "POST"])
def reg():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data)
        db.session.add(UserAdmin(username=form.username.data, password=hashed_pw))
        db.session.commit()
        return redirect(url_for("AdminLogin"))
    return render_template("reg.html", form=form)

# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
