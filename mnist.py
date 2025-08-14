# ---- low-memory env (set before TF import) ----
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------------- config -----------------------
classes = [str(i) for i in range(10)]
image_size = 28
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------- model (Colabと同一アーキ) -------------------
def build_model():
    m = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

def load_npz_weights_model():
    """Colabで保存した .npz 重みを読み込む（学習はしない）"""
    weights_path = "./mnist_weights_v1.npz"  # ここに置いたファイル名
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"{weights_path} が見つかりません。リポジトリ直下に配置してください。")

    m = build_model()
    z = np.load(weights_path, allow_pickle=True)
    # np.savez のデフォルト名 arr_0, arr_1, ... の順に並べ直して set_weights
    ordered = [z[f"arr_{i}"] for i in range(len(z.files))]
    m.set_weights(ordered)
    print(f"Loaded weights from {weights_path} (arrays={len(ordered)})")
    return m

model = load_npz_weights_model()

# ----------------------- routes -----------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません"); return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません"); return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像を読み込み（28x28, グレースケール）
            img = image.load_img(filepath, color_mode="grayscale",
                                 target_size=(image_size, image_size))
            img = image.img_to_array(img).astype("float32")

            # ★ 白地/黒地どちらでもOK：背景が白っぽければ反転
            if img.mean() > 128:
                img = 255.0 - img

            img /= 255.0
            data = np.expand_dims(img, axis=0)  # (1,28,28,1)

            probs = model.predict(data)[0]
            pred = int(probs.argmax())
            return render_template("index.html", answer=f"これは {classes[pred]} です")

    return render_template("index.html", answer="")

# ----------------------- entry point -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)