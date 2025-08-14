# ---- env vars for low memory (must be set before importing TensorFlow) ----
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ----------------------- standard imports -----------------------
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------------- app/config -----------------------------
classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")  # flash で必要
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 無ければ作成

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- model loader (TF2.3 互換) -----------------
def ensure_model_h5():
    """Try to load model.h5; if incompatible/missing, build a tiny model and save it (fits 512MB)."""
    from tensorflow.keras.models import load_model
    try:
        return load_model("./model.h5")
    except Exception as e:
        print("model.h5 の読み込みに失敗。軽量モデルで再作成します:", e)
        # 超軽量でメモリ節約して学習（512MBのFreeプラン想定）
        from tensorflow.keras import layers, models, datasets, backend as K
        (x_train, y_train), _ = datasets.mnist.load_data()

        # 使う枚数を絞ってメモリ削減
        x_train = x_train[:10000]
        y_train = y_train[:10000]

        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
        model.save("./model.h5")  # TF2.3 互換のHDF5
        del x_train, y_train
        K.clear_session()
        print("model.h5 を作成しました。")
        return model

# アプリ起動時にモデルを用意
model = ensure_model_h5()

# ----------------------- routes -----------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 画像を読み込み→正規化→(1, 28, 28, 1) へ
            img = image.load_img(filepath, color_mode="grayscale",
                                 target_size=(image_size, image_size))
            img = image.img_to_array(img).astype("float32") / 255.0
            data = np.expand_dims(img, axis=0)

            # 推論
            result = model.predict(data)[0]
            predicted = int(result.argmax())
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")

# ----------------------- entry point -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)