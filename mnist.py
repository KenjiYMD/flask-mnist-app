import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

# app.secret_key = "your_secret_key_here"  
# submitボタンを押した際にエラーが出た場合上の行のコメントアウトを削除し、your_secret_key_hereに任意の文字列（例:aidemy)を指定し、再度アプリケーションを実行してください。

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

import os
from tensorflow.keras.models import load_model

def ensure_model_h5():
    try:
        # まず既存の model.h5 を読み込み（成功すればそのまま返す）
        return load_model('./model.h5')
    except Exception as e:
        print("model.h5 の読み込みに失敗しました。TF2.3 互換形式で作り直します:", e)
        # --- ここから簡易学習して TF2.3 で読める h5 を作成 ---
        from tensorflow.keras import layers, models, datasets
        (x_train, y_train), _ = datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

        model = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # 速さ優先で 1〜3epoch。教材要件に合わせて調整可
        model.fit(x_train, y_train, epochs=1, batch_size=128, verbose=1)
        model.save('./model.h5')   # TF2.3互換のHDF5として保存
        print("model.h5 を作成しました。")
        return model

# ここでモデルを用意
model = load_model('./model.h5')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)