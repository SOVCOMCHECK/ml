from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import torch
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)
api = Api(app, version='1.0', title='Category Prediction API',
          description='API for predicting categories from product checks')

# Пространство имён для Swagger
ns = api.namespace('predict', description='Prediction operations')

# Модель данных для Swagger
check_model = api.model('Check', {
    'check': fields.List(fields.String, required=True, description='List of products in the check')
})

# Функция для удаления размерностей
def remove_dimensions(text):
    """
    Удаляет числа и единицы измерения из текста.
    """
    text = re.sub(r'\d+[\w\s]*', '', text).strip()
    return text

# Функция для препроцессинга текста
def preprocess_check(check):
    """
    Функция для препроцессинга текста:
    - Удаление размерностей
    - Удаление лишних символов
    - Токенизация
    - Удаление стоп-слов
    - Лемматизация
    - Удаление латинских символов
    """
    result = []
    
    for product in check:
        # Удаление всех специальных символов (оставляем только кириллические буквы и пробелы)
        product = re.sub(r'[^\u0400-\u04FF\s]', '', product.lower()).strip()
        
        # Токенизация
        tokens = word_tokenize(product)
        
        # Удаление стоп-слов
        stop_words = set(stopwords.words('russian'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Лемматизация
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Сборка токенов обратно в строку
        processed_product = ' '.join(tokens)
        
        # Удаление размерностей
        processed_product = remove_dimensions(processed_product)
        
        result.append(processed_product)
    
    return result

# Загрузка модели
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Параметры модели
input_size = 1000  # Размерность TF-IDF
num_classes = 19
model = NeuralNet(input_size, num_classes)

# Загрузка state_dict модели
model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))
model.eval()

# Загрузка TfidfVectorizer и LabelEncoder
with open("./model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("./model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Маршрут для предсказания
@ns.route('/')
class Predict(Resource):
    @ns.expect(check_model)
    @ns.response(200, 'Success')
    @ns.response(400, 'Invalid input')
    def post(self):
        """
        Предсказать категорию по чеку
        """
        try:
            # Получение данных из POST-запроса
            data = request.json
            check = data.get("check", [])
            
            if not check:
                return {"error": "Empty check provided"}, 400
            
            # Препроцессинг чека
            processed_check = preprocess_check(check)
            
            # TF-IDF векторизация
            X = vectorizer.transform(processed_check).toarray()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Предсказание
            model.eval()
            with torch.no_grad():
                output = model(X_tensor)
                _, predicted = torch.max(output, 1)
                predicted_category = label_encoder.inverse_transform(predicted.numpy())
            
            # Возвращение результата
            return {"predicted_category": predicted_category.tolist()[0]}, 200
        
        except Exception as e:
            return {"error": str(e)}, 500

# Запуск сервера
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)