import requests

url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv'
response = requests.get(url)

# Збереження у файл
with open('Associative\\res\\groceries.csv', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Файл groceries.csv збережено локально.")