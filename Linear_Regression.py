import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Création du DataFrame
df = pd.DataFrame({
    "surface": [50, 60, 70, 100],
    "prix": [40, 60, 80, 140],
})

# Affichage des premières lignes du DataFrame
print(df.head())

# Affichage de la forme du DataFrame
print(df.shape)

# Vérification des valeurs manquantes dans le DataFrame
print(df.isnull().sum())

# Séparation des variables explicatives (features) et de la variable cible (target)
x = df[['surface']].values
y = df['prix'].values

# Séparation des données en ensemble d'entraînement et ensemble de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(x_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(x_test)

# Affichage des prédictions et des valeurs réelles
print("Prédictions:", y_pred)
print("Valeurs réelles:", y_test)

# Tracé du graphique
plt.scatter(x_train, y_train, color='blue', label='Données d\'entraînement')
plt.plot(x_train, model.predict(x_train), color='red', label='Régression linéaire')
plt.title('Régression linéaire entre la surface et le prix')
plt.xlabel('Surface')
plt.ylabel('Prix')
plt.legend()
plt.show()

