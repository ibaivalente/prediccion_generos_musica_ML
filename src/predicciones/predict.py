import pandas as pd
import pickle
import datetime
dia_de_hoy = datetime.datetime.now().date()
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

# Modelo de filtro preliminar
modelo_path_clasi = os.path.join(current_dir, '../modelos/final_clasi.pkl')
with open(modelo_path_clasi, 'rb') as file:
    final_clasi = pickle.load(file)

# Modelo de clasificación de géneros buenos
modelo_path_buenos = os.path.join(current_dir, '../modelos/final_buenos.pkl')
with open(modelo_path_buenos, 'rb') as file:
    final_buenos = pickle.load(file)

# Modelo de clasificación de géneros malos
modelo_path_malos = os.path.join(current_dir, '../modelos/final_malos.pkl')
with open(modelo_path_malos, 'rb') as file:
    final_malos = pickle.load(file)

# Predicción de géneros de nuestro dataframe completo

os.chdir('C:\\Users\\Ibai Valente Lavado\\OneDrive\\Escritorio\\Data Science\\yo\\DS_TheBridgeBBK_SBIL2023\\3-Machine_Learning\\Entregas\\ML_project\\src\\predicciones')

generos_sin_target = pd.read_csv('../data/df_sin_target.csv')

predicciones_clasi = final_clasi.predict(generos_sin_target)
generos_sin_target['clasificacion'] = predicciones_clasi

# Predicciones de géneros "buenos" y "malos"
generos_sin_target['genero_predicho'] = None

for indice, fila in generos_sin_target.iterrows():
    if fila['clasificacion'] == 0:
        predicciones_malas = final_malos.predict(fila.drop(['clasificacion', 'genero_predicho']).values.reshape(1, -1))
        predicciones_malas_transformadas = {0: 'pop', 1: 'latin', 2: 'R&B'}.get(predicciones_malas[0], predicciones_malas[0])
        generos_sin_target.at[indice, 'genero_predicho'] = predicciones_malas_transformadas

    else:
        predicciones_buenas = final_buenos.predict(fila.drop(['clasificacion', 'genero_predicho']).values.reshape(1, -1))
        predicciones_buenas_transformadas = {0: 'rap', 1: 'rock', 2: 'EDM'}.get(predicciones_buenas[0], predicciones_buenas[0])
        generos_sin_target.at[indice, 'genero_predicho'] = predicciones_buenas_transformadas

# Guardar el DataFrame actualizado
generos_sin_target.to_csv(f'../predicciones/predicciones{dia_de_hoy}.csv', index=False)