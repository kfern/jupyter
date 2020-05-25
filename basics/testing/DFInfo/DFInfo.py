import pandas as pd

def get(dfX, target_column):
  """devuelve el resultado del análisis"""
  # Crear el df para el resultado
  df = pd.DataFrame(index=pd.Series(dfX.columns.values))
  # @TODO: Descartar las filas en las que el target sea null
  
  # Añadir las series
  df['dtype'] = dfX.dtypes
  df['nulls'] = dfX.isnull().sum()  
  # No numéricos
  tmp = dfX.select_dtypes(exclude='number').describe().T[['count', 'unique']]
  df['count'] = pd.concat([dfX.describe().loc['count'], tmp['count']])
  # Numéricos
  df['unique'] = tmp['unique']
  for col in dfX.select_dtypes(include='number').columns.values:
    df.loc[col, 'unique'] = dfX[col].value_counts().sum()

  # Quitar el target
  df.drop(target_column, axis='index', inplace=True)

  return df

