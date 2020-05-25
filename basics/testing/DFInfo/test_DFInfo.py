import unittest
import numpy as np
import pandas as pd
import DFInfo

class TestCalc(unittest.TestCase):

  def test_get(self):
    # get devuelve un df con información de interés sobre cada feature

    # Construir un DataFrame con tres columnas y dos filas, valores nulos, etc
    fakeCols = ['A', 'B', 'C']
    fakeData = {
      'row_1': [2, 1], 
      'row_2': [3, 'f', 'h'],
      'row_3': [4, 'f', 'h']
    }
    dfX = pd.DataFrame.from_dict(fakeData, orient='index', columns=fakeCols)  
    dfX = dfX.astype(dtype= {"A":"int64"})
    dfX['target'] = dfX['A'] * 3.14

    # Act
    r = DFInfo.get(dfX, 'target')

    # Assertions
    # Hay una fila por cada columna, excepto para el target
    np.testing.assert_array_equal(r.index.values, fakeCols, 'No se han recibido las columnas esperadas')

    # Tiene la estructura esperada
    self.assertEqual(r.shape, (3, 4), 'Tiene la estructura esperada')

    # Están las series esperadas
    tmp = ['dtype', 'nulls', 'count', 'unique']
    np.testing.assert_array_equal(r.columns.values, tmp)

    # nulls: La columna C tiene un valor nulo => 50%
    tmp = r.loc[r['nulls'] > 0]['nulls'].to_dict()    
    self.assertEqual(tmp, {'C': 1}, 'La columna tiene un valor nulo')

    # count: El resto son no nulos
    tmp = r['count'].sum()
    self.assertEqual(tmp, 8, 'El resto no son nulos')

    # unique
    tmp = r['unique'].sum()
    self.assertEqual(tmp, 6, 'TDD: Valores únicos, incluyendo los numéricos')


# End of tests

if __name__ == '__main__':
    unittest.main()
