# Healthcare Cost Analytics & Modeling
¿Qué factores explican mejor el coste sanitario individual y cómo podemos predecirlo con precisión razonable?

Objetivo
Este análisis parte de un problema clásico en seguros de salud: cómo estimar los costes médicos futuros a partir de información básica del paciente. Las principales preguntas fueron:

¿Qué peso tienen la edad, el IMC y el tabaquismo en los gastos médicos?
¿Hasta qué punto fumar cambia el coste esperado?
¿Se puede mejorar un modelo lineal básico sin perder interpretabilidad?

Las hipótesis iniciales fueron que el tabaquismo sería el factor más influyente, que la edad y el IMC tendrían relación directa con el coste, y que un modelo lineal serviría como buena base antes de intentar mejoras más complejas.

Dataset

El dataset proviene de Kaggle — Medical Cost Personal Dataset (insurance.csv). Contiene unas 1.3k observaciones y 7 variables principales: age, sex, bmi, children, smoker, region, charges.
Durante la auditoría de datos se eliminó un duplicado, se aplicó imputación sencilla (mediana y moda) y se redujeron outliers en charges mediante winsorización al 1–99%. El dataset final quedó limpio, consistente y listo para análisis y modelado.

Métodos
El proceso comenzó con un análisis exploratorio (EDA) para entender las distribuciones de coste, comparar fumadores y no fumadores, y estudiar las diferencias entre regiones. Después se construyó un modelo lineal como línea base: un modelo simple, interpretativo y fácilmente explicable.

A continuación se probó una mejora teórica mediante la incorporación de términos polinómicos e interacciones entre variables numéricas, con la intención de capturar relaciones no lineales (por ejemplo, el impacto de la edad no necesariamente crece de forma uniforme). Finalmente, se adoptó una aproximación más controlada con regularización Ridge, incorporando solo las interacciones con sentido de negocio: la combinación de edad e IMC con el hábito tabáquico.

Este enfoque permitió equilibrar la complejidad y la interpretabilidad, validando los modelos con K-Fold Cross Validation para garantizar estabilidad y consistencia.

Resultados
El modelo lineal inicial logró un rendimiento sólido (R² = 0.82; MAE ≈ €4,043). Al introducir polinomios e interacciones entre todas las variables, el rendimiento empeoró (R² = 0.81; MAE ≈ €4,270), mostrando síntomas claros de sobreajuste. Sin embargo, el modelo final —Ridge Regularizado con interacciones dirigidas por negocio (age × smoker, bmi × smoker)— mejoró de forma sustancial, alcanzando R² = 0.89 y MAE ≈ €2,708, una mejora del 33 % respecto al modelo base.

Además, las pruebas cruzadas (10 folds) confirmaron la estabilidad del modelo final, con un MAE medio de €2,908 ± 148 y un R² medio de 0.838 ± 0.032. Esta combinación demostró que la complejidad aplicada con criterio puede mejorar el rendimiento sin perder robustez.

Insights de negocio
Los resultados confirman que el tabaquismo es el factor con mayor impacto económico. Los fumadores generan, en promedio, gastos significativamente superiores a los no fumadores. La edad y el IMC también muestran una correlación positiva con los costes, y su efecto se ve amplificado en personas fumadoras.

Aunque la región influye en cierta medida, su impacto es menor frente a los factores individuales. Desde un punto de vista empresarial, estos hallazgos apoyan estrategias de prevención y programas específicos de reducción del riesgo, especialmente centrados en dejar de fumar y controlar el peso.

Visualizaciones incluidas
El análisis visual incluyó varias perspectivas complementarias: la distribución general de charges, un boxplot comparando fumadores y no fumadores, un gráfico de barras con los costes medios por región, y una dispersión entre edad y gasto médico separando ambos grupos. Además, se representaron los coeficientes del modelo lineal en escala logarítmica y los Top 7 predictores en un gráfico horizontal para destacar su peso relativo.

Decisiones de modelado
El uso de Polynomial + Interactions se probó con la intención de capturar posibles relaciones no lineales y efectos combinados entre edad, BMI y número de hijos. Sin embargo, la complejidad añadida superó el tamaño y la variabilidad del dataset, reduciendo el rendimiento.

La posterior decisión de usar Ridge con interacciones seleccionadas manualmente (basadas en conocimiento del dominio) resultó ser la más efectiva. Esta estrategia permitió introducir complejidad útil sin sobreajustar, manteniendo la interpretación de los coeficientes y logrando una mejora real del modelo.

En términos prácticos, el resultado muestra que el mejor modelo no siempre es el más complejo, sino el que combina información relevante con regularización adecuada.

