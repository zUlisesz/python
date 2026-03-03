import matplotlib.pyplot as plt

def riemann_rearrangement(L, iterations=1000):
    positivos = []
    negativos = []
    
    # Generamos suficientes términos para la demostración
    # Usamos la serie armónica: 1/n
    for n in range(1, iterations * 2):
        term = ((-1)**(n+1)) / n
        if term > 0:
            positivos.append(term)
        else:
            negativos.append(term)

    sumas_parciales = []
    suma_actual = 0
    p_idx = 0  # puntero para positivos
    n_idx = 0  # puntero para negativos
    
    # Algoritmo de reordenamiento
    for _ in range(iterations):
        # Si la suma actual es menor que L, añadimos positivos
        if suma_actual < L:
            if p_idx < len(positivos):
                suma_actual += positivos[p_idx]
                p_idx += 1
        # Si la suma actual es mayor o igual a L, añadimos negativos
        else:
            if n_idx < len(negativos):
                suma_actual += negativos[n_idx]
                n_idx += 1
        
        sumas_parciales.append(suma_actual)
    
    return sumas_parciales,suma_actual

# Configuración del experimento
L_objetivo = 5# Puedes cambiar este valor a cualquier número real
iteraciones = 100000
historico, current = riemann_rearrangement(L_objetivo, iteraciones)
print(current)

# Graficación
plt.figure(figsize=(12, 6))
plt.plot(historico, label='Suma Parcial Reordenada', color='#2c3e50', linewidth=1)
plt.axhline(y=L_objetivo, color='r', linestyle='--', label=f'Objetivo L = {L_objetivo}')
plt.title(f'Convergencia de la Serie Armónica Alternada reordenada hacia L = {L_objetivo}')
plt.xlabel('Número de términos añadidos')
plt.ylabel('Suma acumulada')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()