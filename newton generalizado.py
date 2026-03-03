import flet as ft
import sympy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart

# Configuración para que Matplotlib no abra ventanas externas
matplotlib.use("agg")

def main(page: ft.Page):
    page.title = "Newton Optimizer Pro + Visualizer"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 1100
    page.window_height = 850
    page.padding = 20

    # --- Componentes de la Interfaz ---
    txt_func = ft.TextField(label="Función f(x, y)", value="(x-2)**4 + (x-2*y)**2", expand=True)
    txt_point = ft.TextField(label="Punto inicial (x, y)", value="0, 0", expand=True)
    
    results_col = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=True)
    
    # Contenedor para la gráfica
    chart_container = ft.Container(
        content=ft.Text("La gráfica aparecerá aquí", color=ft.Colors.GREY_400),
        alignment=ft.alignment.center,
        expand=True,
    )

    def optimize_click(e):
        results_col.controls.clear()
        chart_container.content = ft.ProgressRing() # Indicador de carga
        page.update()

        try:
            # 1. Parsing y preparación
            expr_str = txt_func.value
            point_str = txt_point.value
            expr = sp.sympify(expr_str)
            vars_sym = sorted(list(expr.free_symbols), key=lambda s: s.name)
            
            if len(vars_sym) != 2:
                raise ValueError("La visualización 3D solo soporta 2 variables (x, y).")

            x_val = np.array([float(val.strip()) for val in point_str.split(",")], dtype=float)
            
            # Funciones numéricas
            f_num = sp.lambdify(vars_sym, expr, 'numpy')
            grad_sym = [sp.diff(expr, v) for v in vars_sym]
            hess_sym = [[sp.diff(g, v) for v in vars_sym] for g in grad_sym]
            f_grad = sp.lambdify(vars_sym, grad_sym, 'numpy')
            f_hessian = sp.lambdify(vars_sym, hess_sym, 'numpy')

            # 2. Ciclo de Newton y guardado de trayectoria para el plot
            history = [x_val.copy()]
            tol = 1e-6
            for i in range(1, 15):
                g = np.array(f_grad(*x_val), dtype=float)
                H = np.array(f_hessian(*x_val), dtype=float)
                
                if np.linalg.norm(g) < tol:
                    results_col.controls.append(ft.Text(f"✅ Convergencia: Iter {i-1}", color="green", weight="bold"))
                    break

                delta = np.linalg.solve(H, -g)
                x_val = x_val + delta
                history.append(x_val.copy())
                results_col.controls.append(ft.Text(f"Iter {i}: {np.round(x_val, 4)}"))

            # 3. Creación de la Gráfica 3D
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection='3d')
            
            # Crear malla alrededor del resultado
            center_x, center_y = x_val
            x_range = np.linspace(center_x - 8, center_x + 8, 40)
            y_range = np.linspace(center_y - 8, center_y + 8, 40)
            X, Y = np.meshgrid(x_range, y_range)
            Z = f_num(X, Y)

            # Dibujar superficie
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
            
            # Dibujar trayectoria del algoritmo
            history = np.array(history)
            z_history = [f_num(p[0], p[1]) for p in history]
            ax.plot(history[:, 0], history[:, 1], z_history, color='purple', marker='o', markersize=4, label='Trayectoria')
            
            ax.set_title("Superficie y Trayectoria de Newton")
            ax.set_xlabel(vars_sym[0].name)
            ax.set_ylabel(vars_sym[1].name)
            
            chart_container.content = MatplotlibChart(fig, expand=True)

        except Exception as ex:
            results_col.controls.append(ft.Text(f"❌ Error: {str(ex)}", color="red"))
            chart_container.content = ft.Text("Error al generar gráfica")
        
        page.update()

    # --- Diseño de la Página (Layout) ---
    page.add(
        ft.Text("Newton Optimizer & Surface Visualizer", size=32, weight="bold"),
        ft.Divider(),
        ft.Row([txt_func, txt_point, ft.ElevatedButton("Optimizar", icon=ft.Icons.PLAY_ARROW, on_click=optimize_click)]),
        ft.Row([
            # Columna Izquierda: Logs
            ft.Container(
                content=ft.Column([
                    ft.Text("Log de Iteraciones", weight="bold"),
                    ft.Container(results_col, bgcolor=ft.Colors.GREY_50, padding=10, border_radius=10, expand=True)
                ]),
                expand=1,
                height=500
            ),
            # Columna Derecha: Gráfica
            ft.Container(
                content=chart_container,
                expand=2,
                height=500,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=10
            )
        ], expand=True)
    )

ft.app(target=main)