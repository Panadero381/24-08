import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Detector de Anomalías — Módulo 4",
    layout="wide"
)

st.title("🚨 Detector de Anomalías: Lógica + Big-O + NumPy")

st.caption(
    "Módulo 4 — Matemáticas Discretas y Complejidad. "
    "Detector con tres proposiciones lógicas y comparación "
    "entre implementación ingenua y vectorizada."
)

tab1, tab2, tab3 = st.tabs(
    ["🔎 Simulación de alarma", "📈 Notación Big-O", "⚡ Benchmark en vivo"]
)


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------

def generar_datos(n, seed=42):
    rng = np.random.default_rng(seed)

    temperaturas = rng.uniform(15, 40, n)
    humedades = rng.uniform(20, 80, n)

    # Generamos si cada lectura corresponde o no a un fin de semana.
    # False = día entre semana
    # True  = fin de semana
    es_fin_de_semana = rng.choice([False, True], size=n)

    return temperaturas, humedades, es_fin_de_semana


# ---------------------------------------------------------------------------
# Versión con LOOP
# ---------------------------------------------------------------------------

def alarma_logica_loop(
    temperaturas,
    humedades,
    es_fin_de_semana,
    temp_umbral,
    hum_umbral
):
    resultados = []

    for temp, hum, fin_de_semana in zip(
        temperaturas,
        humedades,
        es_fin_de_semana
    ):
        resultado = (
            temp > temp_umbral
            and hum < hum_umbral
            and not fin_de_semana
        )

        resultados.append(resultado)

    return np.array(resultados)


# ---------------------------------------------------------------------------
# Versión VECTORIZADA con NumPy
# ---------------------------------------------------------------------------

def alarma_logica_vectorizada(
    temperaturas,
    humedades,
    es_fin_de_semana,
    temp_umbral,
    hum_umbral
):
    return (
        (temperaturas > temp_umbral)
        & (humedades < hum_umbral)
        & (~es_fin_de_semana)
    )


# ---------------------------------------------------------------------------
# Tab 1: Simulación de alarma
# ---------------------------------------------------------------------------

with tab1:

    st.subheader("Alarma por regla lógica")

    st.write(
        "La alarma se activa cuando se cumplen simultáneamente "
        "tres condiciones:"
    )

    st.markdown(
        """
        - **Temperatura > umbral**
        - **Humedad < umbral**
        - **NO es fin de semana**
        """
    )

    st.write(
        "En lógica proposicional:"
    )

    st.latex(
        r"(Temperatura > 30) \land "
        r"(Humedad < 40) \land "
        r"\neg(Es\_fin\_de\_semana)"
    )

    col_cfg, col_data = st.columns([1, 2])

    with col_cfg:

        n = st.slider(
            "Número de lecturas (n)",
            50,
            5000,
            500,
            step=50
        )

        temp_umbral = st.slider(
            "Umbral temperatura (°C) — mayor que",
            15,
            40,
            30
        )

        hum_umbral = st.slider(
            "Umbral humedad (%) — menor que",
            20,
            80,
            40
        )

    temps, hums, fines = generar_datos(n)

    with col_cfg:

        alarmas = alarma_logica_vectorizada(
            temps,
            hums,
            fines,
            temp_umbral,
            hum_umbral
        )

        st.metric(
            "Alarmas detectadas",
            f"{alarmas.sum()} / {n}"
        )

    with col_data:

        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.scatter(
            temps[~alarmas],
            hums[~alarmas],
            c="steelblue",
            alpha=0.5,
            label="Normal",
            s=15,
        )

        ax.scatter(
            temps[alarmas],
            hums[alarmas],
            c="crimson",
            alpha=0.8,
            label="Alarma / anomalía",
            s=25,
        )

        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Humedad (%)")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    with st.expander("Ver datos y lógica aplicada"):

        df = pd.DataFrame({
            "temperatura": temps.round(2),
            "humedad": hums.round(2),
            "es_fin_de_semana": fines,
            "alarma": alarmas,
        })

        st.dataframe(
            df,
            use_container_width=True,
            height=250
        )


# ---------------------------------------------------------------------------
# Tab 2: Notación Big-O
# ---------------------------------------------------------------------------

with tab2:

    st.subheader("¿Por qué no cambia la complejidad?")

    st.write(
        "Aunque agregamos una tercera proposición lógica, el detector "
        "continúa recorriendo las n lecturas una sola vez."
    )

    st.markdown(
        """
        La complejidad sigue siendo:

        **O(n)**

        Esto ocurre porque por cada lectura solamente realizamos una cantidad
        constante de operaciones:

        1. Comparar la temperatura.
        2. Comparar la humedad.
        3. Comprobar si es fin de semana.
        4. Combinar las tres condiciones.
        """

    )

    st.info(
        "Agregar una condición no cambia Big-O porque seguimos procesando "
        "cada elemento una sola vez. Lo que aumenta es la cantidad constante "
        "de operaciones realizadas por elemento."
    )

    n_max = st.slider(
        "Tamaño máximo de n para la gráfica",
        10,
        200,
        50
    )

    n_valores = np.arange(1, n_max + 1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.plot(
        n_valores,
        np.ones_like(n_valores),
        label="O(1) — constante"
    )

    ax2.plot(
        n_valores,
        n_valores,
        label="O(n) — lineal"
    )

    ax2.plot(
        n_valores,
        n_valores * np.log2(
            np.maximum(n_valores, 2)
        ),
        label="O(n log n)"
    )

    ax2.plot(
        n_valores,
        n_valores ** 2,
        label="O(n²) — cuadrática"
    )

    ax2.set_xlabel("Tamaño de los datos (n)")
    ax2.set_ylabel("Operaciones (teórico)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)


# ---------------------------------------------------------------------------
# Tab 3: Benchmark en vivo
# ---------------------------------------------------------------------------

with tab3:

    st.subheader(
        "Loop vs. NumPy: misma lógica, distinta velocidad real"
    )

    st.write(
        "Se ejecutan las mismas tres condiciones lógicas usando un loop "
        "de Python y una operación vectorizada con NumPy."
    )

    n_bench = st.select_slider(
        "Tamaño de datos para el benchmark",
        options=[
            1_000,
            10_000,
            100_000,
            500_000,
            1_000_000
        ],
        value=1_000_000,
    )

    temp_umbral_b = st.slider(
        "Umbral temperatura (°C)",
        15,
        40,
        30,
        key="temp_bench"
    )

    hum_umbral_b = st.slider(
        "Umbral humedad (%)",
        20,
        80,
        40,
        key="hum_bench"
    )

    if st.button(
        "▶️ Ejecutar benchmark",
        type="primary"
    ):

        temps_b, hums_b, fines_b = generar_datos(
            n_bench
        )

        # Una ejecución para el loop porque es más lento.
        repeticiones_loop = 1

        # Varias ejecuciones para NumPy porque normalmente
        # termina mucho más rápido.
        repeticiones_vec = 20

        # ---------------------------------------------------------------
        # Benchmark LOOP
        # ---------------------------------------------------------------

        inicio = time.perf_counter()

        for _ in range(repeticiones_loop):

            resultado_loop = alarma_logica_loop(
                temps_b,
                hums_b,
                fines_b,
                temp_umbral_b,
                hum_umbral_b
            )

        t_loop = (
            time.perf_counter() - inicio
        ) / repeticiones_loop

        # ---------------------------------------------------------------
        # Benchmark NUMPY
        # ---------------------------------------------------------------

        inicio = time.perf_counter()

        for _ in range(repeticiones_vec):

            resultado_vec = alarma_logica_vectorizada(
                temps_b,
                hums_b,
                fines_b,
                temp_umbral_b,
                hum_umbral_b
            )

        t_vec = (
            time.perf_counter() - inicio
        ) / repeticiones_vec

        # ---------------------------------------------------------------
        # Comprobar que ambas versiones producen el mismo resultado
        # ---------------------------------------------------------------

        resultados_iguales = np.array_equal(
            resultado_loop,
            resultado_vec
        )

        st.subheader("Resultados")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Tiempo con loop",
            f"{t_loop * 1000:.3f} ms"
        )

        col2.metric(
            "Tiempo con NumPy",
            f"{t_vec * 1000:.3f} ms"
        )

        if t_vec > 0:

            speedup = t_loop / t_vec

            col3.metric(
                "NumPy es más rápido por",
                f"{speedup:,.0f}x"
            )

        else:

            col3.metric(
                "NumPy es más rápido por",
                "Demasiado rápido para medir"
            )

        # ---------------------------------------------------------------
        # Verificación de resultados
        # ---------------------------------------------------------------

        if resultados_iguales:

            st.success(
                "Las dos implementaciones producen exactamente "
                "los mismos resultados."
            )

        else:

            st.error(
                "Las implementaciones producen resultados diferentes."
            )

        # ---------------------------------------------------------------
        # Gráfica del benchmark
        # ---------------------------------------------------------------

        fig3, ax3 = plt.subplots(
            figsize=(5, 3.5)
        )

        ax3.bar(
            ["Loop (Python)", "NumPy (vectorizado)"],
            [
                t_loop * 1000,
                t_vec * 1000
            ],
            color=[
                "indianred",
                "seagreen"
            ]
        )

        ax3.set_ylabel(
            "Tiempo (milisegundos)"
        )

        ax3.grid(
            alpha=0.3,
            axis="y"
        )

        st.pyplot(fig3)

        st.caption(
            f"Benchmark realizado con {n_bench:,} lecturas. "
            f"Loop: {repeticiones_loop} corrida(s). "
            f"NumPy: {repeticiones_vec} corridas."
        )

    else:

        st.caption(
            "Ajusta los parámetros y presiona "
            "**Ejecutar benchmark** para ver el resultado."
        )
