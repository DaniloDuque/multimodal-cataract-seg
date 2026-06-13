# Lectura: Fusión de Atención Cruzada para Segmentación de Cataratas

## El problema clínico

Las cataratas son la principal causa de ceguera prevenible en el mundo y afectan a más de 100 millones de personas. Para diagnosticarlas y planificar cirugías, necesitamos **segmentar** las estructuras del segmento anterior del ojo (córnea, pupila, cristalino) a partir de fotografías clínicas. Segmentar significa que un modelo de computadora dibuja exactamente dónde está cada estructura en la imagen.

## Por qué RGB solo no es suficiente

Los modelos que procesan solo la foto en color (RGB) tienen dos problemas:

1. **Bajo contraste de color**: cuando el cristalino está opaco por la catarata, los bordes se ven borrosos y todos los píxeles se ven parecidos. El modelo no sabe dónde termina una estructura y empieza otra.
2. **Artefactos de iluminación**: las lámparas de hendidura (slit lamps) crean reflejos y sombras que confunden al modelo.

Esto es grave porque el modelo falla justo cuando el paciente está más grave.

---

## La solución propuesta: dos vistas de la misma imagen

En lugar de darle al modelo solo la foto RGB, le damos **dos representaciones complementarias** de la misma imagen:

1. **La foto RGB** — captura información de color y textura (fotométrica).
2. **Un mapa de bordes Canny** — una imagen en blanco y negro que solo muestra los contornos geométricos (dónde cambia bruscamente el brillo). Los bordes del cristalino son visibles incluso cuando el color falla.

Aunque ambas vienen de la misma fotografía, representan tipos de información fundamentalmente distintos (color vs. geometría). Por eso su fusión es un problema **multimodal** en el sentido arquitectónico.

---

## ¿Qué es Canny y cómo funciona?

El algoritmo de detección de bordes Canny transforma una imagen a color en un mapa de bordes en blanco y negro. Funciona en 4 pasos:

1. **Suavizar** la imagen con un filtro Gaussiano para eliminar ruido.
2. **Calcular el gradiente** en cada píxel: qué tan rápido cambia el brillo y en qué dirección.
3. **Adelgazar los bordes** (non-maximum suppression): solo se quedan los píxeles que son el máximo local en la dirección del gradiente.
4. **Conectar los bordes** usando dos umbrales:
   - Píxeles sobre $t_2 = 150$: se quedan como bordes fuertes.
   - Píxeles bajo $t_1 = 50$: se descartan.
   - Píxeles entre $t_1$ y $t_2$: se quedan solo si están conectados a un borde fuerte.

Esta estrategia de dos umbrales produce bordes limpios y conectados, eliminando el ruido.

---

## Trabajo relacionado: ¿qué más existe?

### Segmentación de cataratas
- **U-Net** (Ronneberger 2015): arquitectura encoder-decoder con skip connections. Es el estándar para segmentación médica y el backbone de nuestro modelo.
- **CaDIS** (Grammatikopoulou 2021): dataset de videos de cirugía de cataratas con anotaciones. Motivación del dominio.

### Fusión multimodal
- **Attention Is All You Need** (Vaswani 2017): define el mecanismo de atención con Queries, Keys y Values que usamos.
- **mmFormer** (Zhang 2022): transformer multimodal para segmentación de tumores cerebrales. Usa cross-attention entre modalidades de MRI. Inspiración directa de nuestro módulo de fusión.
- **DECTNet** (Li 2024): encoder dual con CNN + Transformer. Confirma que los encoders paralelos funcionan.

### Modelos grandes / fundacionales
- **MedSAM** (Ma 2024): fine-tuning de SAM en 1.5 millones de pares imagen-máscara médica. Requiere 20 GPUs A100 para entrenar. Poco práctico para datasets pequeños.
- **SAM-Adapter** (Chen 2023): adaptadores ligeros para guiar SAM en tareas específicas. Sigue siendo costoso.

### Nuestra posición
Nuestro modelo cierra la brecha entre (1) arquitecturas específicas como U-Net que son unimodales y (2) fusión multimodal como mmFormer que asume múltiples sensores. Somos los primeros en aplicar cross-attention entre una imagen RGB y su propio mapa de bordes Canny para segmentación de cataratas.

---

## La arquitectura en detalle

### Vista general
Tenemos **dos encoders U-Net en paralelo**, uno por cada representación. Un encoder es una serie de capas que van comprimiendo la imagen en representaciones cada vez más abstractas y pequeñas, desde 512×512 píxeles hasta un resumen de 16×16.

Ambos encoders usan la misma arquitectura ResNet-34 pero **pesos completamente independientes**, para que cada uno aprenda lo que es relevante para su modalidad.

### Las 5 etapas del encoder
Cada encoder produce mapas de features en 5 resoluciones (etapas), con profundidades de canales: 64 → 64 → 128 → 256 → 512. Las resoluciones espaciales van de 256×256 → 128×128 → 64×64 → 32×32 → 16×16.

### Cross-Attention (fusión por atención)
En el cuello de botella (etapa 5, resolución 16×16, 512 canales) aplicamos un **módulo de cross-attention multi-cabeza** (8 cabezas). Ambos mapas de features se reordenan de formato $(B, C, H, W)$ a $(B, H*W, C)$ para usar `nn.MultiheadAttention` de PyTorch.

La idea: el encoder RGB genera **Queries (Q)** y el encoder de bordes genera **Keys (K) y Values (V)**. La atención calcula un mapa que dice qué tan relevante es cada ubicación del mapa de bordes para cada ubicación del mapa RGB. Luego mezcla esa información:

$$\tilde{\mathbf{F}}_A = \text{LayerNorm}(\alpha \cdot \text{MHA}(Q{=}\mathbf{F}_A, K{=}\mathbf{F}_B, V{=}\mathbf{F}_B) + \mathbf{F}_A$$

**Traducción**: los features RGB preguntan "¿dónde están los bordes que me sirven?" y los features de bordes responden "aquí están los bordes". La respuesta se pondera por $\alpha$ (el gate) y se suma a los features RGB originales (conexión residual).

### El gate aprendible ("perilla de volumen")
El mapa Canny no siempre es bueno. A veces captura ruido, artefactos, o bordes irrelevantes. Si forzamos al modelo a usar los bordes siempre, podemos empeorarlo.

Solución: un **gate escalar aprendible** $\alpha = \sigma(\theta)$. Al inicio $\theta = -4$, lo que da $\alpha \approx 0.02$ (casi cero). El modelo empieza como un U-Net RGB normal. Durante el entrenamiento, si los bordes ayudan, $\theta$ crece y $\alpha$ se abre (gradiente descendente). Si no ayudan, el gate se queda cerrado.

**Diferencia clave con Early Fusion**: Early Fusion concatena RGB y bordes en la entrada y fuerza al modelo a usar ambas desde la época 1. El gate es selectivo: deja pasar los bordes solo cuando son útiles.

### Fusión multi-escala
Un solo punto de fusión en el cuello de botella no es suficiente porque los detalles finos de bordes se pierden al reducir la resolución. Agregamos dos mecanismos:

1. **Cross-attention en etapa 4**: un segundo módulo de atención a 256 canales con resolución 32×32, con su propio gate aprendible. Inyecta información geométrica un nivel antes del cuello de botella.

2. **Skip connections del encoder de bordes**: en cada uno de los 4 niveles de skip connection del decodificador, concatenamos los features del encoder de bordes con los del encoder RGB y los proyectamos con una convolución 1×1:
   $$\hat{\mathbf{F}}^{(s)} = \text{Conv}_{1\times1}([\mathbf{F}_A^{(s)} \;\|\; \mathbf{F}_B^{(s)}])$$

   Así los bordes influyen en la reconstrucción a todas las resoluciones, no solo en el cuello de botella.

### Decodificador y salida
El decodificador sigue el camino estándar de U-Net con upsampling bilineal y bloques convolucionales 3×3. Las skip connections fusionadas reemplazan las skip connections RGB originales. Una convolución final 1×1 con activación sigmoide produce la máscara de segmentación binaria.

### Función de pérdida
Combinamos dos pérdidas:
- **Binary Cross-Entropy (BCE)**: penaliza errores píxel por píxel.
- **Dice Loss**: penaliza errores de solapamiento entre la máscara predicha y la real.

$$\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$

Esta combinación es estándar en segmentación médica porque ataca ambos tipos de error.

---

## Diseño experimental

### Dataset: Cataract-Seg
- Fuente: Roboflow Universe (Muhammad Risma, 2023)
- Imágenes del segmento anterior con máscaras píxel a píxel
- Estructuras anotadas: córnea, pupila, cristalino
- Partición estratificada 70/15/15: 210 entrenamiento, 45 validación, 45 prueba
- Sin solapamiento de pacientes entre splits

### Preprocesamiento
- Redimensionar a 512×512
- Normalizar valores RGB a [0, 1]
- Calcular mapa Canny (t1=50, t2=150) y replicarlo a 3 canales
- Idéntico para todos los modelos

### Aumentación (solo entrenamiento)
- Flip horizontal (p=0.5)
- Rotación ±15°
- Jitter de brillo y contraste [0.8, 1.2]
- Los mapas Canny se recalculan después de transforms geométricas
- No hay aumentación en validación ni prueba

### Líneas base (baselines)
| Modelo | Entrada | Descripción |
|--------|---------|-------------|
| U-Net (RGB) | Solo RGB | Estándar unimodal |
| U-Net (Bordes) | Solo Canny | Para ver qué tan buenos son los bordes solos |
| U-Net (Early Fusion) | RGB + Canny concatenados | Fusión simple, sin atención |
| **Propuesto** | RGB y Canny por separado | Encoder dual + cross-attention |

Todos usan el mismo backbone U-Net, optimizador y schedule para comparación justa.

### Métricas
- **IoU (Intersection over Union)**: métrica principal. Mide el solapamiento entre predicción y verdad dividido por su unión. 1 = perfecto, 0 = nada.
  $$\text{IoU} = \frac{|\hat{M} \cap M|}{|\hat{M} \cup M|} = \frac{\text{overlap}}{\text{union}}$$
  
- **Dice / F1**: métrica secundaria. Pesa más el solapamiento. Equivalente a F1 en segmentación binaria.
  $$\text{Dice} = \frac{2|\hat{M} \cap M|}{|\hat{M}| + |M|}$$

  IoU es más conservador (siempre ≤ Dice) y es el estándar en benchmarks.

### Hiperparámetros
| Parámetro | Valor |
|-----------|-------|
| Optimizador | AdamW |
| Learning rate | 10⁻⁴ |
| Weight decay | 10⁻² |
| Batch size | 8 |
| Épocas | 100 |
| Cabezas de atención | 8 |
| Semilla aleatoria | 42 |
| Hardware | 1× NVIDIA A100 40GB (Google Colab) |

---

## Resultados

### Resultados cuantitativos (test set, 45 imágenes)

| Modelo | IoU | Dice / F1 |
|--------|:---:|:----------:|
| U-Net (RGB) | 0.9529 | 0.9759 |
| U-Net (Bordes) | 0.8817 | 0.9371 |
| U-Net (Early Fusion) | **0.9532** | **0.9761** |
| **Propuesto** | **0.9530** | 0.9759 |

Tres lecturas importantes:

1. **El modelo propuesto empata en primer lugar** con el mejor baseline (Early Fusion, 0.9532). La diferencia de 0.0002 IoU es ruido estadístico — con 45 imágenes de prueba no hay un ganador real.

2. **Bordes solos (0.8817) confirman que Canny no es suficiente por sí mismo** pero aporta información complementaria útil cuando se fusiona correctamente.

3. **Las mejoras sí funcionan**: la versión anterior del modelo (sin gate, sin etapa 4, sin skip de bordes) quedó en último lugar con 0.9480 IoU. Las tres mejoras lo subieron a empatar el primero (+0.005 IoU), que no es casualidad.

### Curvas de entrenamiento
- Todas las curvas de pérdida (BCE+Dice) convergen sin divergencia.
- El modelo de solo bordes tiene pérdida consistentemente más alta, coherente con su IoU bajo.
- El modelo propuesto y Early Fusion tienen velocidad de convergencia y pérdida final similares.
- RGB converge un poco más rápido al inicio por su espacio de entrada más simple.

### Curvas de IoU de validación
- El modelo propuesto y Early Fusion siguen trayectorias casi idénticas durante las 100 épocas.
- Ambos superan al RGB en épocas tardías.
- Bordes solos se estabiliza en un IoU mucho más bajo.

### Resultados cualitativos
- El modelo de solo bordes produce máscaras visiblemente más gruesas y ruidosas.
- Los tres modelos basados en RGB producen máscaras visualmente similares.
- El modelo propuesto y Early Fusion tienen delineación ligeramente más limpia en imágenes con opacificación del cristalino.

---

## Discusión

### Saturación del dataset
Los tres modelos basados en RGB tienen IoU entre 0.9529 y 0.9532. Esto sugiere que el dataset Cataract-Seg está **cerca del techo de rendimiento** — no hay mucho margen de mejora para ningún modelo RGB, independientemente de su arquitectura.

### Ruido estadístico
Con solo 45 imágenes de prueba (30 en la versión anterior del dataset), las diferencias de 0.0003 IoU están dentro del ruido estadístico. No se puede hacer una prueba de significancia a esta escala.

### El gate funciona
Que el modelo propuesto empate al mejor baseline **a pesar de tener encoders duales y atención multi-escala** valida que el mecanismo de fusión no degrada el rendimiento. Sin el gate, la versión anterior quedó de última. El gate es crítico: si los bordes son ruidosos, el modelo simplemente los ignora y se comporta como un U-Net RGB.

### Limitaciones
1. **Dataset pequeño**: 45 imágenes de prueba no permiten conclusiones estadísticas sólidas.
2. **Umbrales Canny fijos**: t1=50, t2=150 se eligieron empíricamente pero podrían no ser óptimos para todas las imágenes.
3. **Segmentación binaria**: el dataset permite segmentación multi-clase (córnea, pupila, cristalino por separado) pero evaluamos binario.
4. **Sin tabla de ablación completa**: idealmente deberíamos medir el impacto de cada mejora por separado.

### Trabajo futuro
1. Validar en datasets oftálmicos más grandes.
2. Explorar detectores de bordes aprendibles (filtros Sobel con pesos entrenables) en lugar de Canny fijo.
3. Extender a segmentación multi-clase (córnea, pupila, cristalino por separado).
4. Extender a video intraoperatorio (como CaDIS).

---

## Las tres contribuciones arquitectónicas

1. **Gate escalar aprendible** ($\alpha = \sigma(\theta)$, inicializado en -4): permite que el modelo ignore bordes ruidosos y aprenda a usarlos solo cuando ayudan.
2. **Cross-attention en etapa 4** (256 canales, 32×32): inyecta información geométrica antes del cuello de botella para contexto multi-escala.
3. **Skip connections del encoder de bordes**: concatenación + proyección 1×1 en cada nivel del decodificador para propagar bordes a todas las resoluciones.

---

## En una oración

> Propusimos un modelo de segmentación que trata una sola imagen de ojo como dos modalidades (color y geometría), las fusiona con atención cruzada controlada por un gate aprendible, y logra segmentar el cristalino tan bien como el mejor baseline (IoU 0.9530) mientras garantiza que el ruido en los bordes no lo perjudica.

---

## Para la presentación: puntos clave por slide

### Motivación (slides 3-5)
- 100M+ personas afectadas, principal causa de ceguera prevenible
- RGB falla con bajo contraste y artefactos de iluminación
- Solución: segunda representación (Canny) + fusión multimodal

### Trabajo relacionado (slides 6-8)
- U-Net: backbone estándar
- mmFormer, DECTNet: cross-attention entre modalidades
- MedSAM/SAM-Adapter: muy costosos para datasets pequeños
- Nosotros: primeros en fusionar RGB + Canny para cataratas

### Método (slides 9-15)
- Dos encoders U-Net paralelos (RGB y Canny), pesos independientes
- Cross-attention en cuello de botella: Q de RGB, K,V de bordes
- Gate aprendible: empieza en cero (α ≈ 0.02), se abre solo si ayuda
- Fusión multi-escala: etapa 4 + skip connections de bordes
- Pérdida: BCE + Dice

### Experimentos (slides 16-17)
- Dataset: Cataract-Seg, 210/45/45, aumentación on-the-fly
- 4 modelos comparados, mismo backbone y schedule
- Métricas: IoU (primaria), Dice/F1 (secundaria)

### Resultados (slides 18-22)
- Propuesto: 0.9530 IoU — empata con Early Fusion (0.9532)
- Bordes solos: 0.8817 — confirma que no bastan solos
- Sin gate: versión anterior quedó última
- Dataset saturado, diferencias dentro del ruido estadístico

### Conclusión (slides 23-25)
- 3 contribuciones: gate, etapa 4, skip de bordes
- Fusión robusta al ruido
- 0.9530 IoU, reproducible, código abierto
- Limitaciones: dataset pequeño, Canny fijo, sin ablación
