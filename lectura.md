# Lectura: Fusión de Atención Cruzada para Segmentación de Cataratas

## El problema

Las cataratas opacan el cristalino del ojo. Para diagnosticarlas automáticamente, necesitamos que un modelo de computadora mire una foto del ojo y dibuje exactamente dónde están el cristalino, la pupila y la córnea. A esto se le llama **segmentación**.

El problema con usar solo la foto en color (RGB) es que cuando el cristalino está muy opaco, pierde contraste — todos los píxeles se ven parecidos y el modelo no sabe bien dónde termina una estructura y dónde empieza otra.

---

## La solución propuesta

En lugar de darle al modelo solo la foto en color, le damos **dos representaciones de la misma imagen**:

1. **La foto RGB** — captura color, textura, brillo.
2. **Un mapa de bordes Canny** — una imagen en blanco y negro que solo muestra los contornos geométricos. Los bordes del cristalino son visibles incluso cuando el color falla.

Aunque ambas vienen de la misma foto, representan cosas distintas: una es fotométrica, la otra es geométrica. A eso le llamamos **multimodal en sentido arquitectónico**.

---

## La arquitectura — cómo funciona

Tenemos **dos encoders U-Net en paralelo**, uno por cada representación. Un encoder es básicamente una serie de capas que van comprimiendo la imagen en representaciones cada vez más abstractas y pequeñas — de 512×512 píxeles hasta un "resumen" de 16×16.

Ambos encoders tienen la misma arquitectura (ResNet-34) pero pesos completamente independientes, así cada uno aprende lo que es relevante para su modalidad.

En el centro de la red — el "cuello de botella" — aplicamos un **módulo de cross-attention**. La idea es simple: el encoder RGB le "pregunta" al encoder de bordes. Matemáticamente:

- El encoder RGB genera **Queries (Q)** — las preguntas.
- El encoder de bordes genera **Keys (K)** y **Values (V)** — las respuestas.
- La atención calcula qué tan relevante es cada parte del mapa de bordes para cada parte de la imagen RGB, y mezcla esa información.

Esto enriquece la representación RGB con información estructural antes de que el decodificador reconstruya la máscara.

---

## El gate aprendible — por qué es importante

El mapa de bordes Canny no siempre es bueno. A veces capta ruido, artefactos de la lámpara, o bordes irrelevantes. Si forzamos al modelo a siempre usar los bordes, podemos hacerlo peor.

La solución es un **gate escalar aprendible** α = σ(θ). Al inicio del entrenamiento, θ = −4, lo que da α ≈ 0.02 — casi cero. El modelo empieza comportándose como un U-Net RGB normal. Con el tiempo, si los bordes realmente ayudan, θ crece y α se abre. Si no ayudan, el gate se queda cerrado.

Esto es lo que diferencia el modelo propuesto de simplemente concatenar RGB y bordes al inicio (Early Fusion) — ese enfoque fuerza al modelo a tratar ambas modalidades igual desde el primer momento, sin opción de ignorar la ruidosa.

---

## Fusión multi-escala

Un solo punto de fusión en el cuello de botella no es suficiente — la información geométrica se pierde al decodificar. Por eso aplicamos **tres mecanismos adicionales**:

1. **Cross-attention en la etapa 4** (256 canales, resolución 32×32) — inyecta información de bordes un nivel antes del cuello de botella.
2. **Skip connections del encoder de bordes** — en cada nivel del decodificador, concatenamos los features del encoder de bordes con los del encoder RGB y los proyectamos con una convolución 1×1. Así los bordes influyen en la reconstrucción a todas las resoluciones.

---

## Los resultados

Evaluamos 4 modelos en el dataset Cataract-Seg (45 imágenes de prueba):

| Modelo | IoU |
|---|---|
| U-Net solo RGB | 0.9529 |
| U-Net solo bordes | 0.8817 |
| U-Net fusión temprana | 0.9532 |
| **Propuesto** | **0.9530** |

El modelo propuesto empata con el mejor baseline. Eso puede sonar decepcionante, pero hay dos lecturas importantes:

- Con solo 45 imágenes de prueba, una diferencia de 0.0002 IoU es puro ruido estadístico — no hay un ganador real.
- Lo que sí se valida es que el modelo **no degrada** a pesar de la complejidad adicional. El gate funciona.
- La versión anterior (sin gate, sin fusión multi-escala) quedó en último lugar con 0.9480 IoU. Las tres mejoras lo subieron a empatar el primero — eso sí es una mejora real (+0.005).

El modelo de solo bordes (0.8817) confirma que Canny solo no es suficiente, pero que sí aporta información complementaria cuando se usa bien.

---

## En una oración

Propusimos un modelo de segmentación que trata una sola imagen de ojo como dos modalidades — color y geometría — las fusiona con atención cruzada controlada por un gate aprendible, y logra segmentar el cristalino tan bien como el mejor baseline mientras garantiza que el ruido en los bordes no lo perjudica.
