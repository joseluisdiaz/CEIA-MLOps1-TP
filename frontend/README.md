# Frontend - Predictor de Accidente Cerebrovascular

Este es el frontend web para interactuar con la API de predicción de accidentes cerebrovasculares (ACV). Proporciona una interfaz gráfica intuitiva para que los usuarios puedan ingresar los datos médicos de un paciente y obtener una predicción sobre el riesgo de ACV.

## Acceso Rápido

**Cuando el sistema esté ejecutándose con Docker Compose:**

**Abrir en el navegador: http://localhost:3000**

El frontend estará disponible automáticamente después de ejecutar:
```bash
docker compose --profile all up -d
# o
./iniciar_sistema.sh
```

## Características

- **Interfaz web**: Formulario organizado por secciones médicas
- **Validación de datos**: Los campos se validan mientras el usuario escribe
- **Responsive**: Diseño adaptable para dispositivos móviles y desktop
- **Datos de ejemplo**: Botón para cargar datos de prueba automáticamente
- **Manejo de errores**: Mensajes informativos para diferentes tipos de errores
- **Accesibilidad**: Diseño pensado para usuarios con diferentes capacidades

## Configuración

### URL de la API

Por defecto, el frontend está configurado para conectarse a la API en `http://localhost:8800`. 

Para cambiar esta configuración, edita el archivo `script.js`:

```javascript
// Línea 2 en script.js
const API_BASE_URL = 'http://<tu-servidor>:puerto';
```

### Configuraciones comunes:

- **Desarrollo local**: `http://localhost:8800`
- **Docker Compose**: `http://localhost:8800` (puerto expuesto)
- **Producción**: `https://tu-dominio.com/api`

## Uso del Formulario

### Campos del Formulario

El formulario está organizado en 6 secciones:

1. **Información Básica**
   - Edad (0-150 años)
   - Sexo (Femenino/Masculino)

2. **Síntomas Cardíacos**
   - Tipo de dolor de pecho
   - Angina inducida por ejercicio

3. **Presión y Valores Sanguíneos**
   - Presión arterial en reposo
   - Colesterol sérico
   - Azúcar en sangre en ayunas

4. **Electrocardiograma y Ritmo**
   - Resultados ECG en reposo
   - Frecuencia cardíaca máxima

5. **Pruebas de Estrés**
   - Depresión ST inducida por ejercicio
   - Pendiente del segmento ST

6. **Estudios Especializados**
   - Vasos principales (Fluoroscopía)
   - Talasemia

### Funcionalidades

- **Cargar Ejemplo**: Llena automáticamente el formulario con datos de prueba
- **Realizar Predicción**: Envía los datos a la API y muestra el resultado
- **Limpiar Formulario**: Borra todos los campos y mensajes

### Resultados

El sistema muestra dos tipos de resultados:

- **✅ Paciente Saludable**: Los parámetros están dentro de rangos normales
- **⚠️ Riesgo Detectado**: Se recomienda consultar con un especialista

## Resolución de Problemas

### Error de Conexión

Si aparece un error de conexión:

1. **Verificar que la API esté ejecutándose**:
   ```bash
   curl http://localhost:8800/
   ```

2. **Verificar puertos**: Asegúrate de que la API esté en el puerto correcto

3. **CORS**: Si usas un servidor local, puede haber problemas de CORS. La API FastAPI debe tener CORS habilitado:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # En producción, especificar dominios
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

### Error 422 (Validación)

- Verifica que todos los campos estén completos
- Asegúrate de que los valores estén en los rangos correctos
- Revisa que los campos numéricos no contengan texto

### Error 500 (Servidor)

- Revisa los logs de la API FastAPI
- Verifica que MLflow esté funcionando
- Asegúrate de que el modelo esté registrado correctamente

## Personalización

### Estilos CSS

El archivo `styles.css` contiene todas las definiciones de estilo. Puedes modificar:

- **Colores**: Variables CSS en la parte superior
- **Tipografía**: Fuentes y tamaños
- **Layout**: Grid y espaciado
- **Animaciones**: Transiciones y efectos

### Funcionalidad JavaScript

El archivo `script.js` es modular y permite:

- Agregar nuevos campos
- Modificar validaciones
- Cambiar la lógica de envío
- Personalizar mensajes de error

## Responsive Design

El frontend está optimizado para:

- **Desktop**: Layout en grid de múltiples columnas
- **Tablet**: Layout en grid de 2 columnas
- **Mobile**: Layout en una sola columna

Los breakpoints principales son:
- `768px`: Tablet y dispositivos medianos
- `480px`: Móviles pequeños

## Seguridad

### Consideraciones importantes:

1. **Validación**: Siempre validar en el backend, no solo en el frontend
2. **HTTPS**: Usar HTTPS en producción para proteger datos médicos
3. **Sanitización**: Los datos se envían como JSON, evitando inyecciones
4. **CORS**: Configurar CORS apropiadamente en producción

## Testing

### Datos de Ejemplo

El botón "Cargar Ejemplo" usa estos valores:

```json
{
  "age": 67,
  "sex": 1,
  "cp": 4,
  "trestbps": 160.0,
  "chol": 286.0,
  "fbs": 0,
  "restecg": 2,
  "thalach": 108.0,
  "exang": 1,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 3,
  "thal": 3
}
```


