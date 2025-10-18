// Configuración de la API
const API_BASE_URL = 'http://localhost:8800'; // Puerto de FastAPI en docker-compose

// Referencias a elementos del DOM
const form = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultDiv = document.getElementById('result');
const resultContent = document.getElementById('resultContent');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const errorContent = document.getElementById('errorContent');

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    form.addEventListener('submit', handleSubmit);
    clearBtn.addEventListener('click', clearForm);
    
    // Llenar formulario con datos de ejemplo para testing
    addExampleDataButton();
});

/**
 * Maneja el envío del formulario
 */
async function handleSubmit(event) {
    event.preventDefault();
    
    hideAllMessages();
    showLoading();
    
    try {
        const formData = getFormData();
        const prediction = await makePrediction(formData);
        showResult(prediction);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        hideLoading();
    }
}

/**
 * Obtiene los datos del formulario
 */
function getFormData() {
    const formData = new FormData(form);
    const data = {};
    
    // Lista de campos numéricos que deben ser convertidos
    const numericFields = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'];
    const integerFields = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'];
    
    for (let [key, value] of formData.entries()) {
        if (numericFields.includes(key)) {
            data[key] = parseFloat(value);
        } else if (integerFields.includes(key)) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    }
    
    return data;
}

/**
 * Realiza la predicción llamando a la API
 */
async function makePrediction(data) {
    const response = await fetch(`${API_BASE_URL}/predict/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: data })
    });
    
    if (!response.ok) {
        if (response.status === 422) {
            const errorData = await response.json();
            throw new Error(`Error de validación: ${formatValidationErrors(errorData.detail)}`);
        } else if (response.status === 500) {
            throw new Error('Error interno del servidor. Por favor, inténtelo más tarde.');
        } else {
            throw new Error(`Error HTTP ${response.status}: ${response.statusText}`);
        }
    }
    
    return await response.json();
}

/**
 * Formatea los errores de validación de FastAPI
 */
function formatValidationErrors(errors) {
    if (!Array.isArray(errors)) {
        return 'Error de validación desconocido';
    }
    
    return errors.map(error => {
        const field = error.loc ? error.loc.join('.') : 'campo desconocido';
        return `${field}: ${error.msg}`;
    }).join(', ');
}

/**
 * Muestra el resultado de la predicción
 */
function showResult(prediction) {
    const hasHeartDisease = prediction.prediction;
    const message = prediction.description;
    
    resultDiv.className = `result ${hasHeartDisease ? 'warning' : 'success'}`;
    
    const icon = hasHeartDisease ? '⚠️' : '✅';
    const title = hasHeartDisease ? 'RIESGO CARDÍACO DETECTADO' : 'CORAZÓN SALUDABLE';
    const subtitle = hasHeartDisease ? 
        'Se recomienda consultar con un cardiólogo inmediatamente' :
        'Los parámetros cardíacos están dentro de rangos normales';
    
    resultContent.innerHTML = `
        <div class="result-icon">${icon}</div>
        <div class="result-text">${title}</div>
        <div class="result-subtitle">${message}</div>
        <div class="result-subtitle">${subtitle}</div>
    `;
    
    resultDiv.classList.remove('hidden');
    
    // Scroll suave hacia el resultado
    setTimeout(() => {
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

/**
 * Muestra un mensaje de error
 */
function showError(message) {
    errorContent.innerHTML = `
        <div style="font-size: 2rem;">❌</div>
        <div style="font-weight: 600; font-size: 1.2rem;">Error en la Predicción</div>
        <div>${message}</div>
        <div style="font-size: 0.9rem; margin-top: 10px;">
            Por favor, verifique que todos los campos estén completos y que la API esté funcionando.
        </div>
    `;
    errorDiv.classList.remove('hidden');
}

/**
 * Muestra el indicador de carga
 */
function showLoading() {
    loadingDiv.classList.remove('hidden');
    predictBtn.disabled = true;
    predictBtn.textContent = '🔄 Analizando...';
}

/**
 * Oculta el indicador de carga
 */
function hideLoading() {
    loadingDiv.classList.add('hidden');
    predictBtn.disabled = false;
    predictBtn.innerHTML = '🔍 Realizar Predicción';
}

/**
 * Oculta todos los mensajes
 */
function hideAllMessages() {
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    loadingDiv.classList.add('hidden');
}

/**
 * Limpia el formulario
 */
function clearForm() {
    form.reset();
    hideAllMessages();
    
    // Hacer scroll al inicio del formulario
    form.scrollIntoView({ behavior: 'smooth' });
    
    // Enfocar el primer campo
    setTimeout(() => {
        document.getElementById('age').focus();
    }, 300);
}

/**
 * Valida el formulario en tiempo real
 */
function setupRealtimeValidation() {
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateField(this);
        });
        
        input.addEventListener('blur', function() {
            validateField(this);
        });
    });
}

/**
 * Valida un campo individual
 */
function validateField(field) {
    const value = field.value;
    const isValid = field.checkValidity();
    
    if (value && !isValid) {
        field.style.borderColor = '#e74c3c';
    } else if (value && isValid) {
        field.style.borderColor = '#27ae60';
    } else {
        field.style.borderColor = '#e1e8ed';
    }
}

/**
 * Añade un botón para llenar con datos de ejemplo
 */
function addExampleDataButton() {
    const exampleBtn = document.createElement('button');
    exampleBtn.type = 'button';
    exampleBtn.className = 'btn btn-secondary';
    exampleBtn.innerHTML = '📝 Cargar Ejemplo';
    exampleBtn.style.marginLeft = '10px';
    
    exampleBtn.addEventListener('click', fillExampleData);
    
    const formActions = document.querySelector('.form-actions');
    formActions.appendChild(exampleBtn);
}

/**
 * Llena el formulario con datos de ejemplo
 */
function fillExampleData() {
    // Datos de ejemplo del modelo FastAPI
    const exampleData = {
        age: 67,
        sex: 1,
        cp: 4,
        trestbps: 160.0,
        chol: 286.0,
        fbs: 0,
        restecg: 2,
        thalach: 108.0,
        exang: 1,
        oldpeak: 1.5,
        slope: 2,
        ca: 3,
        thal: 3
    };
    
    // Llenar los campos del formulario
    Object.keys(exampleData).forEach(key => {
        const field = document.getElementById(key);
        if (field) {
            field.value = exampleData[key];
            validateField(field);
        }
    });
    
    hideAllMessages();
    
    // Hacer scroll al botón de predicción
    setTimeout(() => {
        predictBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

/**
 * Maneja errores de conexión
 */
function handleConnectionError() {
    const errorMessage = `
        No se pudo conectar con la API de predicción. 
        Por favor, verifique que:
        <ul style="text-align: left; margin-top: 10px;">
            <li>La API esté ejecutándose en ${API_BASE_URL}</li>
            <li>No hay problemas de CORS</li>
            <li>Su conexión a internet funcione correctamente</li>
        </ul>
    `;
    showError(errorMessage);
}

/**
 * Inicialización adicional
 */
document.addEventListener('DOMContentLoaded', function() {
    setupRealtimeValidation();
    
    // Interceptar errores de red globalmente
    window.addEventListener('unhandledrejection', function(event) {
        if (event.reason && event.reason.name === 'TypeError' && 
            event.reason.message.includes('fetch')) {
            handleConnectionError();
            event.preventDefault();
        }
    });
    
    // Mensaje de bienvenida en consola
    console.log('🫀 Predictor de Enfermedad Cardíaca cargado correctamente');
    console.log(`📡 API configurada en: ${API_BASE_URL}`);
});

// Hacer disponibles algunas funciones globalmente para debugging
window.heartDiseasePredictor = {
    clearForm,
    fillExampleData,
    getFormData,
    API_BASE_URL
};