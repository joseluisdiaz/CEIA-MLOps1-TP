#!/bin/bash

# Script de Inicialización - Predictor de ACV
# Este script inicia todo el sistema MLOps con el frontend web

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con colores
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Header
echo "==============================================="
print_header " Predictor de Accidente Cerebrovascular"
print_header "               MLOps"
echo "==============================================="
echo

# Verificar prerrequisitos
print_message "Verificando prerrequisitos..."

# Verificar Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker no está instalado. Por favor, instalar Docker primero."
    exit 1
fi

# Verificar Docker Compose
if ! docker compose version &> /dev/null; then
    print_error "Docker Compose no está instalado. Por favor, instalar Docker Compose primero."
    exit 1
fi

# Verificar que Docker esté ejecutándose
if ! docker info &> /dev/null; then
    print_error "Docker no está ejecutándose. Por favor, iniciar Docker primero."
    exit 1
fi

print_message "Prerrequisitos verificados"

# Verificar puertos disponibles
print_message "Verificando puertos disponibles..."

check_port() {
    local port=$1
    local service=$2
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Puerto $port ($service) parece estar en uso"
        return 1
    fi
    return 0
}

ports_ok=true
check_port 3000 "Frontend" || ports_ok=false
check_port 8800 "FastAPI" || ports_ok=false
check_port 5000 "MLflow" || ports_ok=false
check_port 8080 "Airflow" || ports_ok=false
check_port 9000 "MinIO API" || ports_ok=false
check_port 9001 "MinIO Console" || ports_ok=false

if [ "$ports_ok" = false ]; then
    print_warning "Algunos puertos están en uso. El sistema puede tener conflictos."
    read -p "¿Continuar de todas formas? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_message "Operación cancelada."
        exit 1
    fi
fi

# Verificar memoria disponible
print_message "Verificando memoria disponible..."
available_memory=$(free -g | awk '/^Mem:/{print $7}')
if [ "$available_memory" -lt 3 ]; then
    print_warning "Memoria disponible baja (${available_memory}GB). Se recomiendan al menos 4GB."
fi

# Menú de opciones
echo
print_header "Opciones de inicio:"
echo "1) Iniciar sistema completo (recomendado)"
echo "2) Solo frontend y API (para testing rápido)"
echo "3) Solo pipeline de datos (ETL + MLflow)"
echo "4) Solo servicios de monitoreo"
echo "5) Limpiar todo y empezar de cero"
echo "6) Salir"
echo

read -p "Selecciona una opción (1-6): " option

case $option in
    1)
        print_message "Iniciando sistema completo..."
        docker compose --profile all up --build -d
        ;;
    2)
        print_message "Iniciando frontend y API únicamente..."
        docker compose up frontend fastapi mlflow postgres s3 --build -d
        ;;
    3)
        print_message "Iniciando pipeline de datos..."
        docker compose --profile airflow --profile mlflow up --build -d
        ;;
    4)
        print_message "Iniciando servicios de monitoreo..."
        docker compose up mlflow s3 postgres --build -d
        ;;
    5)
        print_warning "¡CUIDADO! Esto eliminará todos los datos existentes."
        read -p "¿Estás seguro? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message "Eliminando contenedores y volúmenes..."
            docker compose --profile all down -v
            docker system prune -f
            print_message "Iniciando sistema limpio..."
            docker compose --profile all up --build -d
        else
            print_message "Operación cancelada."
            exit 0
        fi
        ;;
    6)
        print_message "Saliendo..."
        exit 0
        ;;
    *)
        print_error "Opción inválida."
        exit 1
        ;;
esac

# Esperar a que los servicios estén listos
print_message "Esperando a que los servicios estén listos..."
echo

# Función para verificar si un servicio está listo
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "Esperando $service_name... "
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN} ${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED} Timeout${NC}"
    return 1
}

# Verificar servicios según la opción seleccionada
case $option in
    1|2)
        wait_for_service "http://localhost:3000/health" "Frontend"
        wait_for_service "http://localhost:8800/" "FastAPI"
        wait_for_service "http://localhost:5000/" "MLflow"
        ;;
    3|4)
        wait_for_service "http://localhost:5000/" "MLflow"
        ;;
esac

echo
print_message "Sistema iniciado exitosamente!"
echo

# Mostrar URLs de acceso
print_header "URLs de Acceso:"
case $option in
    1)
        echo "• Frontend Web:      http://localhost:3000"
        echo "• API FastAPI:       http://localhost:8800"
        echo "• Airflow Web UI:    http://localhost:8080 (admin/admin)"
        echo "• MLflow UI:         http://localhost:5000"
        echo "• MinIO Console:     http://localhost:9001 (minio/minio123)"
        ;;
    2)
        echo "• Frontend Web:      http://localhost:3000"
        echo "• API FastAPI:       http://localhost:8800"
        echo "• MLflow UI:         http://localhost:5000"
        ;;
    3|4)
        echo "• MLflow UI:         http://localhost:5000"
        if [ "$option" = "3" ]; then
            echo "• Airflow Web UI:    http://localhost:8080 (admin/admin)"
        fi
        ;;
esac

echo
print_header "Próximos Pasos:"
case $option in
    1)
        echo "1. Abrir http://localhost:3000 para usar el predictor"
        echo "2. Ir a Airflow (http://localhost:8080) y ejecutar los DAGs:"
        echo "   - process_etl_stroke_data"
        echo "   - train_and_register_model"
        ;;
    2)
        echo "1. Abrir http://localhost:3000 para probar el frontend"
        echo "2. Si necesitas modelos, ejecutar primero los pipelines ETL"
        ;;
    3)
        echo "1. Ir a Airflow (http://localhost:8080) y ejecutar los DAGs"
        echo "2. Ver resultados en MLflow (http://localhost:5000)"
        ;;
esac

echo
print_header "Comandos Útiles:"
echo "• Ver logs:          docker compose logs -f [servicio]"
echo "• Parar sistema:     docker compose --profile all down"
echo "• Ver estado:        docker compose ps"
echo "• Abrir shell:       docker compose exec [servicio] sh"



# Opción para abrir automáticamente el navegador
if command -v xdg-open &> /dev/null || command -v open &> /dev/null; then
    echo
    read -p "¿Abrir el frontend en el navegador? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:3000
        elif command -v open &> /dev/null; then
            open http://localhost:3000
        fi
    fi
fi

