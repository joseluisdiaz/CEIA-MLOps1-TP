
# Máquina Virtual para VirtualBox

- Link de descarga
	https://drive.google.com/file/d/1pqxlvs36gmqrDSct5KQTMM-NqwxP7en7

Este archivo contiene una máquina virtual con el sistema operativo **Ubuntu 24.04.3 Live Server amd64**, lista para usar y con todos los servicios funcionando.

## Credenciales de acceso
	Usuario (sudo): mlops  
	Contraseña: mlops

## Configuración de red

- La máquina virtual está configurada para obtener una IP automáticamente mediante **DHCP**.
- Para que la VM esté en la misma red que la PC, hay que configurar la placa de red como **adaptador puente**.

## Administración de Docker con Portainer

- Se instaló **Portainer** para la administración de contenedores Docker.
- Acceso a Portainer:  
	URL: `https://<IP del servidor>:9443`  
	Usuario: `mlops`  
	Contraseña: `MlopsMlops1234`

## Directorio del proyecto

- Los archivos del proyecto están en /home/mlops/CEIA-MLOps1-TP. HAcer un git pull para traer las actualizaciones del repositorio 
