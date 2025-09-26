README interno
--------------

Configuración:
-------------
Dentro del "folder" airflow crear los siguientes folders:
  . "config"
  . "logs"
  . "plugins"


Levantar servicio:
-----------------
En la carpeta raíz del repositorio, ejecutar:
docker compose --profile all up


Detener servicio:
----------------
En la carpeta raíz del repositorio, ejecutar:
docker compose --profile all down



Eliminar servicios:
------------------
En la carpeta raíz del repositorio, ejecutar:
docker compose down --rmi all --volumes

--------------------------------------------------------------------------------------------------------------------

MLFlow portal: http://localhost:5001/
MinIO portal: http://localhost:9001/
Airflow portal: http://localhost:8080/
FastAPI: http://localhost:8800/docs#/default/read_root__get

--------------------------------------------------------------------------------------------------------------------

Cada vez que se cambia el modelo champion hay que reiniciar FastAPI para que tome el cambio.

--------------------------------------------------------------------------------------------------------------------

