@echo off
:: Устанавливаем кодовую страницу на UTF-8
chcp 65001 > nul

set TCL_LIBRARY="C:\Python312\tcl\tcl8.6"
set TK_LIBRARY="C:\Python312\tcl\tk8.6"
set PROJ_DATA="C:\Program Files\PostgreSQL\17\share\contrib\postgis-3.5\proj"
set PROJ_LIB="C:\Program Files\PostgreSQL\17\share\contrib\postgis-3.5\proj"
fastapi dev main.py